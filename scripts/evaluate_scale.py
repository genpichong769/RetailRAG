"""Fixed-config scale evaluation. Results are checkpointed; no threshold tuning.

Only evaluation utilities and datasets change. All retrieval and generation use
the existing VectorStore/RAGSystem. k=5 is explicitly diagnostic, k=3/M=2 online.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(4)
from base import Config
from rag_qa.core.vector_store import VectorStore

def load(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def save(name,obj):
    target=ROOT/name
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(target)
def mean(xs): return round(sum(xs)/len(xs),6) if xs else None
def cohort(s): return 'expanded' if s.get('cohort')=='expanded' else 'legacy'

class CachedReranker:
    """Evaluation-only exact-pair memoization; computes missing pairs with the unchanged model.

    Persist each batch so long CPU runs can resume. No cached labels or desired
    predictions are used; cache identity includes local model file metadata.
    """
    def __init__(self,model,cfg):
        self.model=model
        manifest=[]
        for p in sorted(Path(cfg.bge_reranker).glob('*')):
            if p.is_file(): manifest.append((p.name,p.stat().st_size,p.stat().st_mtime_ns))
        self.identity=hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        self.path=ROOT/'RETAILRAG_SCALE_RERANKER_CACHE.json'
        previous=load(self.path.name) if self.path.exists() else {}
        self.values=previous.get('values',{}) if previous.get('model_identity')==self.identity else {}
    def key(self,pair): return hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest()
    def predict(self,pairs,batch_size=8,show_progress_bar=False,**kwargs):
        pairs=list(pairs); missing={self.key(p):p for p in pairs if self.key(p) not in self.values}
        items=list(missing.items())
        for i in range(0,len(items),8):
            block=items[i:i+8]
            values=self.model.predict([p for _,p in block],batch_size=8,show_progress_bar=False,**kwargs)
            self.values.update({k:float(v) for (k,_),v in zip(block,values)})
            save(self.path.name,dict(model_identity=self.identity,pair_count=len(self.values),values=self.values))
            if show_progress_bar: print(f'Rerank new pairs {min(i+8,len(items))}/{len(items)} cached_total={len(self.values)}',flush=True)
        return np.asarray([self.values[self.key(p)] for p in pairs])

def score(s,docs):
    names=[d.metadata.get('document_name') for d in docs]
    expected=set(s['reference_documents'])
    rank=next((i for i,n in enumerate(names,1) if n in expected),None)
    r=dict(id=s['id'],question=s['question'],question_type=s['question_type'],cohort=cohort(s),
           expected_documents=sorted(expected),retrieved_documents=names,
           reciprocal_rank=1/rank if rank else 0.,metadata=[d.metadata for d in docs])
    for k in (1,2,3,5):
        found=expected&set(names[:k])
        r[f'any_at_{k}']=int(bool(found)); r[f'all_at_{k}']=int(expected.issubset(set(names[:k])))
        r[f'document_recall_at_{k}']=len(found)/len(expected) if expected else None
        r[f'context_precision_at_{k}']=len(found)/max(len(names[:k]),1)
    return r

def retrieval_metrics(rows):
    keys=['reciprocal_rank']+[f'{p}_at_{k}' for k in (1,2,3,5) for p in ('any','all','document_recall','context_precision')]
    result={key:mean([r[key] for r in rows if r[key] is not None]) for key in keys}
    result['sample_count']=len(rows); result['mrr']=result['reciprocal_rank']
    result['top1_accuracy']=result['any_at_1']; result['hit_rate']=result['any_at_5']
    return result

def run_faq(store,cfg):
    from evaluate_faq_phase5 import _metrics,_public_metrics
    data=load('RETAILRAG_FAQ_EVALUATION_SET.json')['samples']
    with (ROOT/'mysql_qa/data/retail_faq.csv').open(encoding='utf-8-sig',newline='') as f: faq=list(csv.DictReader(f))
    fq=np.asarray(store.embedding_function([r['问题'] for r in faq])['dense'],dtype=np.float32)
    parts=[]
    for i in range(0,len(data),24):
        parts.append(np.asarray(store.embedding_function([s['query'] for s in data[i:i+24]])['dense'],dtype=np.float32))
        print(f'FAQ embeddings {min(i+24,len(data))}/{len(data)}',flush=True)
    sim=np.vstack(parts)@fq.T; indices=sim.argmax(axis=1); scores=sim[np.arange(len(data)),indices]
    ids=[f'faq_{i+1:03d}' for i in range(len(faq))]
    def aggregate(mask):
        selected=[s for s,m in zip(data,mask) if m]
        result=_public_metrics(_metrics(selected,indices[mask],scores[mask],ids,cfg.FAQ_MATCH_THRESHOLD))
        neg=[i for i,m in enumerate(mask) if m and data[i]['question_type']=='hard_negative']
        result['hard_negative_accuracy']=mean([int(scores[i]<cfg.FAQ_MATCH_THRESHOLD) for i in neg])
        result['sample_count']=len(selected)
        return result
    rows=[]
    for i,s in enumerate(data):
        pred=ids[int(indices[i])] if scores[i]>=cfg.FAQ_MATCH_THRESHOLD else None
        correct=(pred==s['expected_faq_id']) if s['expected_behavior']=='answer' else pred is None
        rows.append(dict(**s,predicted_faq_id=pred,score=float(scores[i]),top_question=faq[int(indices[i])]['问题'],correct=bool(correct)))
    out=dict(status='COMPLETED',threshold=cfg.FAQ_MATCH_THRESHOLD,method=cfg.FAQ_RETRIEVAL_MODE,
             metrics=aggregate(np.ones(len(data),dtype=bool)),
             cohorts={c:aggregate(np.array([cohort(s)==c for s in data])) for c in ('legacy','expanded')},
             results=rows,failures=[r for r in rows if not r['correct']])
    save('RETAILRAG_SCALE_FAQ_RESULTS.json',out)
    print('FAQ',out['metrics'],flush=True)

def rerank_batch(store,sets):
    pairs=[[q,d.page_content] for q,docs in sets for d in docs]
    scores=store.reranker.predict(pairs,batch_size=8,show_progress_bar=True) if pairs else []
    result=[]; offset=0
    for q,docs in sets:
        copied=copy.deepcopy(docs)
        for d,v in zip(copied,scores[offset:offset+len(docs)]): d.metadata['reranker_score']=float(v)
        offset+=len(docs)
        result.append(sorted(copied,key=lambda d:d.metadata['reranker_score'],reverse=True))
    return result

def run_retrieval(store,cfg):
    samples=load('RETAILRAG_SCALE_RETRIEVAL_SET.json')['samples']
    modes={}; candidates=[]
    for name,mode in [('sparse_only','sparse'),('dense_only','dense'),('hybrid','hybrid')]:
        rows=[]
        for i,s in enumerate(samples,1):
            docs=store.search_documents(s['question'],mode=mode,rerank=False,candidate_k=5,limit=5)
            rows.append(score(s,docs))
            if name=='hybrid': candidates.append((s['question'],docs))
            if i%25==0: print(f'{name} {i}/{len(samples)}',flush=True)
        modes[name]=dict(metrics=retrieval_metrics(rows),results=rows)
        save('RETAILRAG_SCALE_RETRIEVAL_RESULTS.json',dict(status='RUNNING',modes=modes))
    ranked=rerank_batch(store,candidates)
    rows=[score(s,docs) for s,docs in zip(samples,ranked)]
    modes['hybrid_reranker']=dict(metrics=retrieval_metrics(rows),results=rows)
    for mode in modes.values():
        rr=mode['results']
        mode['multi_document_metrics']=retrieval_metrics([r for r in rr if len(r['expected_documents'])>1])
        mode['cohorts']={c:retrieval_metrics([r for r in rr if r['cohort']==c]) for c in ('legacy','expanded')}
        mode['failures']=[r for r in rr if not r['all_at_3']]
    gain={k:round(modes['hybrid_reranker']['metrics'][k]-modes['hybrid']['metrics'][k],6) for k in ('any_at_1','any_at_3','mrr','context_precision_at_2')}
    # Same online k=3, M=2; candidate count is not silently enlarged.
    online_candidates=[(s['question'],store.search_documents(s['question'],mode='hybrid',rerank=False,candidate_k=3,limit=3)) for s in samples]
    online_ranked=rerank_batch(store,online_candidates)
    online=[score(s,docs[:2]) for s,docs in zip(samples,online_ranked)]
    output=dict(status='COMPLETED',sample_count=len(samples),candidate_k=5,online_candidate_k=3,online_m=2,
                recall_definition='any_at_k = at least one required doc; document_recall_at_k = fraction of required docs; all_at_k = every required doc',
                modes=modes,reranker_gain=gain,online_metrics=retrieval_metrics(online),online_results=online)
    save('RETAILRAG_SCALE_RETRIEVAL_RESULTS.json',output)
    versions=[]
    for s in samples:
        if s['question_type']!='version_conflict': continue
        docs=store.search_documents(s['question'],candidate_k=3,limit=2)
        row=score(s,docs)
        today=date.today().isoformat()
        row['filter_safe']=all(d.metadata.get('status')=='active' and d.metadata.get('effective_date','2000')<=today<=d.metadata.get('expire_date','9999') for d in docs)
        row['required_context_complete']=bool(row['all_at_2'])
        versions.append(row)
    # Unfiltered audit verifies excluded records really exist in the same collection.
    inventory=store.client.query(collection_name=store.collection_name,filter='',output_fields=['document_name','status','effective_date','expire_date','source','category','version'],limit=5000)
    save('RETAILRAG_SCALE_VERSION_RESULTS.json',dict(status='COMPLETED',sample_count=len(versions),filter_safety=mean([int(r['filter_safe']) for r in versions]),
         all_required_at_2=mean([r['all_at_2'] for r in versions]),results=versions,
         inactive_or_future_records=[r for r in inventory if r['status']!='active' or r['effective_date']>date.today().isoformat()],
         note='Production uses execution date, not time mentioned in question. Future/historical question failures retained.'))
    print('Retrieval', {k:v['metrics'] for k,v in modes.items()},flush=True)

def run_refusal(store,cfg):
    from rag_qa.core.new_rag_system import RAGSystem
    all_samples=load('RETAILRAG_RAG_EVALUATION_SET.json')['samples']
    positives=[s for s in all_samples if s['question_type']=='single_document']
    negatives=[s for s in all_samples if s['question_type']=='unanswerable']
    partial=[s for s in all_samples if s['expected_behavior']=='partial_answer']
    samples=positives+negatives+partial
    sets=[(s['question'],store.search_documents(s['question'],rerank=False,candidate_k=3,limit=3)) for s in samples]
    ranked=rerank_batch(store,sets); rows=[]
    for s,docs in zip(samples,ranked):
        decision=RAGSystem._evidence_decision(docs[:2])
        rows.append(dict(**s,decision=decision,gate_correct=decision['accepted']==(s['expected_behavior']!='reject'),retrieved_documents=[d.metadata.get('document_name') for d in docs[:2]]))
    p=[r for r in rows if r['expected_behavior']=='answer']; n=[r for r in rows if r['expected_behavior']=='reject']
    tp=sum(not r['decision']['accepted'] for r in n); fp=sum(not r['decision']['accepted'] for r in p)
    fn=len(n)-tp
    metrics=dict(answerable_accuracy=mean([int(r['decision']['accepted']) for r in p]),
                 refusal_precision=round(tp/(tp+fp),6) if tp+fp else 0.,refusal_recall=round(tp/len(n),6),
                 false_refusal_rate=round(fp/len(p),6),false_answer_rate=round(fn/len(n),6),
                 true_refusal=tp,false_refusal=fp,false_accept_unanswerable=fn)
    save('RETAILRAG_SCALE_REFUSAL_RESULTS.json',dict(status='COMPLETED',threshold=cfg.EVIDENCE_RERANKER_THRESHOLD,
         sample_count=len(rows),answerable_count=len(p),unanswerable_count=len(n),partial_count=len(partial),metrics=metrics,
         note='These measure fixed code evidence gate, not full-answer safety. Partial cases separately require live generation and judge.',results=rows))
    print('Refusal',metrics,flush=True)

def llm_client(cfg):
    import httpx
    from openai import OpenAI
    return OpenAI(api_key=cfg.DASHSCOPE_API_KEY,base_url=cfg.DASHSCOPE_BASE_URL,http_client=httpx.Client(trust_env=False,timeout=90),max_retries=1)

def stream_function(client,cfg):
    def stream(prompt):
        result=client.chat.completions.create(model=cfg.LLM_MODEL,messages=[{'role':'system','content':'你是 Nova Retail 智能知识助手。'},{'role':'user','content':prompt}],temperature=.1,timeout=90,stream=True)
        for chunk in result:
            if chunk.choices and chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
    return stream

def judge_json(client,cfg,prompt):
    response=client.chat.completions.create(model=cfg.LLM_MODEL,messages=[{'role':'user','content':prompt}],temperature=0.,timeout=90)
    text=response.choices[0].message.content
    return json.loads(re.search(r'\{.*\}',text,re.S).group(0))

def run_rewrite(store,cfg,rag,client):
    samples=[s for s in load('RETAILRAG_RAG_EVALUATION_SET.json')['samples'] if s['question_type']=='multiturn']
    rows=[]; sets=[]
    for i,s in enumerate(samples,1):
        rewritten=rag.rewrite_query(s['question'],s['history'])
        row=dict(**s,rewritten_query=rewritten)
        try:
            row['semantic_judge']=judge_json(client,cfg,f'''判断改写是否保留原对话追问的主体、时间、否定、数量与约束，不凭参考答案评价。不能添加未给定事实。返回JSON {{"error":true或false,"reason":"理由"}}。
历史：{json.dumps(s['history'],ensure_ascii=False)}
追问：{s['question']}
改写：{rewritten}''')
        except Exception as e: row['judge_error']=str(e)
        for q in (s['question'],rewritten): sets.append((q,store.search_documents(q,rerank=False,candidate_k=5,limit=5)))
        rows.append(row); print(f'Rewrite {i}/{len(samples)}',flush=True)
    ranked=rerank_batch(store,sets)
    for i,(s,r) in enumerate(zip(samples,rows)):
        r['without_rewrite']=score(s,ranked[i*2]); r['with_rewrite']=score(s,ranked[i*2+1])
        r['retrieval_regression']=bool(r['without_rewrite']['any_at_3'] and not r['with_rewrite']['any_at_3'])
    judged=[r for r in rows if 'semantic_judge' in r]
    out=dict(status='COMPLETED' if len(judged)==len(rows) else 'PARTIAL',sample_count=len(rows),candidate_k=5,
             without=retrieval_metrics([r['without_rewrite'] for r in rows]),with_rewrite=retrieval_metrics([r['with_rewrite'] for r in rows]),
             rewrite_error_rate=mean([int(r['semantic_judge']['error']) for r in judged]),judge_count=len(judged),
             retrieval_regression_rate=mean([int(r['retrieval_regression']) for r in rows]),results=rows,
             limitation='Semantic rewrite errors judged by same Qwen family; retrieval regression is a separate automatic observable.')
    save('RETAILRAG_SCALE_REWRITE_RESULTS.json',out)

def generation_selection():
    samples=load('RETAILRAG_RAG_EVALUATION_SET.json')['samples']
    counts=dict(single_document=10,multi_document=15,keyword_exact=5,semantic_paraphrase=5,version_conflict=10,unanswerable=15,multiturn=10,hard_confusion=10)
    selected=[]
    for kind,n in counts.items():
        candidates=[s for s in samples if s['question_type']==kind]
        # Additions first, then legacy: selection is fixed, no picking by observed results.
        candidates=sorted(candidates,key=lambda s:(s['expected_behavior']!='partial_answer',cohort(s)!='expanded',s['id']))
        selected.extend(candidates[:n])
    assert len(selected)==80
    return selected

def run_generation(store,cfg,rag,client):
    selected=generation_selection()
    save('RETAILRAG_SCALE_GENERATION_SELECTION.json',dict(sample_count=80,samples=selected))
    path=ROOT/'RETAILRAG_SCALE_GENERATION_RESULTS.json'
    rows=load(path.name).get('results',[]) if path.exists() else []
    done={r['id'] for r in rows}
    for i,s in enumerate(selected,1):
        if s['id'] in done: continue
        row=dict(**s)
        try:
            row['answer']=''.join(rag.generate_answer(s['question'],history=s.get('history',[])))
            trace=copy.deepcopy(rag.last_generation_trace or {})
            row['trace']=trace
            context=trace.get('final_context','')
            row['scores']=judge_json(client,cfg,f'''你是严格的RAG评测裁判。仅按给出的原始证据与参考判断，不利用外部知识替缺失证据补全。每项0到1。
faithfulness：回答事实由实际检索上下文支持的比例；不得因内容听起来合理而给满分。
answer_relevance：是否回应用户全部可答条件；遗漏、错误拒答、回避需要扣分。
context_precision：实际上下文与所问直接相关的比例；不能因为有一条相关就给全部满分。
context_recall：参考答案所需事实在实际上下文中的覆盖比例；不能用回答反推上下文已包含。
纯无答案且正确拒答时faithfulness与relevance可为1；检索到无关上下文应降低precision。
partial_answer必须回答有证据部分，并明确不能确认未知部分；把整个问题都拒答不算完整。
另外判断refused（是否明确拒绝回答缺乏证据部分），unsupported_specifics（是否编造未知具体事实），partial_handled（仅用于部分可回答）。
返回JSON：{{"faithfulness":0.0,"answer_relevance":0.0,"context_precision":0.0,"context_recall":0.0,"refused":false,"unsupported_specifics":false,"partial_handled":false,"rationale":"具体说明缺失或错误"}}
历史：{json.dumps(s.get('history',[]),ensure_ascii=False)}
问题：{s['question']}
期望行为：{s['expected_behavior']}
参考：{s['ground_truth']}
实际检索上下文：{context or '无上下文'}
实际回答：{row['answer']}''')
            for key in ('faithfulness','answer_relevance','context_precision','context_recall'):
                row['scores'][key]=max(0.,min(1.,float(row['scores'][key])))
        except Exception as e: row['error']=str(e)
        rows.append(row)
        valid=[r for r in rows if 'scores' in r]
        metrics={key:mean([r['scores'][key] for r in valid]) for key in ('faithfulness','answer_relevance','context_precision','context_recall')}
        neg=[r for r in valid if r['expected_behavior']=='reject']; part=[r for r in valid if r['expected_behavior']=='partial_answer']
        out=dict(status='COMPLETED' if len(rows)==80 and len(valid)==80 else 'RUNNING_OR_PARTIAL',sample_count=len(rows),successfully_judged=len(valid),metrics=metrics,
                 unanswerable_refusal_accuracy=mean([int(r['scores'].get('refused',False) and not r['scores'].get('unsupported_specifics',False)) for r in neg]),
                 partial_answer_accuracy=mean([int(r['scores'].get('partial_handled',False)) for r in part]),
                 generator_model=cfg.LLM_MODEL,judge_model=cfg.LLM_MODEL,same_family_optimism_warning=True,
                 results=rows)
        save(path.name,out)
        print(f'Generation {i}/80 {s["id"]}: {row.get("scores",row.get("error"))}',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stages',default='faq,retrieval,refusal,rewrite,generation'); args=parser.parse_args()
    cfg=Config()
    assert (cfg.PARENT_CHUNK_SIZE,cfg.CHILD_CHUNK_SIZE,cfg.CHUNK_OVERLAP,cfg.RETRIEVAL_K,cfg.CANDIDATE_M)==(1200,300,50,3,2)
    assert cfg.FAQ_MATCH_THRESHOLD==.84 and cfg.EVIDENCE_RERANKER_THRESHOLD==.020635
    audit=load('RETAILRAG_SCALE_DATA_AUDIT.json')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in audit['sha256'].items()),'Dataset changed after freeze'
    store=VectorStore(); store.reranker=CachedReranker(store.reranker,cfg); stages=args.stages.split(',')
    for stage in stages:
        print('START',stage,flush=True)
        if stage=='faq': run_faq(store,cfg)
        elif stage=='retrieval': run_retrieval(store,cfg)
        elif stage=='refusal': run_refusal(store,cfg)
        else:
            from rag_qa.core.new_rag_system import RAGSystem
            client=llm_client(cfg); rag=RAGSystem(store,stream_function(client,cfg))
            if stage=='rewrite': run_rewrite(store,cfg,rag,client)
            elif stage=='generation': run_generation(store,cfg,rag,client)
        print('DONE',stage,flush=True)

if __name__=='__main__': main()
