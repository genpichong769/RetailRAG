"""Execute all 350 fixed questions through the existing RAGSystem.

Reuse the 80 real generation outputs, then complete every remaining question.
Automatic context coverage is not represented as semantic answer accuracy.
"""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from evaluate_scale import load,save,mean,llm_client,stream_function,CachedReranker
from base import Config
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore

def main():
    cfg=Config(); samples=load('RETAILRAG_RAG_EVALUATION_SET.json')['samples']
    generated=load('RETAILRAG_SCALE_GENERATION_RESULTS.json')['results']
    path=ROOT/'RETAILRAG_SCALE_FULL_RAG_RESULTS.json'
    rows=load(path.name)['results'] if path.exists() else copy.deepcopy(generated)
    done={r['id'] for r in rows}; store=VectorStore(); store.reranker=CachedReranker(store.reranker,cfg)
    rag=RAGSystem(store,stream_function(llm_client(cfg),cfg))
    for i,s in enumerate(samples,1):
        if s['id'] in done: continue
        r=dict(**s)
        try:
            r['answer']=''.join(rag.generate_answer(s['question'],history=s.get('history',[])))
            r['trace']=copy.deepcopy(rag.last_generation_trace or {})
        except Exception as e: r['error']=str(e)
        rows.append(r)
        enrich(rows)
        save(path.name,dict(status='COMPLETED' if len(rows)==350 else 'RUNNING',sample_count=len(rows),results=rows,metrics=summarize(rows)))
        print(f'Full RAG completed {len(rows)}/350 {s["id"]}',flush=True)
    enrich(rows)
    save(path.name,dict(status='COMPLETED',sample_count=len(rows),results=rows,metrics=summarize(rows),
        note='350 real RAGSystem executions; 80 selected outputs have independent calls to same-family grounded judge. Automatic reference coverage and gate decisions are not semantic answer accuracy.'))

def enrich(rows):
    for r in rows:
        trace=r.get('trace',{}); refs=set(trace.get('references',[])); expected=set(r.get('reference_documents',[]))
        r['any_reference_hit']=bool(expected&refs) if expected else None
        r['all_required_covered']=expected.issubset(refs) if expected else None
        r['code_refusal']=trace.get('llm_called') is False and not trace.get('evidence_decision',{}).get('accepted',True)
        r['citation_present']=any(n in r.get('answer','') for n in refs) if refs else None

def summarize(rows):
    positive=[r for r in rows if r['expected_behavior']!='reject']
    negative=[r for r in rows if r['expected_behavior']=='reject']
    return dict(execution_errors=sum('error' in r for r in rows),answers_returned=sum(bool(r.get('answer')) for r in rows),
                required_doc_any_hit=mean([int(r['any_reference_hit']) for r in positive]),
                all_required_covered=mean([int(r['all_required_covered']) for r in positive]),
                answerable_code_false_refusal_rate=mean([int(r['code_refusal']) for r in positive]),
                unanswerable_code_refusal_rate=mean([int(r['code_refusal']) for r in negative]),
                citation_presence=mean([int(r['citation_present']) for r in rows if r['citation_present'] is not None]))

if __name__=='__main__':
    main()
    from judge_scale_refusal import main as judge_refusal
    judge_refusal()
