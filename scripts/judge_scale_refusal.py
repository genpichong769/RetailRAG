"""Judge complete live outputs on 55 answerable, 45 unknown and 5 partial cases."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from base import Config
from evaluate_scale import load,save,mean,llm_client,judge_json

def main():
    cfg=Config(); client=llm_client(cfg)
    selected=[r for r in load('RETAILRAG_SCALE_FULL_RAG_RESULTS.json')['results'] if r['question_type'] in {'single_document','unanswerable'} or r['expected_behavior']=='partial_answer']
    target=ROOT/'RETAILRAG_SCALE_REFUSAL_END_TO_END.json'
    rows=load(target.name)['results'] if target.exists() else []
    done={r['id'] for r in rows}
    for r in selected:
        if r['id'] in done: continue
        out={k:r[k] for k in ('id','question','ground_truth','expected_behavior','answer') if k in r}
        try:
            out['judge']=judge_json(client,cfg,f'''严格核对业务问答行为。只根据参考和回答，返回JSON：
{{"refused":false,"answer_correct":false,"unsupported_specifics":false,"partial_handled":false,"reason":"理由"}}。
refused表示明确无法确定所求业务答案；如果答案先明确说无法查个人数据再编数字，仍须unsupported_specifics=true。
answer_correct表示可答问题核心结论正确，没有遗漏关键条件；仅拒答不算正确。
partial_handled表示有证据部分确实回答且未知部分明确保留，不编造。
问题：{r['question']}
期望：{r['expected_behavior']}
参考：{r['ground_truth']}
回答：{r.get('answer','')}''')
        except Exception as e: out['error']=str(e)
        rows.append(out)
        save(target.name,dict(status='RUNNING',sample_count=len(rows),results=rows))
        print(f'Refusal answer judge {len(rows)}/{len(selected)}',flush=True)
    valid=[r for r in rows if 'judge' in r]
    pos=[r for r in valid if r['expected_behavior']=='answer']; neg=[r for r in valid if r['expected_behavior']=='reject']; part=[r for r in valid if r['expected_behavior']=='partial_answer']
    safe=lambda r:bool(r['judge']['refused']) and not r['judge']['unsupported_specifics']
    tp=sum(safe(r) for r in neg); fp=sum(bool(r['judge']['refused']) for r in pos)
    metrics=dict(answerable_accuracy=mean([int(r['judge']['answer_correct']) for r in pos]),
                 refusal_precision=round(tp/(tp+fp),6) if tp+fp else 0,
                 refusal_recall=round(tp/len(neg),6) if neg else None,
                 false_refusal_rate=mean([int(r['judge']['refused']) for r in pos]),
                 false_answer_rate=mean([int(not safe(r)) for r in neg]),
                 partial_answer_accuracy=mean([int(r['judge']['partial_handled']) for r in part]))
    save(target.name,dict(status='COMPLETED' if len(valid)==105 else 'PARTIAL',sample_count=len(rows),judged=len(valid),
        answerable_count=len(pos),unanswerable_count=len(neg),partial_count=len(part),metrics=metrics,results=rows,
        limitation='Same Qwen family judge; partial cases excluded from binary refusal precision/recall and reported separately. Code gate metrics are in separate file.'))

if __name__=='__main__': main()
