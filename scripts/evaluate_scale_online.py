"""HTTP/WebSocket regression against unchanged running app, retain failed cases."""
import json
import sys
import uuid
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from test_api_online import stream_query

def main():
    rows=[]
    with httpx.Client(trust_env=False,timeout=180) as c:
        for name,path in [('health','/health'),('page','/')]:
            r=c.get('http://127.0.0.1:8003'+path)
            rows.append(dict(case=name,pass_check=r.status_code==200,status=r.status_code))
        for name,q,stream in [
            ('legacy_faq','会员等级怎么划分？',False),
            ('new_faq','银卡会员有免费配送次数吗？',False),
            ('general_faq_boundary','法国的首都是什么？',True),
            ('cross_document_faq_boundary','金卡在生日月买了100元符合积分条件商品，不参加其他活动，基础和生日积分一共多少？',True),
            ('multi_condition_faq_boundary','闭店前现金、设备和库存分别要检查什么？',True),
        ]:
            r=c.post('http://127.0.0.1:8003/api/query',json={'query':q,'session_id':str(uuid.uuid4())})
            payload=r.json(); rows.append(dict(case=name,query=q,response=payload,pass_check=r.status_code==200 and payload.get('is_streaming')==stream))
    def ws(name,q,check,session=None):
        try:
            r=stream_query(q,session or str(uuid.uuid4()))
            r.update(case=name,transport_pass=r['end_received'] and not r['error_events'],pass_check=bool(check(r['answer'])) and r['end_received'] and not r['error_events'])
        except Exception as e: r=dict(case=name,query=q,pass_check=False,error=str(e))
        rows.append(r); persist(rows); print(name,r['pass_check'],flush=True)
    ws('rag_and_citation','顾客买的商品想退货',lambda a:'来源' in a and 'return_exchange' in a)
    ws('general','法国的首都是什么？',lambda a:'巴黎' in a)
    ws('version','现在金卡积分怎么算？',lambda a:'1.5' in a)
    ws('refusal','Nova Retail 今年营业收入多少？',lambda a:'信息不足' in a or '无法' in a)
    session=str(uuid.uuid4())
    ws('multiturn_start','顾客买的商品想退货',lambda a:bool(a),session)
    ws('multiturn_followup','买了5天，有小票，包装完整也没用过',lambda a:'退' in a and ('7' in a or '七' in a),session)
    with httpx.Client(trust_env=False,timeout=30) as c:
        r=c.get(f'http://127.0.0.1:8003/api/history/{session}')
        history=r.json().get('history',[])
        rows.append(dict(case='history_persisted',pass_check=r.status_code==200 and len(history)==2,history=history))
    persist(rows,True)

def persist(rows,completed=False):
    out=dict(status=('PASS' if all(r['pass_check'] for r in rows) else 'FAIL') if completed else 'RUNNING',passed=sum(bool(r['pass_check']) for r in rows),total=len(rows),results=rows)
    (ROOT/'RETAILRAG_SCALE_ONLINE_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
