import React, { useEffect, useRef, useState } from 'react';

const SOURCE_LABELS = {
  membership: '会员制度',
  promotion: '促销活动',
  after_sales: '售后政策',
  store_operation: '门店运营',
  product: '商品知识',
  equipment: '设备操作',
  policy: '企业制度',
  notice: '内部通知',
};

const EXAMPLES = [
  '会员积分什么时候过期？',
  '促销订单部分退货怎么处理？',
  '闭店时现金和库存要检查什么？',
  'POS 支付结果不明怎么办？',
];

const App = () => {
  const [query, setQuery] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [answer, setAnswer] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const answerRef = useRef(null);

  useEffect(() => {
    fetch('/api/create_session', { method: 'POST' })
      .then((response) => response.json())
      .then((data) => setSessionId(data.session_id))
      .catch((error) => console.error('初始化会话失败:', error));
  }, []);

  useEffect(() => {
    if (answerRef.current) answerRef.current.scrollTop = answerRef.current.scrollHeight;
  }, [answer]);

  const refreshHistory = async () => {
    if (!sessionId) return;
    const response = await fetch(`/api/history/${sessionId}`);
    const data = await response.json();
    setHistory(data.history || []);
  };

  const streamAnswer = (payload) => new Promise((resolve, reject) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/api/stream`);
    socket.onopen = () => socket.send(JSON.stringify(payload));
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'token') setAnswer((current) => current + message.token);
      if (message.type === 'end') {
        socket.close();
        resolve();
      }
      if (message.type === 'error') {
        socket.close();
        reject(new Error(message.error));
      }
    };
    socket.onerror = () => reject(new Error('WebSocket 连接失败'));
  });

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || !sessionId || isLoading) return;
    setIsLoading(true);
    setAnswer('');
    const payload = { query: query.trim(), source_filter: sourceFilter || null, session_id: sessionId };
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (data.is_streaming) await streamAnswer(payload);
      else setAnswer(data.answer || '未找到答案');
      await refreshHistory();
      setQuery('');
    } catch (error) {
      console.error('查询失败:', error);
      setAnswer('当前无法获取回答，请稍后重试。');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (!sessionId) return;
    await fetch(`/api/history/${sessionId}`, { method: 'DELETE' });
    setHistory([]);
    setAnswer('');
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-center">RetailRAG · Nova Retail 智能知识助手</h1>
      <p className="text-center text-gray-600 mb-6">连锁零售内部知识问答演示系统；内容不代表真实企业承诺。</p>

      <div className="flex flex-wrap gap-2 mb-4">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" onClick={() => setQuery(example)} className="text-sm bg-gray-100 px-3 py-1 rounded">
            {example}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg shadow-md mb-6">
        <label className="block text-sm font-medium text-gray-700">问题</label>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：无小票退货需要什么凭证？" disabled={isLoading} className="mt-1 mb-4 block w-full rounded-md border-gray-300" />
        <label className="block text-sm font-medium text-gray-700">知识类别（可选）</label>
        <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value)} disabled={isLoading} className="mt-1 mb-4 block w-full rounded-md border-gray-300">
          <option value="">全部类别</option>
          {Object.entries(SOURCE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <button type="submit" disabled={isLoading || !sessionId} className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md disabled:bg-gray-400">
          {isLoading ? '处理中…' : '提交'}
        </button>
      </form>

      {answer && <section ref={answerRef} className="bg-white p-6 rounded-lg shadow-md mb-6 whitespace-pre-wrap"><h2 className="font-semibold mb-2">回答</h2>{answer}</section>}
      {history.length > 0 && <section className="bg-white p-6 rounded-lg shadow-md"><div className="flex justify-between"><h2 className="font-semibold">最近对话</h2><button type="button" onClick={handleClearHistory}>清除</button></div>{history.map((item, index) => <div key={index} className="mt-3"><p>问：{item.question}</p><p className="text-gray-600">答：{item.answer}</p></div>)}</section>}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
