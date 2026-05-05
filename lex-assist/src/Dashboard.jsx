import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function Dashboard() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('lex_token');
    if (!token) navigate('/auth');
    
    // Welcome message from friendly bot
    setMessages([
        { role: 'assistant', content: "Hi there! 👋 I'm Lex, your friendly AI legal assistant. How can I help you with Indian Consumer Law today?" }
    ]);
  }, [navigate]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages([...messages, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          use_llm_expansion: true,
          claim_value: 0,
          preferred_language: 'English',
          filters: {}
        })
      });
      
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Oops! I had trouble connecting to the legal database. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ display: 'flex', flexDirection: 'column', height: '90vh', maxWidth: '900px' }}>
      
      <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', padding: '1.5rem', marginBottom: '1rem' }}>
        <div style={{ fontSize: '2.5rem', marginRight: '1rem', background: 'white', borderRadius: '50%', padding: '10px', boxShadow: '0 4px 15px rgba(0,0,0,0.2)' }}>
            🤖
        </div>
        <div>
            <h2 style={{ margin: 0 }}>Lex Assistant</h2>
            <p style={{ margin: 0, color: 'var(--accent-gold)', fontWeight: 'bold' }}>Online & Ready to Help</p>
        </div>
      </div>

      <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '1.5rem', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: '10px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          {messages.map((msg, i) => (
            <div key={i} style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              display: 'flex',
              flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
              gap: '12px',
              alignItems: 'flex-end'
            }}>
              {msg.role === 'assistant' && <div style={{ fontSize: '1.5rem' }}>⚖️</div>}
              <div className={msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'} style={{ whiteSpace: 'pre-wrap' }}>
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
              <div style={{ alignSelf: 'flex-start', display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
                  <div style={{ fontSize: '1.5rem' }}>⚖️</div>
                  <div className="chat-bubble-ai" style={{ fontStyle: 'italic', color: '#64748b' }}>Lex is analyzing the law...</div>
              </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-4" style={{ marginTop: '1.5rem', background: 'rgba(0,0,0,0.15)', padding: '10px', borderRadius: '16px' }}>
          <input 
            style={{ margin: 0, flex: 1, border: 'none', background: 'transparent' }}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && handleSend()}
            placeholder="E.g., Can I return a defective product if the seller has a 'no return' policy?"
          />
          <button className="btn-primary" onClick={handleSend} disabled={loading} style={{ borderRadius: '50%', width: '50px', height: '50px', padding: 0 }}>
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
