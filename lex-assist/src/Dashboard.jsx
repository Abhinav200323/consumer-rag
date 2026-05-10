import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function Dashboard() {
  const [conversations, setConversations] = useState([]);
  const [activeConv, setActiveConv] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const token = localStorage.getItem('lex_token');

  useEffect(() => {
    if (!token) {
        navigate('/auth');
        return;
    }
    fetchConversations();
  }, [navigate, token]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchConversations = async () => {
    try {
      const res = await fetch('http://localhost:8000/auth/conversations', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setConversations(data);
      if (data.length > 0 && !activeConv) {
        selectConversation(data[0].id);
      } else if (data.length === 0) {
        startNewChat();
      }
    } catch (err) {
      console.error("Failed to fetch conversations", err);
    }
  };

  const selectConversation = async (id) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/auth/conversations/${id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setActiveConv(id);
      setMessages(data.messages);
    } catch (err) {
      console.error("Failed to fetch history", err);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/auth/conversations', {
        method: 'POST',
        headers: { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json' 
        }
      });
      const data = await res.json();
      setConversations([data, ...conversations]);
      setActiveConv(data.id);
      setMessages([
        { role: 'assistant', content: "Hi there! 👋 I'm Lex. How can I help you with Indian Consumer Law today?" }
      ]);
    } catch (err) {
      console.error("Failed to start new chat", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !activeConv) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
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
          filters: { conversation_id: activeConv }
        })
      });
      
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
      
      // Refresh sidebar to update titles if needed
      fetchConversations();
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Oops! I had trouble connecting. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ display: 'flex', height: '85vh', maxWidth: '1200px', gap: '20px' }}>
      
      {/* Sidebar - History */}
      <div className="glass-panel" style={{ width: '300px', display: 'flex', flexDirection: 'column', padding: '1rem' }}>
        <button className="btn-primary" onClick={startNewChat} style={{ marginBottom: '1rem', width: '100%' }}>
            + New Chat
        </button>
        
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {conversations.map(conv => (
                <div 
                    key={conv.id} 
                    onClick={() => selectConversation(conv.id)}
                    style={{
                        padding: '12px',
                        borderRadius: '12px',
                        background: activeConv === conv.id ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)',
                        cursor: 'pointer',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s',
                        border: activeConv === conv.id ? '1px solid var(--accent-gold)' : '1px solid transparent'
                    }}
                >
                    <div style={{ fontWeight: 'bold', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {conv.title}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                        {new Date(conv.created_at).toLocaleDateString()}
                    </div>
                </div>
            ))}
        </div>

        <div style={{ marginTop: '1rem', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '10px', fontSize: '0.75rem', color: 'var(--accent-gold)' }}>
            ℹ️ Chats are stored for 1 day by default (max 7 days in settings).
        </div>
      </div>

      {/* Main Chat Area */}
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
                  <div className="chat-bubble-ai" style={{ fontStyle: 'italic', color: '#64748b' }}>Lex is analyzing...</div>
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
            placeholder="Type your legal question..."
          />
          <button className="btn-primary" onClick={handleSend} disabled={loading || !activeConv} style={{ borderRadius: '50%', width: '50px', height: '50px', padding: 0 }}>
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
