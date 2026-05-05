import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function Inbox() {
  const [contacts, setContacts] = useState([]);
  const [selectedContact, setSelectedContact] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const token = localStorage.getItem('lex_token');
  const currentUser = JSON.parse(localStorage.getItem('lex_user'));

  useEffect(() => {
    if (!token) {
        navigate('/auth');
        return;
    }
    fetchContacts();
  }, [navigate]);

  useEffect(() => {
    if (selectedContact) {
      fetchMessages(selectedContact.id);
    }
  }, [selectedContact]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchContacts = async () => {
    try {
      const res = await fetch('http://localhost:8000/auth/messages/contacts', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setContacts(data);
      if (data.length > 0) {
        setSelectedContact(data[0]);
      }
      setLoading(false);
    } catch (err) {
      console.error("Failed to fetch contacts", err);
      setLoading(false);
    }
  };

  const fetchMessages = async (contactId) => {
    try {
      const res = await fetch(`http://localhost:8000/auth/messages/${contactId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setMessages(data);
    } catch (err) {
      console.error("Failed to fetch messages", err);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !selectedContact) return;
    
    const newMsgContent = input;
    setInput('');
    
    // Optimistic update
    const tempMsg = {
        id: Date.now(),
        sender_id: -1, // will be replaced
        receiver_id: selectedContact.id,
        content: newMsgContent,
        timestamp: new Date().toISOString()
    };
    setMessages([...messages, tempMsg]);

    try {
      const res = await fetch('http://localhost:8000/auth/messages', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          receiver_id: selectedContact.id,
          content: newMsgContent
        })
      });
      
      const savedMsg = await res.json();
      setMessages(prev => prev.map(m => m.id === tempMsg.id ? savedMsg : m));
    } catch (err) {
      console.error("Failed to send message", err);
    }
  };

  if (loading) {
      return <div className="container text-center">Loading Inbox...</div>;
  }

  return (
    <div className="container" style={{ display: 'flex', height: '85vh', gap: '2rem' }}>
      
      {/* Sidebar for Contacts */}
      <div className="glass-panel" style={{ width: '300px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h3 style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', margin: 0 }}>Conversations</h3>
          <div style={{ flex: 1, overflowY: 'auto' }}>
              {contacts.length === 0 ? (
                  <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                      No conversations yet.
                  </div>
              ) : (
                  contacts.map(c => (
                      <div 
                        key={c.id} 
                        onClick={() => setSelectedContact(c)}
                        style={{ 
                            padding: '1.5rem', 
                            borderBottom: '1px solid rgba(255,255,255,0.05)', 
                            cursor: 'pointer',
                            background: selectedContact?.id === c.id ? 'rgba(255,255,255,0.1)' : 'transparent',
                            transition: 'background 0.3s'
                        }}
                      >
                          <div style={{ fontWeight: 'bold' }}>{c.name}</div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--accent-gold)' }}>
                              {c.role === 'lawyer' ? 'Advocate' : 'Client'}
                          </div>
                      </div>
                  ))
              )}
          </div>
      </div>

      {/* Main Chat Area */}
      <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {selectedContact ? (
              <>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ 
                        width: '40px', height: '40px', borderRadius: '50%', 
                        background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold'
                    }}>
                        {selectedContact.name.charAt(0).toUpperCase()}
                    </div>
                    <div>
                        <h3 style={{ margin: 0 }}>{selectedContact.name}</h3>
                    </div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {messages.length === 0 ? (
                        <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '2rem' }}>
                            Start the conversation by sending a message!
                        </div>
                    ) : (
                        messages.map((msg) => {
                            const isMe = msg.receiver_id === selectedContact.id;
                            return (
                                <div key={msg.id} style={{
                                    alignSelf: isMe ? 'flex-end' : 'flex-start',
                                    maxWidth: '75%',
                                    background: isMe ? 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))' : 'rgba(255,255,255,0.1)',
                                    padding: '12px 18px',
                                    borderRadius: '16px',
                                    borderBottomRightRadius: isMe ? '4px' : '16px',
                                    borderBottomLeftRadius: !isMe ? '4px' : '16px',
                                }}>
                                    {msg.content}
                                </div>
                            );
                        })
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border)' }}>
                    <div className="flex gap-4" style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '16px' }}>
                        <input 
                            style={{ margin: 0, flex: 1, border: 'none', background: 'transparent' }}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyPress={e => e.key === 'Enter' && handleSend()}
                            placeholder="Type a message..."
                        />
                        <button className="btn-primary" onClick={handleSend} style={{ borderRadius: '50%', width: '50px', height: '50px', padding: 0 }}>
                            ➤
                        </button>
                    </div>
                </div>
              </>
          ) : (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                  Select a conversation to start chatting.
              </div>
          )}
      </div>

    </div>
  );
}

export default Inbox;
