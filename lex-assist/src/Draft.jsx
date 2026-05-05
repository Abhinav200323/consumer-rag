import React, { useState } from 'react';

function Draft() {
  const [docType, setDocType] = useState('Legal Notice to Seller');
  const [facts, setFacts] = useState('');
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);

  const handleDraft = async () => {
    if (!facts.trim()) return;
    setLoading(true);
    setDraft('');
    
    try {
      const res = await fetch('http://localhost:8000/draft_document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_type: docType, facts })
      });
      
      const data = await res.json();
      setDraft(data.draft);
    } catch (err) {
      setDraft('Error generating draft. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ display: 'flex', gap: '2rem' }}>
      <div className="glass-panel" style={{ flex: 1, padding: '2rem' }}>
        <h2 className="mb-4">Draft Legal Document</h2>
        
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Document Type</label>
        <select value={docType} onChange={e => setDocType(e.target.value)}>
          <option>Legal Notice to Seller</option>
          <option>Consumer Complaint (District Commission)</option>
          <option>RTI Application</option>
        </select>

        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Facts of the Case</label>
        <textarea 
          rows={6}
          placeholder="E.g., I bought a TV on 1st Jan from XYZ electronics. It stopped working after 10 days. The seller refused to repair it under warranty."
          value={facts}
          onChange={e => setFacts(e.target.value)}
        />

        <button className="btn-primary" onClick={handleDraft} disabled={loading} style={{ width: '100%' }}>
          {loading ? 'Drafting...' : 'Generate Draft'}
        </button>
      </div>

      <div className="glass-panel" style={{ flex: 1.5, padding: '2rem', overflowY: 'auto', maxHeight: '80vh' }}>
        <h2 className="mb-4">Generated Draft</h2>
        {draft ? (
          <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px' }}>
            {draft}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Your generated legal document will appear here.</p>
        )}
      </div>
    </div>
  );
}

export default Draft;
