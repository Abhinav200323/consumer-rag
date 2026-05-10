import React, { useState, useRef, useEffect } from 'react';

function Draft() {
  const [docType, setDocType] = useState('Legal Notice to Seller');
  const [facts, setFacts] = useState('');
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [refineInput, setRefineInput] = useState('');
  const draftEndRef = useRef(null);

  const handleDraft = async (customFacts = null) => {
    const factsToUse = customFacts || facts;
    if (!factsToUse.trim()) return;
    
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/draft_document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            document_type: docType, 
            facts: factsToUse 
        })
      });
      
      const data = await res.json();
      setDraft(data.draft);
      setRefineInput('');
    } catch (err) {
      alert('Error generating draft. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRefine = () => {
    if (!refineInput.trim()) return;
    const newContext = `Previous Draft:\n${draft}\n\nUpdate Request: ${refineInput}\n\nPlease regenerate the draft with these updates included.`;
    handleDraft(newContext);
  };

  const downloadDoc = async () => {
    if (!draft) return;
    try {
      const res = await fetch('http://localhost:8000/download_draft', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: draft })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${docType.replace(/\s+/g, '_')}_draft.docx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert("Failed to download document");
    }
  };

  return (
    <div className="container" style={{ display: 'flex', gap: '2rem', height: '80vh' }}>
      
      {/* Left Panel: Input & Settings */}
      <div className="glass-panel" style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column' }}>
        <h2 className="mb-4 text-gradient">Generate Legal Draft</h2>
        
        <div style={{ flex: 1 }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Select Document Type</label>
            <select value={docType} onChange={e => setDocType(e.target.value)}>
                <option>Legal Notice to Seller</option>
                <option>Consumer Complaint (District Commission)</option>
                <option>RTI Application</option>
                <option>Notice for Refund of Money</option>
            </select>

            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', marginTop: '1rem' }}>Facts of the Case</label>
            <textarea 
                rows={10}
                style={{ resize: 'none' }}
                placeholder="Describe your situation in detail. E.g., Date of purchase, the defect found, your interaction with the seller, etc."
                value={facts}
                onChange={e => setFacts(e.target.value)}
            />
        </div>

        <button className="btn-primary" onClick={() => handleDraft()} disabled={loading} style={{ width: '100%', marginTop: '1rem' }}>
          {loading ? 'Processing...' : draft ? 'Regenerate Base Draft' : 'Generate Initial Draft'}
        </button>
      </div>

      {/* Right Panel: The Draft & Refinement Chat */}
      <div className="glass-panel" style={{ flex: 1.5, padding: '2rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="flex justify-between items-center mb-4">
            <h2 style={{ margin: 0 }}>Final Draft</h2>
            {draft && (
                <button className="btn-outline" onClick={downloadDoc} style={{ padding: '8px 16px', fontSize: '0.8rem', borderColor: 'var(--accent-gold)', color: 'var(--accent-gold)' }}>
                    📥 Download .DOCX
                </button>
            )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', background: 'rgba(0,0,0,0.3)', padding: '1.5rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)', marginBottom: '1rem' }}>
          {draft ? (
            <div style={{ whiteSpace: 'pre-wrap', fontFamily: "'Courier New', Courier, monospace", lineHeight: '1.6', color: '#e2e8f0' }}>
              {draft}
              <div ref={draftEndRef} />
            </div>
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>
                <p>Provide the facts on the left to generate your legal document.<br/>You can refine it via chat once generated.</p>
            </div>
          )}
        </div>

        {/* Refinement Chat Window */}
        {draft && (
            <div className="flex gap-2" style={{ background: 'rgba(255,255,255,0.1)', padding: '10px', borderRadius: '12px' }}>
                <input 
                    style={{ margin: 0, border: 'none', background: 'transparent', flex: 1 }}
                    placeholder="Ask Lex to change something... (e.g. 'Add my address: 123 Street')"
                    value={refineInput}
                    onChange={e => setRefineInput(e.target.value)}
                    onKeyPress={e => e.key === 'Enter' && handleRefine()}
                />
                <button className="btn-primary" onClick={handleRefine} disabled={loading} style={{ padding: '8px 20px', borderRadius: '8px' }}>
                    Refine
                </button>
            </div>
        )}
      </div>
    </div>
  );
}

export default Draft;
