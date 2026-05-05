import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function HireLawyer() {
  const [lawyers, setLawyers] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetch('http://localhost:8000/auth/lawyers')
      .then(res => res.json())
      .then(data => {
        setLawyers(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const handleHire = async (lawyerId) => {
    try {
      const token = localStorage.getItem('lex_token');
      if (!token) {
          navigate('/auth');
          return;
      }
      const res = await fetch(`http://localhost:8000/auth/hire/${lawyerId}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        }
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      navigate('/inbox');
    } catch (err) {
      alert("Failed to connect: " + err.message);
    }
  };

  return (
    <div className="container">
      <h2 className="mb-8 text-center" style={{ fontSize: '3rem' }}>Find Your <span className="text-gradient">Legal Champion</span></h2>
      
      {loading ? (
        <p className="text-center">Loading registered lawyers...</p>
      ) : lawyers.length === 0 ? (
        <div className="glass-panel text-center" style={{ padding: '3rem', maxWidth: '600px', margin: '0 auto' }}>
          <h3>No lawyers available</h3>
          <p>We are currently onboarding top legal professionals to our platform.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '2rem' }}>
          {lawyers.map(lawyer => (
            <div key={lawyer.id} className="glass-panel lawyer-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
                <div style={{ 
                    width: '60px', height: '60px', borderRadius: '50%', 
                    background: 'linear-gradient(135deg, #fcd34d, #f59e0b)', 
                    display: 'flex', alignItems: 'center', justifyContent: 'center', 
                    fontSize: '1.8rem', fontWeight: 'bold', color: '#fff',
                    boxShadow: '0 4px 10px rgba(245, 158, 11, 0.4)'
                  }}>
                  {lawyer.name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <h3 style={{ margin: '0 0 5px 0', fontSize: '1.4rem' }}>Adv. {lawyer.name}</h3>
                  <div className="badge" style={{ display: 'inline-block' }}>{lawyer.specialization || 'General Law'}</div>
                </div>
              </div>
              
              <div style={{ background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '12px', marginTop: '0.5rem' }}>
                <p style={{ margin: '0 0 8px 0', fontWeight: '600', color: 'var(--text-primary)' }}>🎓 {lawyer.experience_years || 0}+ Years Experience</p>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>{lawyer.practicing_courts || 'Supreme Court & High Court'}</p>
              </div>

              <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto', paddingTop: '1rem' }}>
                <button className="btn-primary" style={{ flex: 1, padding: '10px' }} onClick={() => handleHire(lawyer.id)}>
                  Contact Me
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default HireLawyer;
