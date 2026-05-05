import React from 'react';
import { useNavigate } from 'react-router-dom';

function Landing() {
  const navigate = useNavigate();

  return (
    <div className="container" style={{ paddingTop: '10vh', textAlign: 'center' }}>
      <h1 style={{ fontSize: '4rem', marginBottom: '1rem' }}>
        Your Personal <span className="text-gradient">AI Legal Buddy</span>
      </h1>
      <p style={{ fontSize: '1.25rem', maxWidth: '800px', margin: '0 auto 3rem' }}>
        Lex Assist helps everyday Indians navigate consumer law. Chat with our intelligent agent, draft legal documents instantly, or hire a professional lawyer to take your case forward.
      </p>
      <div className="flex gap-4" style={{ justifyContent: 'center' }}>
        <button className="btn-primary" onClick={() => navigate('/auth')}>Get Started for Free</button>
        <button className="btn-outline" onClick={() => navigate('/hire-lawyer')}>Find a Lawyer</button>
      </div>

      <div className="glass-panel" style={{ marginTop: '5rem', padding: '3rem' }}>
        <h2 className="mb-8">How it works</h2>
        <div className="flex gap-4">
          <div style={{ flex: 1 }}>
            <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>💬</h3>
            <h3>Chat with AI</h3>
            <p>Ask legal questions in plain English and get answers backed by Indian Consumer Law.</p>
          </div>
          <div style={{ flex: 1 }}>
            <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>📝</h3>
            <h3>Draft Documents</h3>
            <p>Generate formal, ready-to-use legal notices and complaints instantly.</p>
          </div>
          <div style={{ flex: 1 }}>
            <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>👨‍⚖️</h3>
            <h3>Hire Professionals</h3>
            <p>Connect with registered lawyers who can help you file your case and represent you.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Landing;
