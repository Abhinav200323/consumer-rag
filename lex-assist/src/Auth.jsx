import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('customer'); // customer or lawyer
  
  // Lawyer specific
  const [specialization, setSpecialization] = useState('');
  const [experienceYears, setExperienceYears] = useState('');
  const [practicingCourts, setPracticingCourts] = useState('');

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const endpoint = isLogin ? '/auth/login' : '/auth/signup';
    const payload = isLogin 
      ? { email, password } 
      : { 
          email, password, name, role,
          specialization: role === 'lawyer' ? specialization : undefined,
          experience_years: role === 'lawyer' ? parseInt(experienceYears) || 0 : undefined,
          practicing_courts: role === 'lawyer' ? practicingCourts : undefined
        };

    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      
      localStorage.setItem('lex_token', data.access_token);
      localStorage.setItem('lex_user', JSON.stringify(data));
      navigate('/dashboard');
      // trigger page reload to update navbar state
      window.location.reload();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
      <div className="glass-panel" style={{ padding: '3rem', width: '100%', maxWidth: '400px' }}>
        <h2 className="mb-8 text-center" style={{ fontSize: '2rem' }}>
          {isLogin ? 'Welcome Back' : 'Join Lex Assist'}
        </h2>
        
        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <>
              <input 
                type="text" 
                placeholder="Full Name" 
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="customer">I am a Customer</option>
                <option value="lawyer">I am a Lawyer</option>
              </select>
              
              {role === 'lawyer' && (
                  <>
                    <input 
                        type="text" 
                        placeholder="Specialization (e.g., Consumer Protection)" 
                        value={specialization}
                        onChange={(e) => setSpecialization(e.target.value)}
                        required
                    />
                    <input 
                        type="number" 
                        placeholder="Years of Experience" 
                        value={experienceYears}
                        onChange={(e) => setExperienceYears(e.target.value)}
                        required
                        min="0"
                    />
                    <input 
                        type="text" 
                        placeholder="Practicing Courts (e.g., Supreme Court & High Court)" 
                        value={practicingCourts}
                        onChange={(e) => setPracticingCourts(e.target.value)}
                        required
                    />
                  </>
              )}
            </>
          )}
          
          <input 
            type="email" 
            placeholder="Email Address" 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input 
            type="password" 
            placeholder="Password" 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          
          <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '1rem' }}>
            {isLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>
        
        <p className="text-center" style={{ marginTop: '1.5rem', cursor: 'pointer', color: 'var(--accent-primary)' }} onClick={() => setIsLogin(!isLogin)}>
          {isLogin ? "Don't have an account? Sign up" : "Already have an account? Login"}
        </p>
      </div>
    </div>
  );
}

export default Auth;
