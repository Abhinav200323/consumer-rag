import React from 'react';
import { BrowserRouter, Routes, Route, Link, Navigate, useNavigate } from 'react-router-dom';
import Landing from './Landing';
import Auth from './Auth';
import Dashboard from './Dashboard';
import Draft from './Draft';
import HireLawyer from './HireLawyer';
import Inbox from './Inbox';

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('lex_token');
  if (!token) {
    return <Navigate to="/auth" replace />;
  }
  return children;
};

function App() {
  const token = localStorage.getItem('lex_token');
  
  const handleLogout = () => {
    localStorage.removeItem('lex_token');
    localStorage.removeItem('lex_user');
    window.location.href = '/';
  };

  return (
    <BrowserRouter>
      <nav className="navbar flex justify-between items-center">
        <Link to="/" className="nav-brand">
          <span style={{ fontSize: '1.8rem' }}>⚖️</span> Lex Assist
        </Link>
        <div className="nav-links" style={{ display: 'flex', alignItems: 'center' }}>
          {!token ? (
            <Link to="/auth" className="btn-primary" style={{ padding: '8px 16px', marginLeft: '1.5rem' }}>Login / Sign Up</Link>
          ) : (
            <>
              <Link to="/dashboard">Dashboard</Link>
              <Link to="/draft">Draft Document</Link>
              <Link to="/hire-lawyer">Hire Lawyer</Link>
              <Link to="/inbox">Inbox</Link>
              <button onClick={handleLogout} className="btn-outline" style={{ padding: '8px 16px', marginLeft: '1.5rem', fontSize: '0.9rem' }}>Logout</button>
            </>
          )}
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/draft" element={<ProtectedRoute><Draft /></ProtectedRoute>} />
        <Route path="/hire-lawyer" element={<ProtectedRoute><HireLawyer /></ProtectedRoute>} />
        <Route path="/inbox" element={<ProtectedRoute><Inbox /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
