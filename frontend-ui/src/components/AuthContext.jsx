import { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';

const AuthContext = createContext(null);

const API_BASE = '/api/auth';

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('auth_token'));
  const [loading, setLoading] = useState(true);

  // Validate token on mount
  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    fetch(`${API_BASE}/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error('Token invalid');
        return res.json();
      })
      .then((data) => {
        setUser(data);
      })
      .catch(() => {
        localStorage.removeItem('auth_token');
        setToken(null);
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, [token]);

  const login = useCallback(async (email, password) => {
    const res = await fetch(`${API_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Invalid email or password.');
    }
    localStorage.setItem('auth_token', data.token);
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  const signup = useCallback(async (formData) => {
    const res = await fetch(`${API_BASE}/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Registration failed.');
    }
    return data;
  }, []);

  const verifyEmail = useCallback(async (email, otp) => {
    const res = await fetch(`${API_BASE}/verify-email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, otp }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Verification failed.');
    }
    return data;
  }, []);

  const resendOtp = useCallback(async (email) => {
    const res = await fetch(`${API_BASE}/resend-otp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Failed to resend code.');
    }
    return data;
  }, []);

  const logout = useCallback(async () => {
    if (token) {
      try {
        await fetch(`${API_BASE}/logout`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch {
        // Clear local credentials even if network call fails
      }
    }
    localStorage.removeItem('auth_token');
    setToken(null);
    setUser(null);
    // Push a clean state so back-button won't reveal authenticated pages
    window.history.replaceState(null, '', '/');
  }, [token]);

  const updateProfile = useCallback(async (updates) => {
    const res = await fetch(`${API_BASE}/me`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(updates),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Profile update failed.');
    }
    setUser(data);
    return data;
  }, [token]);

  const value = useMemo(
    () => ({ user, token, loading, login, signup, logout, updateProfile, verifyEmail, resendOtp, isAuthenticated: !!user }),
    [user, token, loading, login, signup, logout, updateProfile, verifyEmail, resendOtp],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
