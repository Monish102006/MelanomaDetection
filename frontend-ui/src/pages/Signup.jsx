import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../components/AuthContext';
import { useToast } from '../components/Toast';

export default function Signup({ onNavigate }) {
  const { signup, verifyEmail, resendOtp } = useAuth();
  const showToast = useToast();
  const navigate = useNavigate();

  const [step, setStep] = useState('form'); // 'form' | 'verify'
  const [registeredEmail, setRegisteredEmail] = useState('');

  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    phone: '',
    email: '',
    password: '',
    confirm_password: '',
  });

  const [otp, setOtp] = useState('');
  const [devOtpCode, setDevOtpCode] = useState('');
  const [otpError, setOtpError] = useState('');
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [showDuplicateModal, setShowDuplicateModal] = useState(false);

  const handleNav = (target) => {
    if (onNavigate) {
      onNavigate(target);
    } else {
      navigate(target.startsWith('/') ? target : `/${target}`);
    }
  };

  const validate = () => {
    const errs = {};
    if (!form.first_name.trim()) errs.first_name = 'First name is required.';
    if (!form.last_name.trim()) errs.last_name = 'Last name is required.';

    if (!form.phone.trim()) {
      errs.phone = 'Phone number is required.';
    } else if (!/^[+]?[\d\s\-().]{7,20}$/.test(form.phone.trim())) {
      errs.phone = 'Please enter a valid phone number.';
    }

    if (!form.email.trim()) {
      errs.email = 'Email is required.';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
      errs.email = 'Please enter a valid email address.';
    }

    if (!form.password) {
      errs.password = 'Password is required.';
    } else if (form.password.length < 6) {
      errs.password = 'Password must be at least 6 characters.';
    }

    if (!form.confirm_password) {
      errs.confirm_password = 'Please confirm your password.';
    } else if (form.password !== form.confirm_password) {
      errs.confirm_password = 'Passwords do not match.';
    }

    return errs;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setApiError('');
    const errs = validate();
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setLoading(true);
    try {
      const data = await signup({
        first_name: form.first_name.trim(),
        last_name: form.last_name.trim(),
        phone: form.phone.trim(),
        email: form.email.trim(),
        password: form.password,
      });
      setRegisteredEmail(data.email || form.email.trim());
      setStep('verify');
      if (data?.dev_otp) {
        setDevOtpCode(data.dev_otp);
        setOtp(data.dev_otp);
        showToast(`Verification code: ${data.dev_otp}`, 'info');
      } else {
        showToast('Verification code sent!', 'success');
      }
    } catch (err) {
      if (err.message?.includes('already exists') || err.message?.includes('Already Used')) {
        setShowDuplicateModal(true);
      } else {
        setApiError(err.message || 'Registration failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async (e) => {
    e.preventDefault();
    setOtpError('');
    if (otp.length !== 6) {
      setOtpError('Please enter a valid 6-digit code.');
      return;
    }

    setLoading(true);
    try {
      await verifyEmail(registeredEmail, otp);
      showToast('Email verified successfully! You can now sign in.', 'success');
      handleNav('/login');
    } catch (err) {
      setOtpError(err.message || 'Invalid verification code.');
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    setLoading(true);
    try {
      const data = await resendOtp(registeredEmail);
      if (data?.dev_otp) {
        setDevOtpCode(data.dev_otp);
        setOtp(data.dev_otp);
        showToast(`New verification code: ${data.dev_otp}`, 'info');
      } else {
        showToast('A new verification code has been sent.', 'success');
        setOtp('');
      }
      setOtpError('');
    } catch (err) {
      setOtpError(err.message || 'Failed to resend code.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
    if (errors[field]) setErrors((prev) => ({ ...prev, [field]: '' }));
    if (apiError) setApiError('');
  };

  return (
    <div className="auth-page auth-page-minimal">
      <div className="auth-card-minimal card" style={{ maxWidth: '480px' }}>
        {/* Brand Header */}
        <div className="auth-minimal-header">
          <div className="auth-minimal-logo">
            <i className="fas fa-plus-square"></i>
          </div>
          <h1 className="auth-minimal-brand">
            Mela<span>Detect</span> AI
          </h1>
          <h2 className="auth-minimal-title">
            {step === 'form' ? 'Create Account' : 'Verify Email'}
          </h2>
          <p className="auth-minimal-subtitle">
            {step === 'form'
              ? 'Join to track and evaluate your skin health'
              : `Enter the 6-digit code sent to ${registeredEmail}`}
          </p>
        </div>

        {apiError && (
          <div role="alert" className="auth-error-banner">
            <i className="fas fa-exclamation-circle"></i>
            <span>{apiError}</span>
          </div>
        )}

        {step === 'form' ? (
          <form onSubmit={handleSubmit} noValidate className="auth-minimal-form">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div className="form-group">
                <label className="form-label" htmlFor="signup-first-name">
                  First Name
                </label>
                <input
                  id="signup-first-name"
                  type="text"
                  className="form-input"
                  placeholder="Jane"
                  value={form.first_name}
                  onChange={handleChange('first_name')}
                  disabled={loading}
                />
                {errors.first_name && <div className="form-error">{errors.first_name}</div>}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="signup-last-name">
                  Last Name
                </label>
                <input
                  id="signup-last-name"
                  type="text"
                  className="form-input"
                  placeholder="Doe"
                  value={form.last_name}
                  onChange={handleChange('last_name')}
                  disabled={loading}
                />
                {errors.last_name && <div className="form-error">{errors.last_name}</div>}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="signup-email">
                Email Address
              </label>
              <input
                id="signup-email"
                type="email"
                className="form-input"
                placeholder="name@example.com"
                value={form.email}
                onChange={handleChange('email')}
                autoComplete="email"
                disabled={loading}
              />
              {errors.email && <div className="form-error">{errors.email}</div>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="signup-phone">
                Phone Number
              </label>
              <input
                id="signup-phone"
                type="tel"
                className="form-input"
                placeholder="+1 (555) 000-0000"
                value={form.phone}
                onChange={handleChange('phone')}
                disabled={loading}
              />
              {errors.phone && <div className="form-error">{errors.phone}</div>}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div className="form-group">
                <div className="form-label-row">
                  <label className="form-label" htmlFor="signup-password">
                    Password
                  </label>
                  <button
                    type="button"
                    className="btn-link-subtle"
                    onClick={() => setShowPassword(!showPassword)}
                    tabIndex={-1}
                  >
                    {showPassword ? 'Hide' : 'Show'}
                  </button>
                </div>
                <input
                  id="signup-password"
                  type={showPassword ? 'text' : 'password'}
                  className="form-input"
                  placeholder="6+ chars"
                  value={form.password}
                  onChange={handleChange('password')}
                  disabled={loading}
                />
                {errors.password && <div className="form-error">{errors.password}</div>}
              </div>

              <div className="form-group">
                <div className="form-label-row">
                  <label className="form-label" htmlFor="signup-confirm">
                    Confirm
                  </label>
                  <button
                    type="button"
                    className="btn-link-subtle"
                    onClick={() => setShowConfirm(!showConfirm)}
                    tabIndex={-1}
                  >
                    {showConfirm ? 'Hide' : 'Show'}
                  </button>
                </div>
                <input
                  id="signup-confirm"
                  type={showConfirm ? 'text' : 'password'}
                  className="form-input"
                  placeholder="Re-enter"
                  value={form.confirm_password}
                  onChange={handleChange('confirm_password')}
                  disabled={loading}
                />
                {errors.confirm_password && (
                  <div className="form-error">{errors.confirm_password}</div>
                )}
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-block"
              disabled={loading}
              style={{ marginTop: '8px' }}
            >
              {loading ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i>
                  <span>Creating Account...</span>
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </form>
        ) : (
          /* Minimal OTP View */
          <div className="auth-minimal-form">
            {otpError && (
              <div role="alert" className="auth-error-banner">
                <i className="fas fa-exclamation-circle"></i>
                <span>{otpError}</span>
              </div>
            )}

            <form onSubmit={handleVerify}>
              {devOtpCode && (
                <div
                  style={{
                    background: 'rgba(59, 130, 246, 0.08)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '8px',
                    padding: '12px 14px',
                    marginBottom: '16px',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
                    Verification Code
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '22px', fontWeight: 700, letterSpacing: '4px', color: 'var(--primary-color, #2563eb)' }}>
                      {devOtpCode}
                    </span>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline"
                      style={{ padding: '3px 10px', fontSize: '12px', height: 'auto', cursor: 'pointer' }}
                      onClick={() => setOtp(devOtpCode)}
                    >
                      Fill Code
                    </button>
                  </div>
                </div>
              )}

              <div className="form-group" style={{ textAlign: 'center' }}>
                <label className="form-label" style={{ marginBottom: '6px' }}>
                  6-Digit Verification Code
                </label>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                  Please enter the 6-digit code sent to <strong>{registeredEmail}</strong>
                </p>
                <input
                  type="text"
                  maxLength={6}
                  className="form-input"
                  style={{
                    fontSize: '22px',
                    letterSpacing: '8px',
                    textAlign: 'center',
                    fontWeight: 600,
                  }}
                  placeholder="------"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, ''))}
                  autoFocus
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-block"
                disabled={loading || otp.length !== 6}
                style={{ marginTop: '14px' }}
              >
                {loading ? 'Verifying...' : 'Verify & Continue'}
              </button>
            </form>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '18px',
                fontSize: '13px',
              }}
            >
              <button
                type="button"
                className="btn-link-action"
                style={{ color: 'var(--text-secondary)' }}
                onClick={() => setStep('form')}
              >
                ← Back
              </button>
              <button
                type="button"
                className="btn-link-action"
                onClick={handleResend}
                disabled={loading}
              >
                Resend Code
              </button>
            </div>
          </div>
        )}

        <div className="auth-minimal-footer">
          <span>Already have an account?</span>{' '}
          <button
            type="button"
            className="btn-link-action"
            onClick={() => handleNav('/login')}
          >
            Sign In
          </button>
        </div>
      </div>

      {/* Duplicate Account Modal */}
      {showDuplicateModal && (
        <div className="modal-backdrop" onClick={() => setShowDuplicateModal(false)}>
          <div className="card modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '400px', padding: '24px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
              Account Already Exists
            </h3>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '20px' }}>
              An account with this email is already registered in MelaDetect AI. Would you like to sign in instead?
            </p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-outline"
                onClick={() => setShowDuplicateModal(false)}
              >
                Dismiss
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => handleNav('/login')}
              >
                Go to Sign In
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
