import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useToast } from '../components/Toast';
import { useAuth } from '../components/AuthContext';

export default function Results({ analysisResult, onNavigate }) {
  const { reportId } = useParams();
  const { token } = useAuth();
  const navigate = useNavigate();
  const showToast = useToast();
  const [currentResult, setCurrentResult] = useState(analysisResult || null);
  const [loading, setLoading] = useState(!analysisResult);
  const [activeImageTab, setActiveImageTab] = useState('original');
  const [zoomModal, setZoomModal] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const handleNav = (target) => {
    if (onNavigate) { onNavigate(target); }
    else {
      const mapping = { 'new-check': '/upload', upload: '/upload', history: '/history', reports: '/history' };
      navigate(mapping[target] || target);
    }
  };

  useEffect(() => {
    if (analysisResult) { setCurrentResult(analysisResult); setLoading(false); return; }
    if (!token) { setLoading(false); return; }
    let isMounted = true;
    setLoading(true);
    const targetUrl = reportId ? `/api/analyses/${reportId}` : '/api/analyses';
    fetch(targetUrl, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!isMounted) return;
        if (Array.isArray(data)) {
          if (data.length > 0) {
            return fetch(`/api/analyses/${data[0].analysis_id}`, { headers: { Authorization: `Bearer ${token}` } })
              .then((r) => (r.ok ? r.json() : data[0]))
              .then((full) => { if (isMounted) setCurrentResult(full); });
          } else { setCurrentResult(null); }
        } else { setCurrentResult(data); }
      })
      .catch((e) => { console.warn('Failed to fetch analysis:', e); })
      .finally(() => { if (isMounted) setLoading(false); });
    return () => { isMounted = false; };
  }, [analysisResult, reportId, token]);

  if (loading) {
    return (
      <div className="page-container" id="page-report-loading">
        <div className="card report-empty-card" style={{ padding: '60px 24px', textAlign: 'center' }}>
          <div className="loading-spinner" style={{ margin: '0 auto 16px', fontSize: '28px', color: 'var(--primary)' }}>
            <i className="fas fa-spinner fa-spin"></i>
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>Loading Skin Check Report...</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '6px' }}>Fetching analysis metrics and dermatological assessment.</p>
        </div>
      </div>
    );
  }

  if (!currentResult) {
    return (
      <div className="page-container" id="page-report-empty">
        <div className="card report-empty-card">
          <div className="report-empty-icon" aria-hidden="true"><i className="fas fa-file-waveform"></i></div>
          <h2 className="report-empty-title">No Skin Check Selected</h2>
          <p className="report-empty-subtitle">Start a new skin check or select a past assessment from your history to view its report.</p>
          <div className="report-empty-actions">
            <button type="button" className="btn btn-primary" onClick={() => handleNav('new-check')}><i className="fas fa-camera" aria-hidden="true"></i><span>Start a Skin Check</span></button>
            <button type="button" className="btn btn-outline" onClick={() => handleNav('history')}><i className="fas fa-history" aria-hidden="true"></i><span>View History</span></button>
          </div>
        </div>
      </div>
    );
  }

  if (currentResult.status !== 'completed' || !currentResult.diagnosis) {
    return (
      <div className="page-container" id="page-report-error">
        <div className="card report-error-card">
          <div className="report-error-icon" aria-hidden="true"><i className="fas fa-triangle-exclamation"></i></div>
          <h2 className="report-error-title">{currentResult.status === 'image_quality_insufficient' ? 'Image Quality Needs Adjustment' : 'Analysis Could Not Be Completed'}</h2>
          <p className="report-error-subtitle">{currentResult.message || 'The uploaded photo could not be clearly analyzed. For best results, use good natural lighting and make sure the lesion is sharp and centered.'}</p>
          <button type="button" className="btn btn-primary" onClick={() => handleNav('new-check')}><i className="fas fa-redo" aria-hidden="true"></i><span>Try With Another Photo</span></button>
        </div>
      </div>
    );
  }

  const { diagnosis, explanation, report, evolution_assessment, original_image_url, analysis_id, timestamp, _customContext } = currentResult;
  const pred = diagnosis?.diagnosis?.prediction || 'Benign';
  const isMelanoma = pred === 'Melanoma';
  const confidence = Math.round(diagnosis?.diagnosis?.confidence || 0);
  const melanomaProb = Number(diagnosis?.melanoma_probability ?? diagnosis?.probabilities?.melanoma ?? 0);

  // Log fusion information specifically requested
  if (diagnosis?.diagnosis?.fusion_weights || diagnosis?.diagnosis?.fusion_score) {
    console.log("=== Fusion Model Classification Details ===");
    console.log("Composite Melanoma Score:", diagnosis.diagnosis.fusion_score);
    console.log("Effective Weights for Classification:", diagnosis.diagnosis.fusion_weights);
    console.log("=========================================");
  }

  let riskAssessment = 'Low Risk';
  let riskBadgeClass = 'badge-low';
  let riskSummaryText = 'The features analyzed align with typical benign dermatological characteristics. Continue normal routine self-checks.';
  if (isMelanoma) {
    if (confidence >= 80 || melanomaProb >= 0.7) {
      riskAssessment = 'High Risk'; riskBadgeClass = 'badge-high';
      riskSummaryText = 'The model observed notable structural asymmetry, irregular borders, or pigment heterogeneity. We strongly recommend having this spot examined in person by a qualified dermatologist.';
    } else {
      riskAssessment = 'Moderate Risk'; riskBadgeClass = 'badge-moderate';
      riskSummaryText = 'Some atypical characteristics were identified that warrant professional observation. Consider consulting a healthcare professional for an in-person check.';
    }
  } else if (melanomaProb >= 0.35 || confidence < 65) {
    riskAssessment = 'Moderate Risk'; riskBadgeClass = 'badge-moderate';
    riskSummaryText = 'The assessment indicates mild morphological variance. Monitoring over time or scheduling a routine skin check is advised.';
  }

  const clinicalFeatures = diagnosis?.clinical_features || {};
  const measurements = diagnosis?.measurements?.lesion || {};
  const hasPhysicalScale = Boolean(measurements.physical_scale_available);

  const handleDownloadPDF = () => {
    if (downloading) return;
    setDownloading(true);
    showToast('Preparing your report...', 'info');
    setTimeout(() => {
      if (report?.pdf_url) {
        const link = document.createElement('a');
        link.href = report.pdf_url;
        link.download = `MelaDetect_Report_${analysis_id?.substring(0, 8) || 'SkinCheck'}.pdf`;
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
        setDownloading(false); showToast('Report downloaded successfully.', 'success');
      } else { setDownloading(false); showToast("We couldn't generate the report right now. Please try again.", 'warning'); }
    }, 800);
  };
  const handlePrint = () => { window.print(); };

  let activeDisplayUrl = original_image_url;
  if (activeImageTab === 'mask' && diagnosis?.segmentation?.mask_url) activeDisplayUrl = diagnosis.segmentation.mask_url;
  else if (activeImageTab === 'overlay' && diagnosis?.segmentation?.overlay_url) activeDisplayUrl = diagnosis.segmentation.overlay_url;
  else if (activeImageTab === 'gradcam' && explanation?.grad_cam_url) activeDisplayUrl = explanation.grad_cam_url;

  const displayDate = timestamp
    ? new Date(timestamp).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    : new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });

  const hasExplanationText =
    (explanation?.reasoning && explanation.reasoning.some((r) => r && r.trim().length > 10)) ||
    (explanation?.summary && explanation.summary.trim().length > 10);

  return (
    <div className="page-container report-page-wrapper" id="page-report">
      {/* Top Action Bar */}
      <div className="report-action-bar top-bar">
        <button type="button" className="btn btn-outline report-nav-back-btn" onClick={() => onNavigate('history')}>
          <i className="fas fa-arrow-left" aria-hidden="true"></i><span>Back to Results</span>
        </button>
        <div className="report-actions-right">
          <button type="button" className="btn btn-outline" onClick={handlePrint} title="Print or save as digital copy">
            <i className="fas fa-print" aria-hidden="true"></i><span>Print</span>
          </button>
          <button type="button" className="btn btn-primary btn-download-pdf" onClick={handleDownloadPDF} disabled={downloading}>
            <i className="fas fa-download" aria-hidden="true"></i>
            <span>{downloading ? 'Preparing...' : 'Download PDF'}</span>
          </button>
        </div>
      </div>

      <article className="card report-sheet">
        {/* Header */}
        <header className="report-document-header">
          <div className="report-header-left">
            <div className="report-brand-badge">
              <i className="fas fa-plus-square text-teal" aria-hidden="true"></i><span>MelaDetect AI</span>
            </div>
            <h1 className="report-main-title">Skin Lesion Analysis Report</h1>
            <p className="report-meta-line">
              <span>Date: <strong>{displayDate}</strong></span>
              {analysis_id && <span className="report-id-text">Reference: #{analysis_id.substring(0, 8).toUpperCase()}</span>}
              {_customContext?.location && <span>Location: <strong>{_customContext.location}</strong></span>}
            </p>
          </div>
          <div className="report-header-right">
            <span className={`risk-badge-lg ${riskBadgeClass}`}>{riskAssessment}</span>
          </div>
        </header>

        {/* 1. Assessment Summary */}
        <section className="report-section" aria-labelledby="assessment-summary-heading">
          <h2 id="assessment-summary-heading" className="report-section-heading">Assessment Summary</h2>
          <div className="assessment-summary-grid">
            <div className="summary-stat-box">
              <span className="summary-stat-label">Risk Assessment</span>
              <span className={`summary-stat-value risk-badge ${riskBadgeClass}`}>{riskAssessment}</span>
            </div>
            <div className="summary-stat-box">
              <span className="summary-stat-label">Model Confidence</span>
              <strong className="summary-stat-value text-primary font-manrope">{confidence}%</strong>
            </div>
            <div className="summary-stat-box">
              <span className="summary-stat-label">Status</span>
              <span className="summary-stat-value text-success">
                <i className="fas fa-check-circle" aria-hidden="true"></i>
                <span style={{ marginLeft: '6px' }}>Complete</span>
              </span>
            </div>
          </div>
          <div className="assessment-meaning-card">
            <h3 className="meaning-title">What this assessment means:</h3>
            <p className="meaning-text">{riskSummaryText}</p>
          </div>
        </section>

        {/* 2. AI Clinical Explanation - PRIORITY SECTION */}
        <section className="report-section report-section-featured" aria-labelledby="ai-explanation-heading">
          <h2 id="ai-explanation-heading" className="report-section-heading report-section-heading-featured">
            <i className="fas fa-brain" aria-hidden="true" style={{ marginRight: '10px', color: 'var(--primary)' }}></i>
            AI Clinical Explanation
          </h2>
          <p className="report-section-sub">
            Evidence-grounded interpretation generated by the AI reasoning engine using peer-reviewed medical literature:
          </p>
          {hasExplanationText ? (
            <div className="explanation-content-block">
              {explanation.summary && explanation.summary.trim().length > 10 && (
                <div className="explanation-summary-card">
                  <p className="explanation-summary-text">{explanation.summary}</p>
                </div>
              )}
              {explanation.reasoning && explanation.reasoning.length > 0 ? (
                <div className="explanation-reasoning-body">
                  {explanation.reasoning.map((para, i) => {
                    if (!para || para.trim().length < 5) return null;
                    const cleaned = para.replace(/\*\*/g, '').trim();
                    const isHeader = /^(\d+\.\s*(WHAT|HOW|MEDICAL|CONFIDENCE|NEXT)|WHAT|HOW|MEDICAL|CONFIDENCE|NEXT)/i.test(cleaned);
                    return isHeader
                      ? (<h4 key={i} className="explanation-section-heading">{cleaned}</h4>)
                      : (<p key={i} className="explanation-para">{cleaned}</p>);
                  })}
                </div>
              ) : (
                <p className="explanation-para">{explanation.summary}</p>
              )}
              {_customContext?.notes && (
                <div className="patient-observations-block">
                  <h4 className="patient-obs-title">
                    <i className="fas fa-comment-medical" aria-hidden="true"></i> Patient-Reported Observations
                  </h4>
                  <p className="patient-obs-text">&quot;{_customContext.notes}&quot;</p>
                  <p className="patient-obs-note">This reported history was incorporated into the AI evidence retrieval process.</p>
                </div>
              )}
              {explanation?.evidence_citations && explanation.evidence_citations.length > 0 && (
                <div className="evidence-citations-block">
                  <h4 className="evidence-citations-title">
                    <i className="fas fa-book-medical" aria-hidden="true"></i> Supporting Medical Evidence
                  </h4>
                  <ul className="evidence-citations-list">
                    {explanation.evidence_citations.slice(0, 6).map((cite, i) => (
                      <li key={i} className="evidence-cite-item">
                        <span className="cite-index">[{i + 1}]</span>
                        <span className="cite-title">{cite.title || cite.source || 'Medical Reference'}</span>
                        {cite.source && cite.source !== cite.title && <span className="cite-source"> &mdash; {cite.source}</span>}
                        {cite.relevance_score && <span className="cite-score"> (relevance: {Number(cite.relevance_score).toFixed(2)})</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {explanation?.grad_cam_detail && (
                <div className="gradcam-detail-note">
                  <i className="fas fa-microscope text-teal" aria-hidden="true"></i>
                  <p>{explanation.grad_cam_detail}</p>
                </div>
              )}
              {explanation?.confidence_assessment && (
                <div className="explanation-confidence-row">
                  <span className="confidence-label">Evidence Confidence:</span>
                  <span className={`confidence-pill confidence-${explanation.confidence_assessment.replace(/[^a-zA-Z-]/g, '')}`}>
                    {explanation.confidence_assessment}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="explanation-unavailable-block">
              <div className="explanation-unavailable-icon" aria-hidden="true">
                <i className="fas fa-triangle-exclamation"></i>
              </div>
              <div>
                <strong>AI Explanation Unavailable</strong>
                <p style={{ marginTop: '6px', color: 'var(--text-secondary)' }}>
                  The evidence-grounded explanation could not be generated. This may occur when the knowledge base is not connected or the LLM service is unreachable. The ABCDE feature data below is still available.
                </p>
                {!_customContext?.notes && (
                  <p style={{ marginTop: '6px', color: 'var(--text-muted)', fontSize: '13px' }}>
                    <i className="fas fa-lightbulb" aria-hidden="true" style={{ marginRight: '5px' }}></i>
                    Tip: Next time, describe any observed changes in the upload form to improve AI evidence matching.
                  </p>
                )}
              </div>
            </div>
          )}
        </section>

        {/* 3. Image Inspection */}
        <section className="report-section" aria-labelledby="lesion-image-heading">
          <h2 id="lesion-image-heading" className="report-section-heading">Image Inspection</h2>
          <p className="report-section-sub">Inspection maps displaying model attention and lesion boundary delineation:</p>
          <div className="explainability-workspace">
            <div className="explainability-tab-bar" role="tablist">
              <button type="button" role="tab" aria-selected={activeImageTab === 'original'} className={`tab-pill ${activeImageTab === 'original' ? 'active' : ''}`} onClick={() => setActiveImageTab('original')}>Original Image</button>
              {diagnosis?.segmentation?.overlay_url && (
                <button type="button" role="tab" aria-selected={activeImageTab === 'overlay'} className={`tab-pill ${activeImageTab === 'overlay' ? 'active' : ''}`} onClick={() => setActiveImageTab('overlay')}>Lesion Boundary Overlay</button>
              )}
              {explanation?.grad_cam_url && (
                <button type="button" role="tab" aria-selected={activeImageTab === 'gradcam'} className={`tab-pill ${activeImageTab === 'gradcam' ? 'active' : ''}`} onClick={() => setActiveImageTab('gradcam')}>Model Attention Map</button>
              )}
              {diagnosis?.segmentation?.mask_url && (
                <button type="button" role="tab" aria-selected={activeImageTab === 'mask'} className={`tab-pill ${activeImageTab === 'mask' ? 'active' : ''}`} onClick={() => setActiveImageTab('mask')}>Segmentation Mask</button>
              )}
            </div>
            <div className="explainability-side-by-side">
              <div className="compare-pane">
                <span className="compare-pane-title">Original Image</span>
                <div className="compare-image-box"><img src={original_image_url} alt="Original lesion" /></div>
              </div>
              <div className="compare-pane">
                <span className="compare-pane-title">
                  {activeImageTab === 'overlay' ? 'Boundary Delineation' : activeImageTab === 'gradcam' ? 'Attention Focus (Grad-CAM)' : activeImageTab === 'mask' ? 'Binary Segmentation Mask' : 'Selected Layer'}
                </span>
                <div className="compare-image-box"><img src={activeDisplayUrl} alt="Model explanation visualization" /></div>
              </div>
            </div>
            <div className="explainability-note-box">
              <i className="fas fa-circle-info text-teal" aria-hidden="true"></i>
              <p>The highlighted areas indicate regions that contributed to the model&apos;s assessment. This visualization highlights areas of algorithmic focus and does not prove melanoma.</p>
            </div>
          </div>
        </section>

        {/* 3.5. Scale Calibration Results */}
        {hasPhysicalScale && (
          <section className="report-section" aria-labelledby="scale-analysis-heading">
            <h2 id="scale-analysis-heading" className="report-section-heading">Physical Measurements</h2>
            <div style={{ marginTop: '16px', padding: '16px', background: 'var(--accent-green-bg)', borderRadius: '8px', border: '1px solid rgba(34,197,94,0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <i className="fas fa-ruler-combined" style={{ color: 'var(--accent-green)' }} />
                <strong style={{ color: 'var(--accent-green)' }}>Physical Measurements (Calibrated)</strong>
                {diagnosis?.scale_calibration?.detected && (
                  <span style={{ marginLeft: 'auto', fontSize: '11px', padding: '3px 10px', borderRadius: '12px', background: 'var(--accent-green-bg)', border: '1px solid var(--accent-green)', color: 'var(--accent-green)' }}>
                    {diagnosis.scale_calibration.method} — {(diagnosis.scale_calibration.confidence * 100).toFixed(0)}% confidence
                  </span>
                )}
              </div>
              <div className="metrics-row">
                {measurements.diameter_mm != null && (
                  <div><strong>{measurements.diameter_mm} mm</strong><span>diameter</span></div>
                )}
                {measurements.area_mm2 != null && (
                  <div><strong>{measurements.area_mm2} mm²</strong><span>area</span></div>
                )}
                {measurements.perimeter_mm != null && (
                  <div><strong>{measurements.perimeter_mm} mm</strong><span>perimeter</span></div>
                )}
              </div>
            </div>
            <p style={{ color: 'var(--text-light)', fontSize: '13px', marginTop: '12px', marginBottom: 0 }}>
              Physical measurements were calibrated using a detected reference object. Accuracy depends on calibration quality.
            </p>
          </section>
        )}

        {diagnosis?.scale_calibration && !diagnosis.scale_calibration.calibration_valid && (
          <section className="report-section">
            <div style={{ marginTop: '16px', padding: '14px 16px', background: 'var(--accent-amber-bg)', borderRadius: '8px', color: 'var(--text-light)' }}>
              <strong>Physical measurement: UNAVAILABLE</strong>
              <div style={{ marginTop: '4px' }}>
                {diagnosis.scale_calibration.calibration_reason || 'The reference calibration could not be verified.'}
              </div>
              <div style={{ marginTop: '4px' }}>Pixel measurements remain available.</div>
            </div>
          </section>
        )}

        {/* 4. ABCDE Feature Analysis */}
        <section className="report-section" aria-labelledby="abcde-analysis-heading">
          <h2 id="abcde-analysis-heading" className="report-section-heading">ABCDE Feature Analysis</h2>
          <p className="report-section-sub">Quantitative evaluation of dermatological characteristics used in clinical skin health assessment:</p>
          <div className="abcde-report-sections">
            {/* A */}
            <div className="abcde-report-item">
              <div className="abcde-item-header">
                <div className="abcde-circle-badge">A</div>
                <div>
                  <h3 className="abcde-item-title">A &mdash; Asymmetry</h3>
                  <span className="abcde-item-metric">
                    Assessment: <strong>{clinicalFeatures.asymmetry?.score_label || 'Evaluated'}</strong>
                  </span>
                </div>
              </div>
              <p className="abcde-item-explanation">
                {clinicalFeatures.asymmetry?.score_label === 'High' ? 'The lesion contours show noticeable structural asymmetry between halves.' : clinicalFeatures.asymmetry?.score_label === 'Moderate' ? 'Mild asymmetry observed across the contour axes.' : 'The lesion halves display balanced symmetry and even outline.'}
              </p>
            </div>
            {/* B */}
            <div className="abcde-report-item">
              <div className="abcde-item-header">
                <div className="abcde-circle-badge">B</div>
                <div>
                  <h3 className="abcde-item-title">B &mdash; Border</h3>
                  <span className="abcde-item-metric">
                    Assessment: <strong>{clinicalFeatures.border?.score_label || 'Evaluated'}</strong>
                  </span>
                </div>
              </div>
              <p className="abcde-item-explanation">
                {clinicalFeatures.border?.score_label === 'Irregular' ? 'The perimeter exhibits scalloped, jagged, or less distinct outer boundaries.' : 'The outer margin appears relatively regular and circumscribed.'}
              </p>
            </div>
            {/* C */}
            <div className="abcde-report-item">
              <div className="abcde-item-header">
                <div className="abcde-circle-badge">C</div>
                <div>
                  <h3 className="abcde-item-title">C &mdash; Color</h3>
                  <span className="abcde-item-metric">
                    Assessment: <strong>{clinicalFeatures.color?.score_label || 'Evaluated'}</strong>
                  </span>
                </div>
              </div>
              <p className="abcde-item-explanation">
                {clinicalFeatures.color?.score_label?.includes('Multiple') ? 'Variable pigmentation and mixed color tones detected across the surface.' : 'Color distribution is generally uniform across the lesion surface.'}
              </p>
            </div>
            {/* D */}
            <div className="abcde-report-item">
              <div className="abcde-item-header">
                <div className="abcde-circle-badge">D</div>
                <div>
                  <h3 className="abcde-item-title">D &mdash; Diameter</h3>
                  <span className="abcde-item-metric">
                    Assessment: <strong>{hasPhysicalScale && measurements.diameter_mm ? `${measurements.diameter_mm} mm` : measurements.diameter_px ? `${measurements.diameter_px} px` : 'Evaluated'}</strong>
                    {hasPhysicalScale && measurements.measurement_confidence && (
                      <span style={{ marginLeft: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
                        (scale detected with {(measurements.measurement_confidence * 100).toFixed(1)}% confidence)
                      </span>
                    )}
                  </span>
                </div>
              </div>
              <p className="abcde-item-explanation">
                {hasPhysicalScale && measurements.diameter_mm
                  ? measurements.diameter_mm > 6 ? `Calibrated result: Physical diameter is measured at ${measurements.diameter_mm} mm (larger than 6 mm benchmark).` : `Calibrated result: Physical diameter is measured at ${measurements.diameter_mm} mm (within typical 6 mm threshold).`
                  : 'Lesion dimension estimated. Spots exceeding 6mm or showing rapid expansion warrant in-person review.'}
              </p>
            </div>
            {/* E */}
            <div className="abcde-report-item">
              <div className="abcde-item-header">
                <div className="abcde-circle-badge">E</div>
                <div>
                  <h3 className="abcde-item-title">E &mdash; Evolution</h3>
                  <span className="abcde-item-metric">Assessment: <strong>{evolution_assessment ? (evolution_assessment.status === 'clear_evolution' ? 'Clear Evolution' : evolution_assessment.status === 'possible_evolution' ? 'Possible Evolution' : evolution_assessment.status === 'no_significant_evolution' ? 'No Significant Evolution' : 'Unable to Assess') : (_customContext?.notes ? 'Patient History Provided' : 'Baseline Documented')}</strong></span>
                </div>
              </div>
              <div className="abcde-item-explanation">
                {evolution_assessment ? (
                  <div style={{ marginTop: '8px' }}>
                    <p style={{ margin: '0 0 8px 0', color: 'var(--text-primary)' }}><strong>AI Longitudinal Comparison:</strong> {evolution_assessment.observations?.join(' ')}</p>
                    <ul style={{ margin: '0 0 8px 16px', padding: '0', fontSize: '13px' }}>
                      <li><strong>Size:</strong> {evolution_assessment.changes?.size}</li>
                      <li><strong>Shape/Border:</strong> {evolution_assessment.changes?.shape}</li>
                      <li><strong>Color:</strong> {evolution_assessment.changes?.color}</li>
                      <li><strong>Structure:</strong> {evolution_assessment.changes?.structure}</li>
                    </ul>
                    {evolution_assessment.limitations && evolution_assessment.limitations.length > 0 && (
                      <p style={{ margin: '0', fontSize: '12px', color: 'var(--text-muted)' }}>
                        <em>Note: {evolution_assessment.limitations[0]}</em>
                      </p>
                    )}
                  </div>
                ) : (
                  <p>
                    {_customContext?.notes
                      ? `Reported observation: "${_customContext.notes}". Continued tracking of any changes in shape, size, elevation, or sensation is recommended.`
                      : 'Documenting this skin check establishes a baseline. Note any future changes in size, contour, elevation, or symptoms like itching.'}
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* 5. Recommended Next Steps */}
        <section className="report-section" aria-labelledby="next-steps-heading">
          <h2 id="next-steps-heading" className="report-section-heading">Recommended Next Steps</h2>
          <div className="next-steps-card">
            <div className="next-step-row">
              <div className="next-step-icon"><i className="fas fa-user-doctor"></i></div>
              <div className="next-step-text">
                <strong>Schedule a Clinical Skin Examination</strong>
                <p>{explanation?.next_steps || 'If you notice any new, unusual, or rapidly changing spots, consider having them evaluated in person by a certified dermatologist.'}</p>
              </div>
            </div>
            <div className="next-step-row">
              <div className="next-step-icon"><i className="fas fa-calendar-check"></i></div>
              <div className="next-step-text">
                <strong>Perform Monthly Self-Checks</strong>
                <p>Keep track of moles across your skin once a month. Take photos under good lighting to monitor changes over time.</p>
              </div>
            </div>
            {explanation?.limitations && explanation.limitations.length > 0 && (
              <div className="next-step-row limitations-row">
                <div className="next-step-icon" style={{ color: 'var(--text-muted)' }}><i className="fas fa-info-circle"></i></div>
                <div className="next-step-text">
                  <strong style={{ color: 'var(--text-muted)' }}>Analysis Limitations</strong>
                  <ul style={{ margin: '6px 0 0', paddingLeft: '16px' }}>
                    {explanation.limitations.map((lim, i) => (<li key={i} style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '3px' }}>{lim}</li>))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </section>

        <footer className="medical-disclaimer-box" role="note">
          <p>MelaDetect AI provides AI-assisted information and is not a medical diagnosis. If you notice concerning or changing skin lesions, consider consulting a qualified healthcare professional.</p>
        </footer>
      </article>

      <div className="report-action-bar bottom-bar">
        <button type="button" className="btn btn-outline" onClick={() => onNavigate('history')}><i className="fas fa-arrow-left" aria-hidden="true"></i><span>Back to Results</span></button>
        <div className="report-actions-right">
          <button type="button" className="btn btn-outline" onClick={handlePrint}><i className="fas fa-print" aria-hidden="true"></i><span>Print Report</span></button>
          <button type="button" className="btn btn-primary" onClick={handleDownloadPDF} disabled={downloading}><i className="fas fa-download" aria-hidden="true"></i><span>{downloading ? 'Preparing...' : 'Download PDF'}</span></button>
        </div>
      </div>

      {zoomModal && (
        <div className="modal-backdrop" onClick={() => setZoomModal(false)} role="dialog" aria-modal="true">
          <div className="modal-card modal-zoom-card" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="modal-close-btn" onClick={() => setZoomModal(false)} aria-label="Close full screen view"><i className="fas fa-times"></i></button>
            <img src={activeDisplayUrl} alt="High resolution skin lesion view" className="zoom-image-element" />
          </div>
        </div>
      )}
    </div>
  );
}
