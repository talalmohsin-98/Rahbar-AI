import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const STEP1 = [
  { id: 'nadra',    label: 'Get or renew a national ID (CNIC)',   icon: '🪪' },
  { id: 'fbr',      label: 'File taxes or check tax status',      icon: '📋' },
  { id: 'license',  label: 'Get a driving license',               icon: '🚗' },
  { id: 'secp',     label: 'Register a business',                 icon: '🏢' },
  { id: 'passport', label: 'Apply for or renew a passport',       icon: '📘' },
];

const STEP2 = {
  nadra: [
    { id: 'new',      label: 'Apply for a new CNIC (first time)' },
    { id: 'renewal',  label: 'Renew an existing or expired CNIC' },
    { id: 'lost',     label: 'Replace a lost or stolen CNIC' },
    { id: 'update',   label: 'Correct or update information on CNIC' },
  ],
  fbr: [
    { id: 'ntn',      label: 'Register for an NTN (first time)' },
    { id: 'return',   label: 'File an annual income tax return' },
    { id: 'atl',      label: 'Check Active Taxpayer List (ATL) status' },
    { id: 'refund',   label: 'Claim a tax refund' },
  ],
  license: [
    { id: 'new',      label: 'Apply for a new driving license' },
    { id: 'renewal',  label: 'Renew an expiring license' },
    { id: 'intl',     label: 'Get an international driving permit' },
  ],
  secp: [
    { id: 'pvt',      label: 'Register a private limited company (Pvt Ltd)' },
    { id: 'smc',      label: 'Register a single-member company (SMC-Pvt)' },
    { id: 'sole',     label: 'Register as sole proprietor' },
    { id: 'change',   label: 'Change company name or details' },
  ],
  passport: [
    { id: 'new',      label: 'Apply for a new passport' },
    { id: 'renewal',  label: 'Renew an expiring passport' },
    { id: 'urgent',   label: 'Urgent/emergency passport' },
    { id: 'child',    label: 'Passport for a minor (under 18)' },
  ],
};

// Deterministic result mapping — this is the architectural claim the Wizard demonstrates
const RESULTS = {
  nadra_new:     { title: 'New CNIC Application', docs: ['Original B-Form or Birth Certificate (Union Council issued)', 'Two recent passport-sized photographs (blue background)', 'Father/guardian\'s original CNIC'], fee: 'Rs. 750 (standard) · Rs. 1,500 (urgent)', time: '30 working days (standard) · 7 days (urgent)', office: 'Nearest NADRA Registration Centre (NRC)', tip: 'Apply at the NRC in your home district. Biometric enrollment is done on-site.'},
  nadra_renewal: { title: 'CNIC Renewal', docs: ['Original expired CNIC', 'One recent passport-sized photograph'], fee: 'Rs. 750 (standard) · Rs. 1,500 (urgent)', time: '30 working days (standard)', office: 'Any NADRA Registration Centre', tip: 'You can also initiate renewal online at my.nadra.gov.pk and collect the card at the NRC.'},
  nadra_lost:    { title: 'Lost CNIC Replacement', docs: ['FIR copy from local police station', 'Photocopy of family registration certificate', 'One passport photograph'], fee: 'Rs. 750', time: '30 working days', office: 'NADRA Registration Centre in home district', tip: 'File the FIR before visiting NADRA. Bring a family witness for verification.'},
  nadra_update:  { title: 'CNIC Information Update', docs: ['Original CNIC', 'Supporting document for the change (e.g., marriage certificate, court order, educational document)'], fee: 'Rs. 750 (standard)', time: '30 working days', office: 'NADRA Registration Centre', tip: 'Nature of supporting document depends on what you want to change. Contact NADRA for specifics.'},
  fbr_ntn:       { title: 'NTN Registration', docs: ['CNIC', 'Bank account number', 'Business address (if applicable)', 'Mobile number registered against CNIC'], fee: 'Free', time: 'Instant (online via IRIS)', office: 'iris.fbr.gov.pk or nearest FBR Regional Tax Office', tip: 'NTN is auto-generated from your CNIC for salaried individuals. Visit IRIS portal.'},
  fbr_return:    { title: 'Annual Tax Return', docs: ['CNIC / NTN', 'Bank statements', 'Salary certificate or business accounts', 'Withholding tax certificates'], fee: 'Free', time: 'Deadline: September 30 each year', office: 'iris.fbr.gov.pk', tip: 'Filing online via IRIS is mandatory. Keep withholding tax certificates from employers/banks.'},
  fbr_atl:       { title: 'ATL Status Check', docs: ['CNIC or NTN number'], fee: 'Free', time: 'Instant', office: 'atl.fbr.gov.pk', tip: 'ATL status is updated after filing your return. Late filing incurs a surcharge to re-enter ATL.'},
  fbr_refund:    { title: 'Tax Refund Claim', docs: ['Filed tax return', 'Bank account details', 'Evidence of excess tax paid'], fee: 'Free', time: '60–90 working days after verification', office: 'IRIS portal or Commissioner Inland Revenue', tip: 'Submit refund application through IRIS. Refund is deposited directly to registered bank account.'},
  license_new:   { title: 'New Driving License', docs: ['Original CNIC', 'Medical fitness certificate (from approved doctor)', 'Learner\'s permit (obtained first)', 'Driving test pass certificate'], fee: 'Rs. 900–1,500 (varies by province)', time: '2–4 weeks after test', office: 'Provincial Excise & Taxation Office (e.g., Punjab: peto.punjab.gov.pk)', tip: 'First obtain a Learner\'s Permit, practice for at least 6 weeks, then appear for the driving test.'},
  license_renewal:{ title: 'Driving License Renewal', docs: ['Original license (or FIR if lost)', 'Original CNIC', 'Medical fitness certificate'], fee: 'Rs. 900–1,500', time: '1–2 weeks', office: 'Nearest Excise & Taxation Office', tip: 'Renew before expiry to avoid late fees. Some provinces allow online renewal via their portal.'},
  license_intl:  { title: 'International Driving Permit', docs: ['Valid Pakistani driving license', 'Original CNIC', 'Two passport photographs'], fee: 'Rs. 500–800', time: '1–2 weeks', office: 'Provincial Excise & Taxation Office', tip: 'IDP is valid for 1 year and recognized in 100+ countries under the Geneva Convention.'},
  secp_pvt:      { title: 'Private Limited Company (Pvt Ltd)', docs: ['Memorandum & Articles of Association', 'CNIC copies of all directors', 'Registered office address proof', 'Form 1 (Declaration of Compliance)'], fee: 'Rs. 1,600–3,000 (based on authorized capital)', time: '1–3 working days (online)', office: 'eservices.secp.gov.pk', tip: 'Minimum 2 directors and 2 shareholders required. All filing done online via SECP\'s e-Services portal.'},
  secp_smc:      { title: 'Single-Member Company (SMC-Pvt)', docs: ['Memorandum & Articles of Association', 'CNIC copy of the single member/director', 'Registered office address proof'], fee: 'Rs. 1,600 (base fee)', time: '1–3 working days (online)', office: 'eservices.secp.gov.pk', tip: 'SMC-Pvt allows a single person to form a company. One person acts as both sole director and sole shareholder.'},
  secp_sole:     { title: 'Sole Proprietorship Registration', docs: ['CNIC', 'Business name', 'Business address'], fee: 'Free (NTN registration with FBR, then business bank account)', time: '1–2 weeks', office: 'FBR (iris.fbr.gov.pk) for NTN; local Chamber of Commerce for optional registration', tip: 'Sole proprietorships are not registered with SECP. Register with FBR for NTN and open a business bank account.'},
  secp_change:   { title: 'Company Name or Details Change', docs: ['Existing company registration number', 'Board resolution approving the change', 'Amended Memorandum & Articles (if applicable)'], fee: 'Rs. 500–2,000', time: '3–5 working days', office: 'eservices.secp.gov.pk', tip: 'Name change requires availability check first. File Form 26 (Special Resolution) via SECP portal.'},
  passport_new:  { title: 'New Passport Application', docs: ['Original CNIC (or B-Form for minors)', 'Two recent passport photographs', 'Completed application form'], fee: 'Rs. 4,500 (36-page standard) · Rs. 5,500 (72-page) · Rs. 9,600 (urgent)', time: '15–30 working days (standard) · 2–3 days (urgent)', office: 'Nearest Passport Office (DGIP) or apply online at onlinemrs.dgip.gov.pk', tip: 'Online appointment is mandatory. Book via the DGIP portal and bring the printout on the day.'},
  passport_renewal:{ title: 'Passport Renewal', docs: ['Original expiring passport', 'Original CNIC', 'Two passport photographs'], fee: 'Rs. 4,500 (standard) · Rs. 9,600 (urgent)', time: '15–30 working days (standard)', office: 'DGIP passport office or online portal', tip: 'Start renewal 6 months before expiry, especially if you travel frequently.'},
  passport_urgent:{ title: 'Urgent Passport', docs: ['Original CNIC', 'Two passport photographs', 'Proof of urgency (e.g., confirmed travel booking)'], fee: 'Rs. 9,600', time: '2–3 working days', office: 'Designated Passport Office (specific offices handle urgent)', tip: 'Urgent passport requires justification. Bring travel booking confirmation or medical emergency proof.'},
  passport_child:{ title: 'Minor\'s Passport', docs: ['B-Form or child\'s birth certificate', 'Father\'s original CNIC', 'Mother\'s original CNIC', 'Two recent photographs of the child', 'Parents\' marriage certificate'], fee: 'Rs. 4,500 (standard)', time: '15–30 working days', office: 'DGIP passport office', tip: 'Both parents must be present (or provide a court order if one parent is absent/deceased).'},
};

export default function Recommend() {
  const [step, setStep] = useState(1);
  const [s1, setS1] = useState(null);
  const [s2, setS2] = useState(null);
  const navigate = useNavigate();

  const result = s1 && s2 ? RESULTS[`${s1}_${s2}`] : null;

  const reset = () => { setStep(1); setS1(null); setS2(null); };

  return (
    <div>
      <div style={{ background: 'var(--navy)', padding: '48px 0 52px' }}>
        <div className="container">
          <div className="badge badge--green" style={{ marginBottom: 14 }}>3-step wizard</div>
          <h1 style={{ fontSize: 34, color: 'var(--white)', marginBottom: 10 }}>
            Find the right procedure
          </h1>
          <p style={{ color: 'rgba(255,255,255,.55)', fontSize: 15 }}>
            Answer two questions and we'll show you exactly what you need.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container" style={{ maxWidth: 680 }}>
          {/* Progress */}
          <div className="step-progress">
            {[1, 2, 3].map((n, i) => (
              <React.Fragment key={n}>
                <div className={`step-dot ${step > n ? 'done' : step === n ? 'active' : 'pending'}`}>
                  {step > n ? '✓' : n}
                </div>
                {i < 2 && <div className={`step-line${step > n ? ' done' : ''}`} />}
              </React.Fragment>
            ))}
          </div>

          {/* Step 1 */}
          {step === 1 && (
            <div>
              <h2 style={{ fontSize: 22, marginBottom: 6 }}>What are you trying to do?</h2>
              <p style={{ color: 'var(--text-soft)', marginBottom: 24, fontSize: 14 }}>
                Select the category that best describes your situation.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {STEP1.map(opt => (
                  <button key={opt.id} onClick={() => { setS1(opt.id); setStep(2); }}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 16,
                      padding: '16px 20px', background: 'var(--white)',
                      border: '1.5px solid var(--border)', borderRadius: 'var(--radius-md)',
                      cursor: 'pointer', textAlign: 'left',
                      transition: 'all var(--transition)',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = 'var(--green)';
                      e.currentTarget.style.background = '#F0FDF9';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = 'var(--border)';
                      e.currentTarget.style.background = 'var(--white)';
                    }}
                  >
                    <span style={{ fontSize: 28 }}>{opt.icon}</span>
                    <span style={{ fontWeight: 500, fontSize: 15 }}>{opt.label}</span>
                    <span style={{ marginLeft: 'auto', color: 'var(--slate-light)' }}>→</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 2 */}
          {step === 2 && s1 && (
            <div>
              <h2 style={{ fontSize: 22, marginBottom: 6 }}>What is your specific situation?</h2>
              <p style={{ color: 'var(--text-soft)', marginBottom: 24, fontSize: 14 }}>
                Choose the option that matches your case.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {STEP2[s1].map(opt => (
                  <button key={opt.id} onClick={() => { setS2(opt.id); setStep(3); }}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 16,
                      padding: '14px 20px', background: 'var(--white)',
                      border: '1.5px solid var(--border)', borderRadius: 'var(--radius-md)',
                      cursor: 'pointer', textAlign: 'left',
                      transition: 'all var(--transition)',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = 'var(--green)';
                      e.currentTarget.style.background = '#F0FDF9';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = 'var(--border)';
                      e.currentTarget.style.background = 'var(--white)';
                    }}
                  >
                    <span style={{ fontWeight: 500, fontSize: 14 }}>{opt.label}</span>
                    <span style={{ marginLeft: 'auto', color: 'var(--slate-light)' }}>→</span>
                  </button>
                ))}
              </div>
              <button className="btn btn-ghost" onClick={() => setStep(1)} style={{ marginTop: 20 }}>
                ← Back
              </button>
            </div>
          )}

          {/* Step 3: Result */}
          {step === 3 && result && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
                <span className="badge badge--green">Result</span>
                <h2 style={{ fontSize: 22, margin: 0 }}>{result.title}</h2>
              </div>

              <div className="card" style={{ padding: '24px', marginBottom: 20 }}>
                <h3 style={{ fontSize: 15, marginBottom: 14, color: 'var(--navy)' }}>
                  📄 Required Documents
                </h3>
                <ul style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {result.docs.map((d, i) => (
                    <li key={i} style={{ fontSize: 14, color: 'var(--text-soft)', lineHeight: 1.6 }}>{d}</li>
                  ))}
                </ul>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
                <div className="card" style={{ padding: '18px 20px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--slate)',
                                textTransform: 'uppercase', letterSpacing: '.6px', marginBottom: 6 }}>
                    Fee
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--navy)' }}>{result.fee}</div>
                </div>
                <div className="card" style={{ padding: '18px 20px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--slate)',
                                textTransform: 'uppercase', letterSpacing: '.6px', marginBottom: 6 }}>
                    Processing Time
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--navy)' }}>{result.time}</div>
                </div>
              </div>

              <div className="card" style={{ padding: '18px 20px', marginBottom: 20 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--slate)',
                              textTransform: 'uppercase', letterSpacing: '.6px', marginBottom: 6 }}>
                  Where to Apply
                </div>
                <div style={{ fontSize: 14, color: 'var(--text-soft)' }}>{result.office}</div>
              </div>

              {result.tip && (
                <div style={{
                  background: '#FFFBEB', border: '1px solid #FDE68A',
                  borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
                  fontSize: 13, color: '#92400E',
                }}>
                  💡 <strong>Tip:</strong> {result.tip}
                </div>
              )}

              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <button className="btn btn-primary" style={{ padding: '10px 22px' }}
                  onClick={() => navigate(`/assistant?q=${encodeURIComponent(
                    `Tell me more about ${result.title} — what are the exact steps and requirements?`
                  )}`)}>
                  Ask for more details →
                </button>
                <button className="btn btn-outline" onClick={() => setStep(2)}>
                  Go Back
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
