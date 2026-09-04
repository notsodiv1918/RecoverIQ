# AI Revenue Recovery Agent

Detects revenue at risk (failed payments, abandoned checkouts, overdue invoices),
diagnoses the root cause, learns which intervention actually works, executes it
under hard guardrails, and measures money recovered across a batch — with a
full audit trail per transaction.

Works with **zero API keys or paid services** out of the box. Two integrations
are optional upgrades with silent, labeled fallback if unset:
- `GROQ_API_KEY` → LLM-assisted diagnosis and bounded Q&A intent routing
- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` → real test-mode Payment Links
  instead of simulated ones (see **Razorpay Integration** below)

## Run it

```bash
./run.sh
```
Starts the backend on `http://localhost:8000`. Open `frontend/index.html`
directly in a browser (no build step — static file hitting the API via fetch).

Click **Seed 300 transactions** → **Run Recovery Batch** a few times (cooldowns
and quiet-hours deferrals mean not everything resolves on the first pass — that's
the guardrails working, not a bug).

## Architecture

```
Synthetic Event Generator (Faker)
        │  payment_failed / checkout_abandoned / invoice_overdue
        ▼
   Detector           → severity score (amount, overdue days, tier) — prioritization only
        ▼
   Diagnosis Agent     → root_cause + confidence
                          (rule-based table by default; optional Groq LLM,
                           constrained to the same enum, silent fallback on error)
        ▼
   Bandit Policy        → Thompson Sampling over a Beta-Bernoulli posterior per
                           (root_cause, action) pair — genuinely learns from
                           outcomes as the batch runs, not a static lookup
        ▼
   Guardrails            → HARD, code-enforced, cannot be overridden by the policy:
                           • max 3 attempts per transaction, then dead/escalate
                           • cooldown between touches (see Regulatory Grounding)
                           • quiet hours block customer contact (see below)
                           • >₹50,000 always requires human sign-off, no auto-send
                           • dispute_pending always routes to a human
                           • opt-out / DND customers never get automated contact
        ▼
   Executor              → send_payment_link calls the REAL Razorpay test-mode
                           API (see below); everything else is simulated and logged
        ▼
   Outcome Simulator      → probabilistic recovery per (action, root_cause) pair,
                           calibrated against public dunning benchmarks (see below)
        ▼
   Audit Trail            → every stage (detect/diagnose/policy/guardrail/action/
                           outcome/razorpay_link) as a structured, queryable event
        ▼
   Dashboard               → recovered ₹ vs at-risk ₹, learned win-rates, static-vs-
                           learned policy comparison, recovery-rate deviation
                           analysis, bounded Q&A, drill-down audit per transaction
```

## Regulatory grounding (not invented "be nice" numbers)

Two guardrail thresholds are modeled on real Indian payments regulation, not
arbitrary ops preferences:

- **Cooldown between retries** (`config.COOLDOWN_BATCHES`): grounded in RBI's
  *Digital Payments – E-Mandate Framework, 2026* (Circular RBI/DPSS/2026-27/396,
  21 Apr 2026), which requires a pre-debit notification at least **24 hours**
  before any recurring-debit re-attempt. A mandate-based retry legally cannot
  fire back-to-back with no gap.
- **Quiet hours** (`config.QUIET_HOURS_START/END`, 21:00–09:00): TRAI's Telecom
  Commercial Communication Customer Preference Regulations (TCCCPR), 2018
  restrict unsolicited commercial calls/SMS to **9:00 AM–9:00 PM IST**. This
  replaced an earlier, invented 22:00–08:00 window with the actual regulated one.

## Razorpay integration (the one real external call)

Every other action in this system is simulated and logged. `send_payment_link`
is not: `razorpay_client.py` calls the real Razorpay **test-mode** Payment
Links API (`client.payment_link.create(...)`) to generate an actual payment
link for the transaction.

- Get free test-mode keys: Razorpay Dashboard → Settings → API Keys (test mode
  is the default on signup, no live business verification needed).
- Set `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` in `backend/.env`.
- Without keys (or if the API call fails for any reason), it falls back to a
  clearly-labeled simulated link — `real: false` is stored in the audit trail
  and the DB, never silently upgraded to look real.
- What's still simulated even with real keys: whether the (synthetic) customer
  actually completes the payment. There's no real customer in this demo to
  click the link, so the outcome (recovered / not) still comes from the
  calibrated probability model, not a real webhook. The link itself, though,
  is a real object created via a real API call.

## Calibration — what the recovery-probability numbers are based on

`simulator.py`'s per-(action, root_cause) recovery probabilities are
self-authored estimates, not measured from real transactions (this system has
no real transaction history). They were set to land in the range reported by
public dunning-management benchmarks, not picked arbitrarily:

- Paddle Retain reports 50%+ recovery on failed payments: https://developer.paddle.com/concepts/retain/payment-recovery-dunning
- ProsperStack: well-run dunning recovers 50–80% of failed payments: https://prosperstack.com/blog/subscription-dunning
- Fungies (2026): businesses without strong dunning recover as low as ~20%,
  while AI-driven retry/outreach reaches 60–80%: https://fungies.io/saas-dunning-management-failed-payment-recovery-2026

`config.BASELINE_RECOVERY_RATE = 0.18` (the "flat, unprioritized process"
comparison point) sits at the low end of that range; the calibrated
simulator's blended output lands in the 50–70% band this system should be
judged against — consistent with, not identical to, the cited public numbers.

## Why the policy stays deterministic even though it "learns"

`bandit.py` learns (Thompson Sampling over real outcomes), but the guardrail
layer that can override it is plain Python, not a model:
- guardrails are provably unbypassable — a policy that hallucinated a bad
  action still can't skip the attempt cap or the amount threshold, because
  guardrails run as a separate code layer *after* the policy proposes something
- every decision has a reproducible rationale for audit, including the exact
  sampled posterior values that led to the choice

## Naming honesty

A few features were originally pitched with punchier names than what they
actually do. Renamed for accuracy:
- "AI Merchant Copilot" → **Bounded Q&A**: a fixed set of intents, each backed
  by a real DB query. Not a freeform LLM chat — ask it something outside the
  fixed set and it says so, rather than guessing.
- "Counterfactual AI" → **static-vs-learned policy comparison**
  (`/api/policy-comparison`): compares the static ladder's expected recovery
  against the bandit's current learned posterior, both scored against the same
  self-authored probability table. This is a comparison of two policies against
  one assumed model, not causal inference from real counterfactual outcomes.
- "New Failure Pattern Discovery" → **sub-cluster recovery-rate deviation
  analysis** (`/api/patterns`): KMeans finds sub-populations *within* an
  already-diagnosed root cause whose real recovery rate deviates from the
  group average. It does not discover new failure types diagnosis.py doesn't
  already know about.

## Mapping to the judging bar

| Requirement | Where |
|---|---|
| Detect revenue at risk | `detector.py` severity scoring over all three event types |
| Diagnose root cause | `diagnosis.py` (rule-based + optional LLM) |
| Right intervention, and it improves | `bandit.py` Thompson Sampling, `policy.py` static baseline for comparison |
| Bounded recovery workflow | `executor.py` guardrails: max attempts, cooldown, quiet hours, amount threshold, dispute/DND handling |
| Measured $ recovered across a batch | `/api/metrics`, `/api/summary` — at-risk vs recovered, by channel, by root cause, vs baseline |
| Compliant escalation | amount threshold + dispute_pending + DND all force `escalate_human_review`; cooldown/quiet-hours grounded in real RBI/TRAI rules |
| Stopping rules | max attempts hard cap, cooldown, quiet-hours deferral |
| Audit trail | `/api/audit` and per-transaction `/api/transactions/{id}` — every stage logged with rationale, including the real/simulated Razorpay link |

## File structure

```
revenue_recovery/
├── run.sh
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py             FastAPI routes
│       ├── config.py            guardrail thresholds + citations for each one
│       ├── database.py          SQLAlchemy engine/session
│       ├── models.py            Customer, Transaction, ActionLog, AuditEvent, PolicyStat, Meta
│       ├── synthetic_data.py    generates realistic customers + transactions
│       ├── detector.py          severity scoring
│       ├── diagnosis.py         root cause classification (rule-based + optional LLM)
│       ├── policy.py            static action ladder (baseline, for comparison)
│       ├── bandit.py            Thompson Sampling policy — the actual decision-maker
│       ├── executor.py          guardrails + execution + audit logging
│       ├── razorpay_client.py   real Razorpay test-mode Payment Links integration
│       ├── simulator.py         probabilistic outcome model (see Calibration above)
│       ├── pattern_discovery.py sub-cluster recovery-rate deviation analysis
│       ├── summary.py           executive summary text + Indian (lakh/crore) number formatting
│       ├── journey.py           audit log → recovery-graph journey stages
│       ├── copilot.py           bounded Q&A (fixed intents, real queries)
│       └── runner.py            shared batch-run logic
└── frontend/
    └── index.html               single-page dashboard (vanilla JS, no build step, no CDN deps)
```

## Known limitations (be upfront about these if asked)

- Every action except `send_payment_link` is **simulated and logged**, not
  wired to a real gateway/SMS/email provider. `send_payment_link` makes one
  real Razorpay test-mode API call; whether the synthetic customer "pays" is
  still simulated (see Razorpay Integration above).
- Recovery outcomes come from a self-authored probability table calibrated
  against public dunning benchmarks (see Calibration above), not from real
  transaction history — there is none, since this is synthetic data.
- Cooldown is measured in "Run Batch" cycles, not real wall-clock time, so a
  demo running in minutes can still show the guardrail firing and clearing.
  The 24h/9-hour windows it's grounded in (see Regulatory Grounding) are real;
  the clock simulating them is compressed for demo purposes.
- Diagnosis rule table covers 15 root causes across 3 event types. LLM
  diagnosis and LLM intent-routing for the bounded Q&A are optional upgrades,
  not load-bearing — the system is fully deterministic and functional without
  any LLM key.