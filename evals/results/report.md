# Evaluation report

Mode: **offline rules**

| Metric | Value | Gate |
|---|---|---|
| red flag recall | 1.0 | 0.85 |
| precision | 1.0 | 0.85 |
| severity accuracy | 1.0 |  |
| decoy false alarms | 0 | 0 |
| metric accuracy | 1.0 | 1.0 |
| citation validity | 1.0 | 1.0 |
| fabrication catch rate | 0.857 | 0.85 |
| findings reported | 7 |  |
| runtime s | 0.02 |  |

## Planted red flags

| ID | Expected | Found as |
|---|---|---|
| RF1 | Top customer is 45% of revenue | COM-1 |
| RF2 | Key customer can terminate on change of control | LEG-1 |
| RF3 | Pending INR 6.5 cr patent claim, unprovisioned | LEG-2 |
| RF4 | Stated revenue 48.0 vs P&L 41.2 | FIN-1 |
| RF5 | About 8 months of cash runway | FIN-2 |
| RF6 | Co-founder CTO resigned, no successor | PPL-1 |
| RF7 | KYC data exposure not reported to regulators | CMP-1 |

## Fabricated findings caught by the critic

| Fabrication type | Rejected |
|---|---|
| invented number | 1/1 |
| wrong number | 3/3 |
| fake source | 1/1 |
| no citation | 1/1 |
| semantic contradiction (no numbers) | 0/1 |
