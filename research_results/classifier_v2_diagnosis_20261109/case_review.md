# Review of harmful BIRD configuration switches

Offline review only; reference SQL was not a classifier input.

| ID | Selected config | Selected / compact correct | Selected / compact predicted quality | Main SQL discrepancy |
| --- | --- | --- | --- | --- |
| bird:4 | 14+32 | 0 / 1 | 0.899 / 0.679 | Chosen SQL filters schools.fundingtype with Directly Funded; reference uses frpm.Charter Funding Type = Directly funded. |
| bird:9 | 14+32 | 0 / 1 | 0.714 / 0.525 | Funding filter uses Directly Funded; compact uses Directly funded and returns the reference result. |

Case and column semantics matter to SQL evaluation. Equality of saved execution results does not establish general semantic equivalence of SQL.
See report.json for exact saved SQL, input evidence, nearest train questions, and additive logit-gap decomposition.
The sigmoid quality scores are model estimates, not calibrated confidence guarantees.
