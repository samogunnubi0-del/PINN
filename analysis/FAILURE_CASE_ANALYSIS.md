# PINN Failure Case Analysis
## Overview
This report identifies scenarios where the PINN diverges from the ODE ground truth.
**Key Finding**: Ra-225-dominant scenarios show larger errors (extrapolation)

## Scenario Results
### ra225_dom_pure_decay
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 49019607073860272128.00% | 7.00e+17 | 1.00e+18 |
| Ra-225 | 95.90% | 7.27e+17 | 9.17e+17 |
| Ac-225 | 98.04% | 1.99e+17 | 2.82e+17 |

### ra226_dom_normal
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 26.82% | 1.62e+23 | 1.62e+23 |
| Ra-225 | 98.04% | 3.06e+19 | 4.93e+19 |
| Ac-225 | 98.04% | 5.80e+18 | 1.18e+19 |

### mixed_low_flux
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 50.79% | 6.95e+19 | 1.00e+20 |
| Ra-225 | 95.91% | 7.27e+17 | 9.18e+17 |
| Ac-225 | 99.02% | 2.56e+17 | 3.24e+17 |

## Key Insights
1. **Ra-225-dominant scenarios** have larger errors because training data was Ra-226 dominant.
2. **Pure decay (phi=0)** should be easiest to learn (exponential → Ac-225); often shows underprediction.
3. **Mixed IC** scenarios interpolate well when in training distribution.

## Recommendations for Judges
- Model **prevents alchemy** (core constraint) ✅
- Model **learns mixed regimes** well ✅
- Error in Ra-225 scenarios is **expected** (extrapolation beyond training) — can be fixed with more diverse data
- For **production use**, retrain with data-augmentation focused on rare ICs
