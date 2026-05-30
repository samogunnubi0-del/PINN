# Data And Source Assumptions

This file tracks which scientific assumptions are fed into the ODE/PINN training
pipeline and which assumptions stay in deterministic dashboard post-processing.

## Fed Into ODE / PINN Training

These values affect generated training rows and therefore must be source-backed
before claiming predictor-grade credibility.

| Item | Current value / model | Used in | Source status |
| --- | --- | --- | --- |
| Ra-226 half-life | 1600 y | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ra-225 half-life | 14.8 d | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ac-225 half-life | 9.920 d | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ra-227 half-life | 42.2 min | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ac-227 half-life | 21.772 y | ODE + PINN physics loss | IAEA LiveChart / DDEP value |
| Ra-226(n,2n) threshold | 6.422 MeV | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Ra-226(n,2n) effective cross section | 26.69 mb fission-spectrum average; table also lists 755.7 mb at 14 MeV | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Ra-226(n,gamma) thermal cross section | 12.78-12.79 b at 0.0253 eV; resonance integral about 282-286 b | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Neutron spectrum model | scalar energy with threshold and 1/v approximations | ODE + PINN physics loss | Must be stated as simplified model |

## Kept Out Of PINN Training

These belong in deterministic post-processing, not the neural-network loss.

| Item | Current value / model | Used in | Source status |
| --- | --- | --- | --- |
| Chemical recovery yield | UI slider, default 90% | Dashboard optimizer | Literature reports >90%, >95%, and >98% depending on resin/process |
| Cooling/transport time | UI slider, default 5 days | Dashboard optimizer | Engineering assumption |
| Ac-227 impurity limit | strict 0.15% activity impurity | Dashboard optimizer | Literature/regulatory-style strict constraint; cite directly before final claim |
| Activity conversion | A = lambda * N | Dashboard optimizer | Standard nuclear decay law |

## Training Data Coverage Targets

Before each full retrain, confirm the training log reports coverage in all regimes:

- Thermal near 0.025 eV.
- Epithermal/resonance-like stress cases.
- Near-threshold fast cases around 6.42 MeV.
- Fast production cases around 14 MeV.
- Pure Ra-226 targets.
- Recycled/interrupted targets with nonzero daughters.
- Empty or near-empty inventory edge cases.

## Source Links To Cite

- NNDC NuDat / ENSDF for isotope half-lives and decay data:
  - https://www.nndc.bnl.gov/nudat3/
  - https://www.nndc.bnl.gov/ensdf/
- IAEA LiveChart / DDEP for Ac-227 decay data:
  - https://nds.iaea.org/relnsd/ddep?NUCID=227AC
- JENDL Ra-226 evaluated neutron cross-section tables:
  - https://wwwndc.jaea.go.jp/jendl/j32/Tabsigs/Ra226.HTML
  - https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?iso=Ra226&lib=J32
  - https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?/data/JENDL/JENDL-4-prc/intern/Ra226.intern=
- JENDL evaluation background for minor actinide neutron data:
  - Journal of Nuclear Science and Technology, evaluation of neutron nuclear data for minor nuclides.
- Ac-225 / Ra separation and recovery:
  - Improved Ac-225/Bi-213 production using DGA/TEHDGA resin, reported overall Ac-225 yield exceeding 98%.
  - Optimization of cation exchange for Ac-225 separation from radioactive thorium/radium/other metals, reported >90% total process recovery and 95-98% step recoveries.
  - Eichrom Actinium-225 separation application notes for DGA resin behavior.

## Next Source Tasks

- Replace source notes above with full bibliography entries in the final paper/poster.
- Decide whether the ODE should use the JENDL fission-spectrum average value or the 14-MeV table value for the fast route; document the choice clearly.
- Keep the 0.15% Ac-227 activity-impurity limit labeled as a strict regulatory-style constraint unless a direct official acceptance specification is found.
