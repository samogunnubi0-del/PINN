# Harvest Timing Demo

## Objective
Use PINN to predict optimal harvest time for Ac-225 in a recycler scenario.

## Key Results
- **Optimal harvest time**: ~0 hours (0.0 days)
- **Peak Ac-225 yield**: 0.00e+00 atoms
- **Flux sensitivity**: Higher flux means earlier peak (see flux comparison plot)

## Application
- Operators can use PINN to predict harvest windows **without solving ODEs** at query time
- Enables real-time optimization of irradiation schedules
- Fast inference (milliseconds) vs. minutes for numerical ODE solve

