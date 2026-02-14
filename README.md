## Repository contents

`phase_diagram.py`  
Generates the phase diagram of aging dynamics, showing stability, drift, and runaway regimes across damage amplification β and repair capacity μ.

`prcc.py`  
Performs global sensitivity analysis using Latin–hypercube sampling and partial rank correlation coefficients (PRCC) to identify parameters governing the asymptotic aging rate.

`beta_demo.py`  
Demonstrates identifiability of the effective damage removal rate β from noisy longitudinal trajectories.

## Reproducing figures

All figures in the manuscript can be generated independently:

```bash
py phase_diagram.py
py prcc.py
py beta_demo.py
