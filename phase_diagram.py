import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def damage_dynamics(t, D, alpha, beta, mu, R_Amax, d):
    D = max(D[0], 0.0)
    production = beta * (D ** alpha)
    repair = mu * (D / (1.0 + D / R_Amax))
    loss = d * D
    return [production - repair - loss]

T_final = 200
t_eval = np.linspace(0, T_final, 2000)
D0 = [0.05]

alpha = 1.15         
d = 0.05             
R_Amax = 1.0         

beta_vals = np.linspace(0.01, 1.2, 60)  
mu_vals = np.linspace(0.01, 1.2, 60)    

phase = np.zeros((len(mu_vals), len(beta_vals)))

for i, mu in enumerate(mu_vals):
    for j, beta in enumerate(beta_vals):

        sol = solve_ivp(
            damage_dynamics,
            (0, T_final),
            D0,
            t_eval=t_eval,
            args=(alpha, beta, mu, R_Amax, d),
            rtol=1e-7,
            atol=1e-9
        )

        D_traj = sol.y[0]

        if np.any(np.isnan(D_traj)) or D_traj[-1] > 10:
            phase[i, j] = 2  
        elif np.std(D_traj[-300:]) < 1e-3:
            phase[i, j] = 0  
        else:
            phase[i, j] = 1

plt.figure(figsize=(8, 6))
plt.imshow(
    phase,
    origin='lower',
    aspect='auto',
    extent=[beta_vals.min(), beta_vals.max(),
            mu_vals.min(), mu_vals.max()]
)

cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels([
    'Stable homeostasis',
    'Aging drift',
    'Runaway damage'
])

plt.xlabel(r'Damage amplification $\beta$')
plt.ylabel(r'Repair capacity $\mu$')
plt.title('Aging phase diagram')

plt.tight_layout()
plt.show()
