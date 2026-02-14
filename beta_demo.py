import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
true_beta = np.linspace(0.02, 0.2, 20)
t = np.linspace(0, 50, 200)
estimated_beta = []

for b in true_beta:
    S = np.exp(-b*t)
    noise = np.random.normal(0, 0.02, size=S.shape)
    S_noisy = np.clip(S + noise, 1e-6, None)
    logS = np.log(S_noisy)
    slope = np.polyfit(t, logS, 1)[0]
    estimated_beta.append(-slope)


estimated_beta = np.array(estimated_beta)

plt.figure()
plt.scatter(true_beta, estimated_beta)
plt.plot(true_beta, true_beta)
plt.xlabel("True beta")
plt.ylabel("Estimated beta proxy")
plt.title("Observable proxy recovers underlying beta")
plt.tight_layout()
plt.savefig("fig3_beta_demo.png", dpi=300)
