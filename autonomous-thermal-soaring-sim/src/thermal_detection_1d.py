"""
simple thermal detection with lift
Aircraft flies straight along the x-axis.
A thermal sits at the origin.
We track: updraft felt, vario reading, and Kalman estimate.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
})

# atmosphere function
def atm(h_p, DISA=0):
    h_p  = h_p * 1000
    DISA = DISA * 491.67 / 273.15
    if h_p < 36089:
        theta = (1 - 6.87535e-6 * h_p) + DISA / 518.67
        delta = (1 - 6.87535e-6 * h_p) ** 5.2561
    elif 36089 <= h_p < 65617:
        theta = 0.75187 + DISA / 518.67
        delta = 0.22336 * np.exp((36089 - h_p) / 20806.7)
    else:
        raise ValueError('h_p out of range')
    sigma = delta / theta
    T     = theta * 518.67
    P     = delta * 2116.22
    rho   = sigma * 0.0023769
    a     = np.sqrt(1.4 * 1716.554 * T)
    mu    = 2.2697e-8 * T**1.5 / (T + 198.72)
    nu    = mu / rho
    return T, P, rho, a, nu, delta, sigma

def TAS(mach, h_p, DISA=0):
    _, _, _, a, _, _, _ = atm(h_p, DISA)
    return mach * a * 0.3048   # ft/s → m/s

ALTITUDE_KFT   = 5.0     # kft
MACH           = 0.12    # cruise Mach (~40 m/s)
THERMAL_PEAK   = 3.0     # m/s updraft at centre
THERMAL_RADIUS = 150.0   # metres
SINK_RATE      = 0.8     # constant aircraft sink (m/s) — simplified
VARIO_NOISE    = 0.3     # variometer noise std (m/s)
TIME_STEP      = 1.0     # seconds
TOTAL_TIME     = 300     # seconds

V = TAS(MACH, ALTITUDE_KFT)            # true airspeed (m/s)
x_start = -V * TOTAL_TIME / 2          # start far enough left to cross thermal

print(f"Airspeed : {V:.1f} m/s")
print(f"Start x  : {x_start:.0f} m")
print(f"Thermal centre at x = 0 m\n")

# thermal model simple
def updraft(x, peak=THERMAL_PEAK, radius=THERMAL_RADIUS):
    return peak * np.exp(-(x / radius)**2)

# kalman filter
est   = 0.0    # current estimate
P     = 4.0    # estimate variance
Q     = 0.05   # process noise
R     = VARIO_NOISE**2

# simulation
x_pos = x_start
xs, updrafts, vario_reads, estimates, uncertainties, altitudes = [], [], [], [], [], []
altitude = 0.0   # track relative altitude gain/loss

for _ in range(TOTAL_TIME):
    w       = updraft(x_pos)               # true updraft at this x
    net     = w - SINK_RATE                # what the vario should read
    vario   = net + np.random.normal(0, VARIO_NOISE)   # noisy measurement

    # Kalman predict + update
    P       = P + Q
    K       = P / (P + R)
    est     = est + K * (vario - est)
    P       = (1 - K) * P

    xs.append(x_pos)
    updrafts.append(w)
    vario_reads.append(vario)
    estimates.append(est)
    uncertainties.append(np.sqrt(P))
    altitude += net * TIME_STEP

    x_pos  += V * TIME_STEP               # fly straight

altitudes_arr = np.cumsum([0] + [updraft(xs[i]) - SINK_RATE for i in range(len(xs)-1)])

# plots
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig.suptitle(f'Thermal Detection — Straight Flight  (V={V:.1f} m/s, Alt={ALTITUDE_KFT} kft)', fontsize=12)

xs      = np.array(xs)
est_arr = np.array(estimates)
unc_arr = np.array(uncertainties)

# Panel 1: thermal profile + Kalman estimate
axes[0].fill_between(xs, 0, updrafts, alpha=0.15, color='steelblue', label='True updraft')
axes[0].plot(xs, updrafts,   color='steelblue', lw=1.5, label='True updraft')
axes[0].plot(xs, vario_reads, color='gray',      lw=0.8, alpha=0.6, label='Vario (noisy)')
axes[0].plot(xs, est_arr,     color='darkorange', lw=2,   label='Kalman estimate')
axes[0].fill_between(xs, est_arr - unc_arr, est_arr + unc_arr,
                     alpha=0.25, color='darkorange', label='±1σ')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].axvline(0, color='red',   lw=0.8, linestyle='--', alpha=0.5, label='Thermal centre')
axes[0].set_ylabel('Net climb (m/s)')
axes[0].set_title('Vario reading and Kalman estimate')
axes[0].legend(fontsize=8, ncol=3)

# Panel 2: detection flag
detected = (est_arr > 0.2).astype(float)
axes[1].fill_between(xs, 0, detected, step='mid', color='seagreen', alpha=0.6, label='Lift detected')
axes[1].plot(xs, detected, color='seagreen', lw=1, drawstyle='steps-mid')
axes[1].axvline(0, color='red', lw=0.8, linestyle='--', alpha=0.5)
axes[1].set_ylabel('Detection (1 = lift)')
axes[1].set_ylim(-0.05, 1.3)
axes[1].set_title('Thermal detection flag  (threshold: Kalman est > 0.2 m/s)')
axes[1].legend(fontsize=8)
axes[1].set_xlabel('X position (m)')


plt.tight_layout()
plt.savefig('thermal_detection_simple.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to thermal_detection_simple.png")
