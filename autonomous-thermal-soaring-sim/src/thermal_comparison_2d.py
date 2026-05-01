"""
2-D Thermal Detection — Sinusoidal Search + Turn + Circle
Aircraft flies a sine wave. A thermal sits above or below the path.
Detection: Kalman vario crosses threshold + sign of lateral motion tells which side.
Turn model: heading_rate = k * U_est  (Powers et al. 
"stupid thermal model"
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
np.random.seed(7)

# params
THERMAL_X        = 400.0   # thermal centre x (m)
THERMAL_Y_OFFSET = 120.0   # above (+) or below (-) the sine path (m)
THERMAL_PEAK     = 4.0     # peak updraft (m/s)
THERMAL_RADIUS   = 100.0   # Gaussian radius (m)
THERMAL_LIFETIME = 500.0   # dissipation time constant (s)

V              = 15.0   # airspeed (m/s)
SINK_RATE      = 0.7    # glide sink (m/s)
ALT_START      = 500.0  # starting altitude (m)

SIN_AMP        = 80.0   # sine path lateral amplitude (m)
SIN_FREQ       = 0.008  # cycles per metre along x

VARIO_NOISE    = 0.25   # sensor noise std (m/s)
KF_Q           = 0.05   # Kalman process noise
KF_P0          = 4.0    # Kalman initial variance

K_TURN         = 0.06   # aggressiveness (rad/s per m/s)
DETECT_THRESH  = 0.30   # Kalman est threshold to trigger turn (m/s)
LOCK_THRESH    = 0.50   # Kalman est threshold to start circling (m/s)
ESCAPE_THRESH  = -0.20  # Kalman est threshold to abort circle (m/s)
CIRCLE_R       = 60.0   # initial circling radius (m)

DT             = 0.5    # time step (s)
T_MAX          = 120.0  # simulation duration (s)

# thermal model
THERMAL_Y = THERMAL_Y_OFFSET

def thermal_updraft(x, y, t):
    dist = np.sqrt((x - THERMAL_X)**2 + (y - THERMAL_Y)**2)
    return THERMAL_PEAK * np.exp(-t / THERMAL_LIFETIME) * np.exp(-(dist / THERMAL_RADIUS)**2)

# kalman filter
kf_x = 0.0
kf_P = KF_P0
kf_R = VARIO_NOISE**2

def kalman_update(meas):
    global kf_x, kf_P
    kf_P = kf_P + KF_Q
    K    = kf_P / (kf_P + kf_R)
    kf_x = kf_x + K * (meas - kf_x)
    kf_P = (1 - K) * kf_P
    return kf_x, np.sqrt(kf_P)

# states
SEARCH  = 'SEARCH'
TURNING = 'TURNING'
CIRCLE  = 'CIRCLE'

# state var
x, y     = 0.0, 0.0
heading  = 0.0
altitude = ALT_START
t        = 0.0
state    = SEARCH

turn_dir   = 0
circle_cx  = 0.0
circle_cy  = 0.0
circle_r   = CIRCLE_R
circle_ang = 0.0

side_history = []
prev_kf = 0.0
prev_y  = 0.0

# logs
log_t, log_x, log_y, log_alt = [], [], [], []
log_true_w, log_kf, log_vario, log_state, log_cmd = [], [], [], [], []

# main loop
print(f"{'t':>6}  {'State':<8}  {'x':>5}  {'y':>5}  {'Alt':>6}  {'TrueW':>6}  {'KF':>6}  {'Cmd':>7}  Note")
print("-" * 78)

for step in range(int(T_MAX / DT)):
    # Sensors
    true_w    = thermal_updraft(x, y, t)
    net_climb = true_w - SINK_RATE
    vario     = net_climb + np.random.normal(0, VARIO_NOISE)
    kf_est, kf_std = kalman_update(vario)
    turn_cmd  = K_TURN * kf_est

    # Side detection: dot product of lateral motion sign with vario change
    side_history.append(np.sign(y - prev_y) * (kf_est - prev_kf))
    if len(side_history) > 12:
        side_history.pop(0)
    side_score = sum(side_history)
    turn_side  = +1 if side_score > 0.05 else (-1 if side_score < -0.05 else 0)
    prev_kf = kf_est
    prev_y  = y

    note = ""

    # search state
    if state == SEARCH:
        y_target = SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * x)
        heading += 0.25 * (np.arctan2(y_target - y, 30.0) - heading)
        dx       = V * np.cos(heading) * DT
        dy_step  = V * np.sin(heading) * DT

        if kf_est > DETECT_THRESH and turn_side != 0:
            turn_dir = turn_side
            state    = TURNING
            note     = f"DETECTED  side={'ABOVE' if turn_dir>0 else 'BELOW'}"

    # turning state
    elif state == TURNING:
        desired_heading = np.arctan2(THERMAL_Y - y, THERMAL_X - x)

        heading_error = (desired_heading - heading + np.pi) % (2*np.pi) - np.pi

        heading += 0.15 * heading_error
        heading = (heading + np.pi) % (2*np.pi) - np.pi

        dx      = V * np.cos(heading) * DT
        dy_step = V * np.sin(heading) * DT

        if kf_est > LOCK_THRESH:
            # Start circling from the aircraft's current position
            circle_cx  = x
            circle_cy  = y + CIRCLE_R
            circle_ang = np.arctan2(y - circle_cy, x - circle_cx)
            circle_r   = CIRCLE_R
            state      = CIRCLE
            note       = f"LOCKED  circle @ ({circle_cx:.0f}, {circle_cy:.0f})"

    # circle state
    elif state == CIRCLE:
        circle_ang += turn_dir * V / circle_r * DT
        new_x   = circle_cx + circle_r * np.cos(circle_ang)
        new_y   = circle_cy + circle_r * np.sin(circle_ang)
        dx      = new_x - x
        dy_step = new_y - y
        heading = np.arctan2(dy_step, dx)

        # Gently nudge circle centre toward thermal
        dist_to_th = np.hypot(circle_cx - THERMAL_X, circle_cy - THERMAL_Y)
        if dist_to_th > 5.0:
            f      = 0.5 * DT / dist_to_th
            new_cx = circle_cx + f * (THERMAL_X - circle_cx)
            new_cy = circle_cy + f * (THERMAL_Y - circle_cy)
            new_r  = np.hypot(x - new_cx, y - new_cy)
            if new_r > 20.0:
                circle_cx, circle_cy = new_cx, new_cy
                circle_r   = new_r
                circle_ang = np.arctan2(y - circle_cy, x - circle_cx)

        if kf_est < ESCAPE_THRESH:
            state   = SEARCH
            heading = 0.0
            note    = "LOST — resuming search"

    # time steps
    x        += dx
    y        += dy_step
    altitude += net_climb * DT
    t        += DT

    log_t.append(t);       log_x.append(x);       log_y.append(y)
    log_alt.append(altitude); log_true_w.append(true_w)
    log_kf.append(kf_est); log_vario.append(vario)
    log_state.append(state); log_cmd.append(turn_cmd)

    if note or step % 20 == 0:
        print(f"{t:6.1f}  {state:<8}  {x:5.0f}  {y:5.0f}  {altitude:6.1f}"
              f"  {true_w:6.3f}  {kf_est:6.3f}  {turn_cmd:7.4f}  {note}")

    if altitude < 10:
        print(f"*** Landed at t={t:.1f}s ***")
        break

# logs
t_arr   = np.array(log_t);    x_arr  = np.array(log_x)
y_arr   = np.array(log_y);    alt_arr= np.array(log_alt)
kf_arr  = np.array(log_kf);   w_arr  = np.array(log_true_w)
v_arr   = np.array(log_vario);cmd_arr= np.array(log_cmd)
st_arr  = log_state

COLORS = {SEARCH: '#4477AA', TURNING: '#EE6677', CIRCLE: '#44BB99'}

# plots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f'Thermal Detection  |  Thermal {"ABOVE" if THERMAL_Y_OFFSET >= 0 else "BELOW"} '
    f'path by {abs(THERMAL_Y_OFFSET):.0f} m  |  k={K_TURN}',
    fontsize=12
)

# Panel 1: 2-D flight path
ax = axes[0]
xg = np.linspace(x_arr.min()-50, x_arr.max()+80, 250)
yg = np.linspace(y_arr.min()-60, y_arr.max()+60, 250)
XG, YG = np.meshgrid(xg, yg)
ax.contourf(XG, YG, thermal_updraft(XG, YG, 0), levels=20, cmap='YlOrRd', alpha=0.4)
for st, col in COLORS.items():
    m = np.array([s == st for s in st_arr])
    if m.any():
        ax.scatter(x_arr[m], y_arr[m], c=col, s=5, label=st, zorder=3)
ax.plot(THERMAL_X, THERMAL_Y, 'r*', ms=14, zorder=6, label='Thermal')
ax.plot(x_arr[0], y_arr[0], 'go', ms=8, zorder=5, label='Start')
x_ref = np.linspace(0, x_arr.max(), 400)
ax.plot(x_ref, SIN_AMP * np.sin(2*np.pi*SIN_FREQ*x_ref), 'w--', lw=1, alpha=0.5)
ax.set_xlabel('East x (m)');  ax.set_ylabel('North y (m)')
ax.set_title('2-D Flight Path')
ax.legend(fontsize=7.5, ncol=2)


# straight line comparison
sl_x, sl_y   = 0.0, 0.0
sl_alt       = ALT_START
sl_t         = 0.0
sl_log_x, sl_log_y, sl_log_alt, sl_log_t = [], [], [], []

for _ in range(int(T_MAX / DT)):
    sl_w       = thermal_updraft(sl_x, sl_y, sl_t)
    sl_alt    += (sl_w - SINK_RATE) * DT
    sl_x      += V * DT          # heading = 0, straight east, y stays 0
    sl_t      += DT
    sl_log_x.append(sl_x);   sl_log_y.append(sl_y)
    sl_log_alt.append(sl_alt); sl_log_t.append(sl_t)
    if sl_alt < 10:
        break

sl_x_arr   = np.array(sl_log_x)
sl_alt_arr = np.array(sl_log_alt)
sl_t_arr   = np.array(sl_log_t)

# Panel 3: straight-line 2-D path
ax = axes[1]
xg2 = np.linspace(-50, max(sl_x_arr.max(), x_arr.max()) + 80, 250)
yg2 = np.linspace(-150, 300, 250)
XG2, YG2 = np.meshgrid(xg2, yg2)
ax.contourf(XG2, YG2, thermal_updraft(XG2, YG2, 0), levels=20, cmap='YlOrRd', alpha=0.4)
ax.plot(sl_x_arr, np.zeros_like(sl_x_arr), color='dodgerblue', lw=2, label='Straight flight')
ax.plot(THERMAL_X, THERMAL_Y, 'r*', ms=14, zorder=6, label='Thermal')
ax.plot(0, 0, 'go', ms=8, zorder=5, label='Start')
ax.set_xlabel('East x (m)');  ax.set_ylabel('North y (m)')
ax.set_title('Straight-Line Flight Path (baseline)')
ax.legend(fontsize=7.5)

plt.tight_layout()
x_max_common = max(x_arr.max(), sl_x_arr.max())

axes[0].set_xlim(0, 550)
axes[1].set_xlim(0, 550)
axes[0].set_ylim(-70,230)
axes[1].set_ylim(-70,230)

# Second figure: altitude comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Altitude Comparison — Steering vs Straight Line', fontsize=12)

# Left: steering aircraft altitude
ax = axes2[0]
for st, col in COLORS.items():
    m = np.array([s == st for s in st_arr])
    if m.any():
        ax.scatter(t_arr[m], alt_arr[m], c=col, s=4, label=st)
ax.axhline(ALT_START, color='gray', lw=0.8, linestyle='--', label='Start alt')
ax.set_xlabel('Time (s)');  ax.set_ylabel('Altitude (m)')
ax.set_title('Steering Aircraft Altitude')
ax.legend(fontsize=7.5)

# Right: straight-line aircraft altitude
ax = axes2[1]
ax.plot(sl_t_arr, sl_alt_arr, color='dodgerblue', lw=1.8, label='Straight flight')
ax.axhline(ALT_START, color='gray', lw=0.8, linestyle='--', label='Start alt')
ax.fill_between(sl_t_arr, ALT_START, sl_alt_arr,
                where=(sl_alt_arr >= ALT_START), alpha=0.2, color='seagreen', label='Gaining')
ax.fill_between(sl_t_arr, ALT_START, sl_alt_arr,
                where=(sl_alt_arr < ALT_START),  alpha=0.2, color='tomato',   label='Losing')
ax.set_xlabel('Time (s)');  ax.set_ylabel('Altitude (m)')
ax.set_title('Straight-Line Aircraft Altitude (no steering)')
ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig('thermal_2d_simple.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → thermal_2d_simple.png")
