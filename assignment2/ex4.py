# %% ==== Imports ====
import numpy as np
from scipy.sparse import identity, diags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %% ==== Parameters ====

EPS = 0.1
xmin = -1
xmax = 1
T_max = 2

# %% ==== trial and BC/IC ====

def trial(x,t):
    global EPS
    eps = EPS
    return -np.tanh((x + 0.5 - t) / (2*eps)) + 1

def gL(t):
    return trial(-1, t)

def gR(t):
    return trial(1, t)

def eta(x):
    return trial(x, 0)



# %% ==== discretize parameters ====

M = 100 # number of interior spatial points 
h = np.abs(xmax - xmin) / (M+1)
x_full = np.linspace(xmin, xmax, num=M+2)
x_interior = x_full[1:-1]

# %% ==== Stability condition ====

k_max_diffusion = h**2 /(2*EPS) # max k-size for diffusion part

# ==== adaptive k step ====
def adaptive_k_step(abs_U_max):
    k_max_advection = h / abs_U_max
    # k_max_advection = h / 2
    return np.min([k_max_advection, k_max_diffusion])


# %% ==== initialize solution ====
U_interior_current = eta(x_interior)
t_current = 0
U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])

# %% ==== Time-step ====

atol = 1e-4
U_full_history = []
t_history = []
U_full_history.append(U_full_current)
t_history.append(t_current)

step_count = 0
while t_current < T_max: # loop time-step
    step_count += 1
    
    if step_count % 100 == 0:
        print(f"Time-step count : {step_count}")
    
    U_current_max = np.max(np.abs(U_full_current))
    
    if U_current_max < atol: # set lower limit for how small U_max_n is -- avoid division by zero
        print(f"max near zero  -- using atol : {atol:.2e}")
        U_current_max = atol
    
    k = adaptive_k_step(U_current_max)
    
    if k + t_current > T_max:
        k = T_max - t_current # make last step be equal to T_max
        
    # U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
    U_interior_next = np.zeros_like(U_interior_current)
    

    U_im1_n = U_full_current[:-2]
    U_i_n = U_full_current[1:-1]
    U_ip1_n = U_full_current[2:]
    diffusion_term = (EPS/h**2) * (U_ip1_n + U_im1_n - 2*U_i_n)
    
    F_im1_n = 0.5*U_im1_n**2 / h
    F_i_n   = 0.5*U_i_n**2 / h
    F_ip1_n = 0.5*U_ip1_n**2 / h
    
    advection_backwards = (F_i_n - F_im1_n)
    advection_forwards = (F_ip1_n - F_i_n) 
    
    advection_term = np.where(U_i_n >= 0, advection_backwards, advection_forwards)
    advection_term = advection_forwards
    U_interior_next = U_i_n + k*(diffusion_term - advection_term)
    
    # iterate
    t_current = t_current + k
    U_interior_current = U_interior_next
    
    # store full solution
    U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
    U_full_history.append(U_full_current)
    t_history.append(t_current)
    # if step_count > 1e4:
    #     print("Couldn't complete loop")
    #     break

U_full_history = np.asarray(U_full_history)
t_history = np.asarray(t_history)
N_final = len(t_history)

print("## computation complete")
# U_full_history
# %% ==== plots ====
U_exact = trial(x_full[None, :], t_history[:, None])
title = lambda idx: f"Burgers Equation (time = {t_history[idx]:.4f})"

fig, ax = plt.subplots()
line_num, = ax.plot(x_full, U_full_history[0, :], "-b", label="FDM")
line_exact, = ax.plot(x_full, U_exact[0,:], "-k", label="Exact")

title_obj = ax.set_title(title(idx=0))
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x,t)$")
ax.set_ylim(0.9*np.min([U_exact.min(), U_full_history.min()]), 1.1*np.max([U_exact.max(), U_full_history.max()]))
# ax.set_ylim(0.9*U_exact.min(), 1.1*U_exact.max())
ax.grid()
ax.legend(loc="lower right")


def animate(frame):
    line_num.set_ydata(U_full_history[frame, :])
    line_exact.set_ydata(U_exact[frame, :])
    ax.set_title(title(idx=frame))
    return line_exact,

animate_frame = list(range(0, N_final - 1, int(N_final/30)))

ani = FuncAnimation(fig, animate, frames=animate_frame, blit=False)
plt.show()

# %%
print(f" Exact shape : {U_exact.shape},\t N_final : {N_final}")
print(f" Exact  (min, max) :  ({U_exact.min()} , {U_exact.max()})")

# %% ==== heatmap ==== 
idx_max = 400
fig, ax = plt.subplots(1,3, figsize=(8,6))
ax = ax.ravel()
vmin = np.min([U_full_history.min(), U_exact.min()])
vmax = np.max([U_full_history.max(), U_exact.max()])
ax[0].imshow(U_full_history[:idx_max], 
    # vmin=vmin, vmax=vmax
    )
ax[0].set_title(f"FMD")

cax = ax[1].imshow(U_full_history[:idx_max] - U_exact[:idx_max], 
    # vmin=vmin, vmax=vmax
    )
ax[1].set_title("Difference")

ax[2].imshow(U_exact[:idx_max], 
    # vmin=vmin, vmax=vmax
    )
ax[2].set_title("Exact")

fig.colorbar(cax)

fig.tight_layout()
plt.show()

# %%
