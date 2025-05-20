# %% ===== Imports =====
####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from matplotlib.animation import FuncAnimation



# %% ===== Problem parameters =====
####################################
EPS = 0.1
_ALPHA = np.array([1, 4, 16])
T = 1
def trial(x, t):
    global EPS, _N, _ALPHA
    eps = EPS
    alpha = _ALPHA
    S = 0
    for a in alpha:
        S += np.exp(-eps*a**2*t)*np.cos(a*x) 
    return S

def eta(x):
    """IC"""
    return trial(x=x, t=0)

def gL(t):
    """Left BC"""
    return trial(x=-1, t=t)

def gR(t):
    """Right BC"""
    return trial(x=1, t=t)




# %%  ===== Problem definition =====
####################################
xmin = -1 
xmax = 1
M = 101# number of x-grid points (including boundary)
h = (xmax - xmin) / (M-1) # step-size
x_full = np.linspace(xmin, xmax, M) # exclude boundary points
x_interior = x_full[1:-1]

# find optimal k-size from
#        h^2
#  k <=  -----
#       2*eps

k_max = h**2 / (2*EPS)
k = 0.95 * k_max # choose something slighly lower for stability

# see what happens if not stable
# k = 1.05 * k_max 
N = int(T / k) + 1 # Number of time-steps (including zero point [+1])
t_range = np.linspace(0, T, num=N)

mu = k*EPS/(h**2)

print(f"Chosen h : {h}, k : {k}, µ : {mu}")
print(f"x-list : shape={x_interior.shape} [interior points only],\nt-list : shape={t_range.shape}")

# print(f"{h = }")
# print(f"{x_interior = }, shape={x_interior.shape}")
# print(f"Max for stable k : {k_max}")
# print(f"Chosen k : {k}")
# print(f"{t_range = }, shape={t_range.shape}")



# %% ===== Check stability =====
####################################

if mu > 0.5:
    raise ValueError(f"Value of  µ = {mu}  too large -- µ = (k*epsilon)/h**2 should be at most 0.5.\n Check step-sizes `(h,k)`.")

# %% ===== Construct system matrix A and iteration matrix G =====
####################################
#           |   -2       1                       0  |
#           |    1      -2       1                  |
#   A  =    |            1      -2       1          |
#           |                    1      -2       1  |
#           |     0                      1      -2  |
#
#   G  =  I + k*A
#
diag_mid = -2 * np.ones(shape=(M)) # -2 diagonal for each interior point
off_diag = np.ones(shape=(M-1)) # 1 lower and upper diagonal in matrix
A = EPS/h**2 * diags([off_diag, diag_mid, off_diag], [-1, 0, 1], shape=(M,M), format="csc")
G = identity(M, format="csc") + k*A # the sparse identity to keep the sparse matrix formulation
G

# %% ===== Loop for time =====

## Initilize U
# Let the columns and rows correspond to x-values and t-values respectively
# going from first column to last column is the x-space
# going from first (top) row to last (bottom) row corresponds to increasing time
U_full = np.zeros(shape=(N, M)) # [Number of t-values, Number x-points (incl. bounds)]

# Set BC
U_full[:, 0] = gL(t_range) # set left (first column) to have gL(t)
U_full[:, -1] = gR(t_range) # set right (last column) to have gR(t)
# Set IC
U_full[0, 1:-1] = eta(x_interior)
print(U_full != 0) # sort of nice True/False matrix for sanity-check (at least in interactive python window)

## Iterate time
#    n+1                n
#   U    =  (I + kA) * U  + g(t)
#                       
for n in range(N-1):
    U_full[n+1, 1:-1] = (G@U_full[n, :])[1:-1] # only update interior points
print("Simulation done")
    
if SANITY_CHECK := True:
    print("Ensure BC")
    print(f"u(-1, t) = g_L(t) : {np.all(U_full[:,0] == gL(t_range))} ")
    print(f" u(1, t) = g_R(t) : {np.all(U_full[:,-1] == gR(t_range))} ")
    print("Ensure IC")
    print(f" u(x, 0) = eta(x) : {np.all(U_full[0,1:-1] == eta(x_interior))} ")
    # ensure that BC holds


# %%  ==== Plot ====
LABELSIZE = 15
print("Plot!")
fig, ax = plt.subplots(figsize=(7,4))

line, = ax.plot(x_full, U_full[0, :])
title_obj = ax.set_title(f"1D Heat equation for $t = {t_range[0]:.2f}$", size=LABELSIZE+2)
ax.set_xlabel(r"$x$", size=LABELSIZE-2)
ax.set_ylabel(rf"$u(x,t)$", size=LABELSIZE-2)
ax.set_ylim(U_full.min() * 1.1, U_full.max() * 1.1) # constant y-range
ax.grid()

def animate(frame):
    line.set_ydata(U_full[frame, :])
    # ax.set_title(f"1D Heat equation for $t = {t_range[frame]:.2f}$")
    ax.set_title(f"1D Heat equation for $t = {t_range[frame]:.2f}$", size=LABELSIZE+2)
    # ax.set_ylabel(rf"$u(x,{t_range[frame]:.2f})$", size=LABELSIZE-2)
    print(f"{frame = }")
    return line,

fig.tight_layout()
ani = FuncAnimation(fig=fig, func=animate, frames=list(range(0, 100, N//100)) + list(range(100, N, N//20)), blit=False)
plt.show()

# %%
T_min = 0
T_max = 101
fig, ax = plt.subplots()
cax = ax.imshow(U_full[T_min:T_max,:], vmin=U_full.min(), vmax=U_full.max())
x_idx = np.arange(len(x_full))[::25]
t_idx = np.arange(len(t_range[T_min:T_max]))[::25]
ax.set_xticks(x_idx)
ax.set_xticklabels(x_full[x_idx])
ax.set_xlabel(r"$x$", size=LABELSIZE-2)

ax.set_yticks(t_idx)
ax.set_yticklabels(np.round(t_range[t_idx], 3))
ax.set_ylabel(r"$t$", size=LABELSIZE-2)

ax.set_title("1D heat equation for $u(x,t)$", size=LABELSIZE+2)
fig.colorbar(cax, ax=ax)
fig.tight_layout()

# %%  ==== make it all a function

class Parabolic:
    def __init__(self, M, IC_func, BC_funcs : tuple, **KWARGS):
        _default_params = {"xmin" : -1,
                          "xmax" : 1,
                          "T_max" : 1,
                          "eps" : 0.1}
        # default parameters
        for k, v in KWARGS.items():
            setattr(self, k, v)
            
        for k,v in _default_params.items():
            if not k in KWARGS:
                setattr(self, k, v)
        
        self.M = M
        self.h = (self.xmax - self.xmin) / (self.M - 1)
        self._ic_func = IC_func
        self._b_funcs = BC_funcs
        self.x_full = np.linspace(self.xmin, self.xmax, num=self.M)
        self.x_interior = self.x_full[1:-1]
        
        self.stable_k_max = self.h**2 / (2*self.eps)
        self.k = 0.95 * self.stable_k_max
        
        
        
