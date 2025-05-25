# %% ==== Imports ====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%
EPS = 0.1
xmin = -1
xmax = 1
T_max = 2

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


# %% define class

class BurgersEquation:
    """Make an object for solving Burgers equation from exercise 4"""
    
    def __init__(self, M, IC_func, BC_funcs, **KWARGS):
        # a default array for values usually not changed in this problem
        _default_params = {
            "xmin" : -1,
            "xmax" : 1,
            "T_max" : 2,
            "eps" : 0.1,
            "use_safe_k" : False,
            "use_adaptive_k" : True,
            "atol" : 1e-6,
            "max_step" : 2*1e4,
        }
        # Set attribute from keyword arguments (KWARGS)
        for key, value in KWARGS.items():
            setattr(self, key, value)
            
        # set attribute from defualt values if value not in KWARGS
        for key, value in _default_params.items():
            if not key in KWARGS:
                setattr(self, key, value)
        
        self.ic_func = IC_func
        self.bc_funcs = BC_funcs
        self._spatial_grid(M)
        self.compute_solution()
        self.shape = (self.N, self.M)
    
    def _spatial_grid(self, M):
        self.M = M
        self.h = np.abs(self.xmax - self.xmin) / (self.M + 1)
        self.x_full = np.linspace(self.xmin, self.xmax, num=M+2)
        self.x_interior = self.x_full[1:-1]
    
    
    def _adaptive_k_step(self, abs_U_max):
        k_max_diffusion = self.h**2 / (2*self.eps)
        k_max_advection = self.h / abs_U_max
        
        k = np.min([k_max_advection, k_max_diffusion])
        
        if self.use_safe_k:
            k = 0.95 * k
        return k
        
    
    def compute_solution(self):
        gL, gR = self.bc_funcs
        U_interior_current = eta(self.x_interior)
        t_current = 0
        U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
        
        U_full_history = []
        t_history = []
        
        U_full_history.append(U_full_current)
        t_history.append(t_current)
        
        step_count = 0
        while t_current < self.T_max:
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Time-step count : {step_count}")
            
            if self.use_adaptive_k:
                U_current_max = np.max(np.abs(U_full_current))
                if U_current_max < self.atol:
                    print(f"max near zero -- using atol :  {self.atol}")
                    U_current_max = self.atol
            else:
                U_current_max = 30
            
            k = self._adaptive_k_step(U_current_max)
            
            if k + t_current > self.T_max:
                k = T_max - t_current # make last step be equal to T_max
            
            # U_interior_next = np.zeros_like(U_interior_current)
            
            U_im1_n = U_full_current[:-2]
            U_i_n = U_full_current[1:-1]
            U_ip1_n = U_full_current[2:]
            diffusion_term = (EPS/self.h**2) * (U_ip1_n + U_im1_n - 2*U_i_n)
            
            F_im1_n = 0.5*U_im1_n**2 / self.h
            F_i_n   = 0.5*U_i_n**2 / self.h
            F_ip1_n = 0.5*U_ip1_n**2 / self.h
            
            advection_backwards = (F_i_n - F_im1_n)
            advection_forwards = (F_ip1_n - F_i_n) 
            
            advection_term = np.where(U_i_n >= 0, advection_backwards, advection_forwards)
            advection_term = advection_forwards
            U_interior_next = U_i_n + k*(diffusion_term - advection_term)            
            
            t_current += k
            U_interior_current = U_interior_next
            
            # store full solution
            U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
            U_full_history.append(U_full_current)
            t_history.append(t_current)
            
            if step_count > self.max_step:
                print(f"Stopped loop prematurely.\n Used  {step_count}  steps")
                break
        
        self.U_full_history = np.asarray(U_full_history)
        self.t_history = np.asarray(t_history)
        self.N = len(t_history)
        print("## Computation complete")
    
    def plot_solution(self, time_idxs, exact_solution = None, *, LABELSIZE=16, figname=None):
        title = lambda idx: fr"$t = {self.t_history[idx]:.4f}$"
        if type(time_idxs) is int:
            time_idxs = np.array([time_idxs,])
        else:
            time_idxs = np.asarray(time_idxs)
        
        if np.any(time_idxs >= self.N):
            raise ValueError(f"Time index   {time_idxs}   too large.\n All should be < N = {self.N}")
        
        ymin = self.U_full_history.min()
        ymax = self.U_full_history.max()
        
        fig, ax = plt.subplots(1, len(time_idxs), figsize=(10,4))
        fig.suptitle(fr"Burgers Equation using  $\epsilon = {self.eps:.4f}$", size=LABELSIZE+4)
        if len(time_idxs) == 1:
            ax = [ax,]
        # print(len(time_idxs))
        for i, frame in enumerate(time_idxs):
            ax[i].set_title(title(frame), size=LABELSIZE+3)
            ax[i].plot(self.x_full, self.U_full_history[frame, :], label="FDM")
        
            if not exact_solution is None:
                U_exact_n = exact_solution(self.x_full, self.t_history[frame])
                ymin = np.min([ymin, U_exact_n.min()])
                ymax = np.max([ymax, U_exact_n.max()])
                ax[i].plot(self.x_full, U_exact_n, label="Exact")
            
            ax[i].set_xlabel(r"$x$", size=LABELSIZE-2)
            ax[i].set_ylabel(rf"$u(x,t)$", size=LABELSIZE-2)
            try:
                ax[i].set_ylim(0.9*ymin, 1.1*ymax)
            except:
                print(f"cannot set y-limits for  t = {self.t_history[frame]}")
            ax[i].tick_params(axis="both", which="major", labelsize=LABELSIZE-4)
            
            ax[i].grid()
            ax[i].legend(loc="lower right", fontsize=LABELSIZE-2)
        
        fig.tight_layout()
        
        if not figname is None:
            print("save figure!")
            fig.savefig(fname=figname)
        plt.show()
        
    def animate_solution(self, exact_solution = None, *, LABELSIZE=15):
        title = lambda idx: "Burgers Equation\n" + fr"($t = {self.t_history[idx]:.4f}$,  $\epsilon = {self.eps:.3f}$)"
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_title(title(0), size=LABELSIZE+4)
        line_fdm, = ax.plot(self.x_full, self.U_full_history[0,:], label="FDM")
        
        ymin = self.U_full_history.min()
        ymax = self.U_full_history.max()
        
        if not exact_solution is None:
            print("Exact solution applied")
            U_exact = exact_solution(self.x_full[None, :], self.t_history[:, None])
            line_exact, = ax.plot(self.x_full, U_exact[0, :], label="Exact")
            ymin = np.min([ymin, U_exact.min()])
            ymax = np.max([ymax, U_exact.max()])
        
        ax.set_xlabel(r"$x$", size=LABELSIZE-2)
        ax.set_ylabel(rf"$u(x,t)$", size=LABELSIZE-2)
        
        try:
            ax.set_ylim(0.9*ymin, 1.1*ymax)
        except ValueError:
            print("Cannot determine y-limits")
            
        ax.tick_params(axis="both", which="major", labelsize=LABELSIZE-4)
        
        ax.grid()
        ax.legend(loc="lower right", fontsize=LABELSIZE-2)            
        
        
        def animate(frame):
            print(f"{frame = }")
            ax.set_title(title(frame), size=LABELSIZE+4)
            line_fdm.set_ydata(self.U_full_history[frame, :])
            try:
                ax.set_ylim(0.9*ymin, 1.1*ymax)
            except ValueError:
                print("Cannot determine y-limits")
                
            if not exact_solution is None:
                print("Exact solution applied")
                # U_exact_n = exact_solution(self.x_full, self.t_history[frame])
                line_exact.set_ydata(U_exact[frame, :])
            return line_fdm
        
        fig.tight_layout()
        frames = list(range(0, self.N - 1, int(self.N / 20)))
        ani = FuncAnimation(fig, func=animate, frames=frames, blit=False)
        
        plt.show()
            

# %% ==== Compute solution ====

# IC = eta
# BCs = (gL, gR)
# M = 100
# test = BurgersEquation(M, IC_func=IC, BC_funcs=BCs, use_adaptive_k=False)
# # %% ==== Plot solution ===

# test.plot_solution([0, 400, test.N-200], exact_solution=trial, figname="assignment2/figures/ex4_sol.png")
        
# # %% ==== plot animation of solution ====

# test.animate_solution(exact_solution=trial)

# %% ==== Using specified IC
print("Test 2:")

new_ic = lambda x: -np.sin(np.pi*x)

new_gL = lambda t: 0*t
new_gR = lambda t: 0*t

test2 = BurgersEquation(M=300, IC_func=new_ic, BC_funcs=(new_gL, new_gR), eps=0.01/np.pi, use_adaptive_k=False)
test2.plot_solution(10, 
                    figname="assignment2/figures/ex4_3-time-specific.png"
                    # figname="figures/ex4_3-time-specific.png"
                    )
# test2.animate_solution()

# %% Estimate value
##   
##     d u(x,t) |
##    --------- |              = -152.00516
##       d x    |(x=0, t=1.6037/pi)

print(f"N = {test2.N}")
t_investigate = np.argwhere(np.isclose(test2.t_history, 1.6037 / np.pi, 1e-2))
print(f"{t_investigate = }")
t_idx = t_investigate[len(t_investigate) // 2 - 1]
print(f"close to 1.6037 / pi = {1.6037/np.pi} :", t_idx)
test2.plot_solution(t_idx, 
                    # figname="assignment2/figures/ex4_3-time-specific.png"
                    figname="figures/ex4_3-time-specific.png"
                    )
print(f"shape = {test2.shape}")
U_im1 = test2.U_full_history[t_idx, test2.M // 2 - 1]
U_ip1 = test2.U_full_history[t_idx, test2.M // 2 + 1]

dudx = (U_ip1 - U_im1 ) / (2*test2.h)
print(dudx)
# %%
