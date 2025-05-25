# %% ==== Imports ====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# %% ==== make directory for files ====

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)


# %% define trial function and IC/BC
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
    """Make an object for solving Burgers equation from exercise 4
    
    Parameters
    ----
    M : int, 
        number of spatial interior points
    IC_func : function,
        Initial value function
    BC_funcs : tuple[func, func],
        Tuple of boundary value function, usch as `(gL, gR)`
    
    *Optional*,
        Can be se with keyword arguments, but are not neccessary (will use default values)
    xmin : int, (default -1)
    
    xmax : int, (default 1)
    
    T_max : int,
        highest time value (default 2)
        
    eps : float, (default 0.1)
        diffusion constant
        
    use_safe_k : bool, (default False),
        scale the k-value by 0.95 to be conservative
        
    use_adaptive_k : bool, (default True),
        if False the restraint on k from the advection term will use h / 30,
        otherwise  k = h/max_abs_U  calculated from previous step
        
    atol : float, (default 1e-6),
        if `max_abs_u < atol` (close to zero) use this value to avoid ZeroDivision-errors
    
    max_step : int,
        Maximum number of steps before forcefull termination when solver the PDE
    
    
    """
    
    
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
            "max_step" : 2*1e5,
        }
        self.kwargs = KWARGS
        # Set attribute from keyword arguments (KWARGS)
        for key, value in KWARGS.items():
            setattr(self, key, value)
            
        # set attribute from defualt values if value not in KWARGS
        for key, value in _default_params.items():
            if not key in KWARGS:
                setattr(self, key, value)
        
        # Set IC/BC attribute
        self.ic_func = IC_func
        self.bc_funcs = BC_funcs
        # Define x-grid and spacing
        self._spatial_grid(M)
        # Compute solution
        self.compute_solution()
        # Set solution shape to be an attribute of the class
        self.shape = (self.N, self.M)
    
    def _spatial_grid(self, M):
        """Determine spatial grid.
        Only for internal use
        
        Parameters
        ---
        M : int, 
            number of interior points
        """
        self.M = M
        self.h = np.abs(self.xmax - self.xmin) / (self.M + 1)
        self.x_full = np.linspace(self.xmin, self.xmax, num=M+2)
        self.x_interior = self.x_full[1:-1]
    
    
    def _adaptive_k_step(self, max_abs_U):
        """Determine temporal step size from the solution of previous time-step. 
        Only for internal use
        
        Paramters
        ---
        max_abs_U : float,
            maximum of the absolute value of U from previous time-step.
        
        Notes
        ---
        If object has set `use_adaptive_k = False`, will force `max_abs_U = 30`
        
        Returns
        ---
        k : flaot
        """
        k_max_diffusion = self.h**2 / (2*self.eps)
        k_max_advection = self.h / max_abs_U
        
        k = np.min([k_max_advection, k_max_diffusion])
        
        if self.use_safe_k:
            k = 0.95 * k
        return k
        
    
    def compute_solution(self):
        """Compute solution to Brugers equation"""
        
        # get BC functions as callables
        gL, gR = self.bc_funcs
        # Initilize `current`` values
        U_interior_current = eta(self.x_interior)
        t_current = 0
        U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
        
        # Initialize list for saving values
        U_full_history = []
        t_history = []
        
        # appennd values to list
        U_full_history.append(U_full_current)
        t_history.append(t_current)
        
        # loop until complete OR until `step_count > max_step``
        step_count = 0
        while t_current < self.T_max:
            step_count += 1
            
            # determine if adaptive k is enabled and account for ZeroDivion by using `atol``
            if self.use_adaptive_k:
                U_current_max = np.max(np.abs(U_full_current))
                if U_current_max < self.atol:
                    print(f"max near zero -- using atol :  {self.atol}")
                    U_current_max = self.atol
            else:
                U_current_max = 30
            
            # set next k-size from previous U_max
            k = self._adaptive_k_step(U_current_max)
            
            # make last step be equal to T_max
            if k + t_current > self.T_max:
                k = T_max - t_current 
            
            ## Computed diffusion components of the stencil
            U_im1_n = U_full_current[:-2]
            U_i_n = U_full_current[1:-1]
            U_ip1_n = U_full_current[2:]
            diffusion_term = (EPS/self.h**2) * (U_ip1_n + U_im1_n - 2*U_i_n)
            
            ## Compute advection components of stencil
            # F_im1_n = 0.5*U_im1_n**2 / self.h
            F_i_n   = 0.5*U_i_n**2 / self.h
            F_ip1_n = 0.5*U_ip1_n**2 / self.h
            
            # advection_backwards = (F_i_n - F_im1_n)
            advection_forwards = (F_ip1_n - F_i_n) 
            
            # advection_term = np.where(U_i_n >= 0, advection_backwards, advection_forwards)
            advection_term = advection_forwards
            
            # Set the next time-step values using stencil
            U_interior_next = U_i_n + k*(diffusion_term - advection_term)            
            
            # increase t and set U from `next` to `current`
            t_current += k
            U_interior_current = U_interior_next
            
            # store full solution
            U_full_current = np.concatenate([[gL(t_current)], U_interior_current, [gR(t_current)]])
            U_full_history.append(U_full_current)
            t_history.append(t_current)
            
            # Break loop if too many steps are taken
            if step_count > self.max_step:
                print(f"Stopped loop prematurely.\n Used  {step_count}  steps")
                break
        
        # Make the solution and t-values numpy-arrays
        self.U_full_history = np.asarray(U_full_history)
        self.t_history = np.asarray(t_history)
        self.N = len(t_history)
        print(f"## Computation completed in  {step_count}  steps")
    
    def plot_solution(self, time_idxs, exact_solution = None, *, LABELSIZE=16, figname=None):
        """Plot solution for some value of time
        
        Parameters
        ---
        time_idxs : int, list[int,],
            index for the t-array to plot. Can be `int` for a single plot, or `list of int`.
            
        exact_solution : function, (default None):
            If None: will only plot FDM.
        
        *Optional*,
            These can only be set using keywords
        LABELSIZE : int, (default 16),

            Relative labelsize for labels, titles, legends, and tickmarks

        figname : str, (default None),
            Save plot to name of the string `figname`. If None: do not save figure

        """
        
        title = lambda idx: fr"$t = {self.t_history[idx]:.4f}$"
        if type(time_idxs) is int: # handle interger cases
            time_idxs = np.array([time_idxs,])
        else:
            time_idxs = np.asarray(time_idxs)
        
        if np.any(time_idxs >= self.N): # ensure that time-index are valid for the solution
            raise ValueError(f"Time index   {time_idxs}   too large.\n All should be < N = {self.N}")
        
        ymin = self.U_full_history.min()
        ymax = self.U_full_history.max()
        
        fig, ax = plt.subplots(1, len(time_idxs), figsize=(10,4))
        fig.suptitle(fr"Burgers Equation using  $\epsilon = {self.eps:.4f}$", size=LABELSIZE+4)
        if len(time_idxs) == 1:
            ax = [ax,]
        
        # plot solution for each of the time-indicies supplied
        for i, frame in enumerate(time_idxs):
            ax[i].set_title(title(frame), size=LABELSIZE+3)
            ax[i].plot(self.x_full, self.U_full_history[frame, :], label="FDM")
        
            # If an exact solution is give, inlcude this plot.
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
        if not figname is None: # save figure if a name is given
            print("save figure!")
            fig.savefig(fname=figname)
        plt.show()
        
    def animate_solution(self, exact_solution = None, *, LABELSIZE=15):
        """Animation plot using `matplotlib.animation.FuncAnimation`
        Plot solution for each time-step; overwriting the previous plot. Nice for illustratory purposes
        
        Parameters
        ---
        exact_solution : function, (default None):
            Plot FDM solution together with exact solution if supplied. If None: only plot FDM.
        
        *Optional*,
            These can only be used with the corresponding keywords
        LABELSIZE : int, (default 15),
            Realtive labelsize for: title, labels, legends, tickmarks.
        """
        
        # Start by plotting the initial values
        title = lambda idx: "Burgers Equation\n" + fr"($t = {self.t_history[idx]:.4f}$,  $\epsilon = {self.eps:.3f}$)"
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_title(title(0), size=LABELSIZE+4)
        line_fdm, = ax.plot(self.x_full, self.U_full_history[0,:], label="FDM")
        
        ymin = self.U_full_history.min()
        ymax = self.U_full_history.max()
        
        # include exact solution if supplied
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
        
        # internal function determining what should change between each frame of animation
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
                line_exact.set_ydata(U_exact[frame, :])
            return line_fdm
        
        fig.tight_layout()
        
        # set list of frame-values to plot using some stepsize. The lower the ratio  `self.N / 20` is the more frames are shown
        frames = list(range(0, self.N - 1, int(self.N / 20)))
        
        # animation call
        ani = FuncAnimation(fig, func=animate, frames=frames, blit=False)
        
        plt.show()
        
    def calculate_error(self, exact_solution):
        """Compute error between exact solution and FDM solution
        
        Parameters
        ---
        exact_solution : function,
            callable function of the form u(x,t). If  x,t  are arrays, use 1D arrays! 
        
        Returns
        ---
        max of absolute difference (float)
        """
        U_exact = exact_solution(self.x_full[None, :], self.t_history[:, None]) # set rows to correspond to increaseing time, and columns to increasing space
        difference = self.U_full_history - U_exact
        
        return np.max(np.abs(difference**2))     
        
    def convergence(self, M_list, exact_solution):
        """Compute errors using a list of interior spatial points and exact solution
        
        Paramters
        ---
        M_list : list[int,],
            list of integers corresponding to `M` values.
        exact_solution : function,
            callable function of the form `u(x,t)`
            
        Returns
        ---
        hs : ndarray,
            values of h for each value of M supplied
        errors : ndarray,
            errors computed for each value of M, using the `calculate_error` method above.
        """
        hs = []
        errors = []
        for M in M_list:
            _iter = BurgersEquation(M, IC_func=self.ic_func, BC_funcs=self.bc_funcs, eps=self.eps)
            err = _iter.calculate_error(exact_solution)
            h = _iter.h
            # k = _iter.k
            
            # ks.append(k)
            hs.append(h)
            errors.append(err)
        
        return np.asarray(hs), np.asarray(errors)
    
    def plot_convergence(self, M_list, exact_solution, *, LABELSIZE=15, figname=None):
        """Plot convergence for FDM.
        
        Paramters
        ---
        M_list : list[int,],
            list of integers corresponding to `M` values.
        exact_solution : function,
            callable function of the form `u(x,t)`
        
        *Optional*,
            The following can only be used with the corresponding keywords
        
        LABELSIZE : int, (default 15),
            Relative labelsize for title, labels, legends, and tickmarks
        
        figname : str, (default None):
            name of figure. If given a string, the figure is save to this name. If None: do not save figure.
        """
        hs, error = self.convergence(M_list=M_list, exact_solution=exact_solution)
        
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        
        ax.loglog(hs, error, "-ob", label="FDM")
        ax.loglog(hs, hs**2, "-k", label=r"$\mathcal{O}(h^2)$")
        
        ax.set_xlabel("step-size $h$", size=LABELSIZE+1)
        ax.set_ylabel("Error", size=LABELSIZE+1)
        ax.tick_params(axis="both", which="major", labelsize=LABELSIZE-2)
        
        fig.suptitle(fr"Error plot ($\epsilon = {self.eps:.4f}$)", size=LABELSIZE+4)
        ax.grid()
        ax.legend(fontsize=LABELSIZE-2)
        fig.tight_layout()
        
        if not figname is None:
            fig.savefig(figname)
        
        plt.show()
                
            
            
        
            

# %% ==== Plot solution ====

IC = eta
BCs = (gL, gR)
M = 800
test = BurgersEquation(M, IC_func=IC, BC_funcs=BCs, use_adaptive_k=True)

test.plot_solution([test.N // 2], exact_solution=trial, 
                #    figname="assignment2/figures/ex4_sol.png"
                   figname="figures/ex4_sol.png",
                   )


# %% ==== plot animation of solution ====
test.animate_solution(exact_solution=trial)


# %% ==== plot convergence of solution ====
test.plot_convergence([2**i * 50 for i in range(5)], 
                      exact_solution=trial, 
                      LABELSIZE=16,
                    #   figname="assignment2/figures/ex4_trial_convergence.png",
                      figname="figures/ex4_trial_convergence.png",
                      )

# %% ==== Using specified IC/BC
#### EXERCISE 4.3)
print("Test 2:")
new_ic = lambda x: -np.sin(np.pi*x)
new_gL = lambda t: 0*t
new_gR = lambda t: 0*t
M = 300

test2 = BurgersEquation(M, IC_func=new_ic, BC_funcs=(new_gL, new_gR), eps=0.01/np.pi, use_adaptive_k=False)
test2.plot_solution([0, 3000, 4000], 
                    # figname="assignment2/figures/ex4_sol.png"
                    figname="figures/ex4_3_sol.png"
                    )
test2.animate_solution()

# %% Estimate value
##   
##     d u(x,t) |
##    --------- |              = -152.00516
##       d x    |(x=0, t=1.6037/pi)

t_investigate = np.argwhere(np.isclose(test2.t_history, 1.6037 / np.pi, 1e-2))
t_idx = t_investigate[len(t_investigate) // 2]
print(f"close to 1.6037 / pi = {1.6037/np.pi} :", t_idx)
test2.plot_solution(t_idx, 
                    # figname="assignment2/figures/ex4_3-time-specific.png"
                    # figname="figures/ex4_3-time-specific.png"
                    figname=None
                    )
U_i = test2.U_full_history[t_idx, test2.M // 2][0]
U_ip1 = test2.U_full_history[t_idx, test2.M // 2 + 1][0]
dudx = (U_ip1 - U_i ) / (test2.h)

digits = 5
print(f"  {U_i = :>{digits+3}.{digits}f}")
print(f"{U_ip1 = :>{digits+3}.{digits}f}")
print(f" {dudx = :>{digits+3}.{digits}f}")
# %%
