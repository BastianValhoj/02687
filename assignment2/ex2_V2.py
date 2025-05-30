# %% ===== Imports =====
####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from matplotlib.animation import FuncAnimation
from pathlib import Path

# %% ==== make directory for figures ====
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

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

# %%  ==== make it all a function

class Parabolic:
    """Class for computing solution to Parabolic PDE for the 1D heat equation.
    
    Parameters
    ---
    M : int,
        number of x-points to use (interior + boundary)
    IC_func : func,
        function for initial values
    BC_funcs : tuple(func,)
        tuple of functions for boundary values. Use as: `BC_funcs = (BC_left, BC_right)`
    
    *KWARGS*   

    xmin : int, (default -1)  

    xmax : int, (default 1)  

    T_max : int, (default 1)  
        max value of time array (and we always start a t=0)
    eps : float, (default 0.1)

    
    These parameters are used explicitly, but you may want to edit these
    """
    def __init__(self, M, IC_func, BC_funcs : tuple, **KWARGS):
        # a default array for values usually not changed in this problem
        _default_params = {"xmin" : -1,
                          "xmax" : 1,
                          "T_max" : 1,
                          "eps" : 0.1,
                          "use_safe_k" : True}
        # Set attribute from keyword arguments (KWARGS)
        for k, v in KWARGS.items():
            setattr(self, k, v)
        
        # set attribute from defualt values if value not in KWARGS
        for k, v in _default_params.items():
            if not k in KWARGS:
                setattr(self, k, v)
        
        self.ic_func = IC_func
        self.bc_funcs = BC_funcs
        self.mesh_size(M, use_safe_k=self.use_safe_k)
        self.define_grid()
        
        
    def mesh_size(self, M, use_safe_k=True):
        """Define meshgrid from input M.
        
        Paramters
        ---
        M : int, 
            Size of x-grid (interior points + boundary)
        
        *Optional*  
        use_safe_k : bool, (default `True`)
            if `True` scale the max stable by 0.95 to ensure stability.
            If `False` use k = h^2/(2*epsilon)
            
        
        """
        self.M = M
        self.h = (self.xmax - self.xmin) / (self.M - 1)
        stable_k_max = self.h**2 / (2*self.eps)
        if use_safe_k:
            self.k = 0.95 * stable_k_max
        else:
            self.k = stable_k_max
        self.mu = self.eps*self.k / (self.h**2)
        self.N = int(self.T_max / self.k) + 1
        
        if self.mu > 0.5:
            self.warn = f"Value of   µ : {self.mu:.6f} > 0.5   System not stable! "
        else:
            self.warn = f"System stable for k : {self.k:.6f},\t h : {self.h:.6f}"
        
        print(self.warn)
    
    def define_grid(self):
        """Define grid for x-values and t-values using meshgrid"""
        self.x_full = np.linspace(self.xmin, self.xmax, num=self.M)
        self.x_interior = self.x_full[1:-1]
        self.t_range = np.linspace(0, self.T_max, num=self.N)
        self.U_solution = self.iterate_time()
        
    
    def CS_matrix(self):
        """Compute Central Space matrix
                    |   -2       1                       0  |
                    |    1      -2       1                  |
            A  =    |           ...     ...     ...         |
                    |                    1      -2       1  |
                    |     0                      1      -2  |
        """
        off_diag = np.ones(shape=(self.M -1))
        mid_diag = -2*np.ones(shape=(self.M))
        A = self.eps/self.h**2 * diags([off_diag, mid_diag, off_diag], 
                                        [-1, 0, 1],
                                        shape=(self.M,self.M),
                                        format="csc")
        return A
    
    def FT_matrix(self):
        """Compute Foward Time matrix
            G = I + k*A 
        using sparse matricies
        """
        A = self.CS_matrix()
        G = identity(self.M, format="csc") + self.k*A
        return G
    
    def Initialize(self):
        """Initialize the array using IC and BC"""
        
        IC = self.ic_func
        BC_L, BC_R = self.bc_funcs
        
        U_init = np.zeros(shape=(self.N, self.M)) #
        # set IC
        U_init[0, 1:-1] = IC(self.x_interior)
        
        # set BC
        U_init[:, 0] = BC_L(self.t_range)
        U_init[:, -1] = BC_R(self.t_range)
        
        return U_init
    
    def iterate_time(self):
        """Iterate the Forward time stencil
             {n+1}        {n}
            U      = G * U
        
        """
        U_full = self.Initialize()
        G = self.FT_matrix()
        for n in range(self.N -1):
            U_n = G@U_full[n, :]
            U_full[n+1, 1:-1] = U_n[1:-1]
        
        return U_full
    
    def plot_sol(self, frames, *, LABELSIZE=15):
        """Plot solution for some index of the time array
        Plot solution for each value of t.
        
        Parameters
        ----
        frame : int or list(int,), 
            Index for time to plot
        *Optional and using keyword*  
        LABELSIZE : int, (default 15)
            The labelsize for refence to use. Title gets +2 and labels get -2
        """
        if type(frames) is int:
            fig, ax = plt.subplots(1, 1, figsize=(8,6))
            ax = [ax,]
            frames = [frames,]
        else:
            fig, ax = plt.subplots(1, len(frames), figsize=(8,4), sharey=True)
        
        # ax[0].plot(self.x_full, self.U_solution[frames[0], :])
        for i, frame in enumerate(frames):
            ax[i].plot(self.x_full, self.U_solution[frame, :])
            ax[i].set_title(f"$t = {self.t_range[frame]:.2f}$", size=LABELSIZE+1)
            ax[i].set_xlabel(r"$x$", size=LABELSIZE)
            ax[i].set_ylabel(rf"$u(x,t)$", size=LABELSIZE)
            ax[i].set_ylim(self.U_solution.min() * 1.1, self.U_solution.max() * 1.1) # constant y-range
            ax[i].grid()
            ax[i].tick_params(axis="both", which="major", labelsize=LABELSIZE-3)
        fig.suptitle("1D heat equation", size=LABELSIZE+3)
        fig.tight_layout()
        fig.savefig("figures/ex2_sol.png")
    
    def plot_sol_animation(self, LABELSIZE=15):
        """Plot solution using `matplotlib.animate.FuncAnimation`
        Plot solution for each value of t.
        
        Parameters
        ----
        *Optional and using keyword*  
        LABELSIZE : int, (default 15)
            The labelsize for refence to use. Title gets +2 and labels get -2
        """
        fig, ax = plt.subplots(figsize=(7,4))
        line, = ax.plot(self.x_full, self.U_solution[0, :])
        ax.set_title(f"1D Heat equation for $t = {self.t_range[0]:.2f}$", size=LABELSIZE+2)
        ax.set_xlabel(r"$x$", size=LABELSIZE-2)
        ax.set_ylabel(rf"$u(x,t)$", size=LABELSIZE-2)
        ax.set_ylim(self.U_solution.min() * 1.1, self.U_solution.max() * 1.1) # constant y-range
        ax.grid()

        def animate(frame):
            line.set_ydata(self.U_solution[frame, :])
            # ax.set_title(f"1D Heat equation for $t = {t_range[frame]:.2f}$")
            ax.set_title(f"1D Heat equation for $t = {self.t_range[frame]:.2f}$", size=LABELSIZE+2)
            # ax.set_ylabel(rf"$u(x,{t_range[frame]:.2f})$", size=LABELSIZE-2)
            print(f"{frame = }")
            return line,

        fig.tight_layout()
        self.ani = FuncAnimation(fig=fig, func=animate, frames=list(range(0, 100, self.N//100)) + list(range(100, self.N, self.N//20)), blit=False)
        plt.show()
    
    def plot_map(self, min_idx=0, max_idx=100, *, LABELSIZE=15):
        """Plot heatmap for solution to PDE
        Usually N (number of points in time) is very large, so we only show a snippet of the solution
        
        Parameters
        ----
        *Optional*  
        min_idx : int, (default : 0)
            the lower index for the solution array to include
        max_idx : int, (default : 100)
            the upper index for the soultion array to include
        
        *Optional and using keyword*  
        LABELSIZE : int, (default 15)
            The labelsize for refence to use. Title gets +2 and labels get -2
        
        """
        fig, ax = plt.subplots()
        cax = ax.imshow(self.U_solution[min_idx:max_idx,:], vmin=self.U_solution.min(), vmax=self.U_solution.max())
        x_idx = np.arange(len(self.x_full))[::25]
        t_idx = np.arange(len(self.t_range[min_idx:max_idx]))[::25]
        ax.set_xticks(x_idx)
        ax.set_xticklabels(self.x_full[x_idx])
        ax.set_xlabel(r"$x$", size=LABELSIZE-2)

        ax.set_yticks(t_idx)
        ax.set_yticklabels(np.round(self.t_range[t_idx], 3))
        ax.set_ylabel(r"$t$", size=LABELSIZE-2)

        ax.set_title("1D heat equation for $u(x,t)$", size=LABELSIZE+2)
        fig.colorbar(cax, ax=ax)
        fig.tight_layout()
        fig.savefig("figures/ex2_heatmap.png")
        return fig, ax
    
    def convergence_error(self, number_x_points):
        # Define master for comparison 
        # make new object with highest number of x-points
        _master = Parabolic(number_x_points[-1], self.ic_func, self.bc_funcs)
        U_master = _master.iterate_time()
        N_master, M_master = U_master.shape
        print(f"Benchmark U  : shape={U_master.shape} [Ntime, Nspace]")
        
        errors = []
        hs = []
        ks = []
        # loop over values for M (exclude last for comparison)
        for Mx in number_x_points[:-1]:
            # make new object with Mx spatial points 
            _iter = Parabolic(M=Mx, IC_func=self.ic_func, BC_funcs=self.bc_funcs)
            
            # compute solution
            U_iter = _iter.iterate_time()
            N, M = U_iter.shape
            
            # save the mash-grid to list
            hs.append(_iter.h)
            ks.append(_iter.k)
            
            # For t=T get values of x next to boundary and x=0
            idx = [1, M//2, -2]
            master_idx = [1, M_master//2, -2]
            U_points = U_iter[-1, idx] 
            master_points = U_master[-1, master_idx]
            
            # Compute 2-norm for the difference of 3 values and save to list
            err = np.linalg.norm(U_points - master_points, 2)
            errors.append(err)
        return np.asarray(ks), np.asarray(hs), np.asarray(errors)
    
    def plot_convergence(self, number_x_points, *, LABELSIZE=15):
        ks, hs, errors = self.convergence_error(number_x_points)
        
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        fig.suptitle("Error vs $k$", size=LABELSIZE+3)
        
        # plot Error vs h
        ax = [ax,]
        ax[0].loglog(hs, errors, "-o", label="FTCS")
        ax[0].loglog(hs, hs**2, "-k", label=r"$\mathcal{O}(h^2)$")
        ax[0].set_xlabel(r"$h$", size=LABELSIZE)
        ax[0].legend(fontsize=LABELSIZE-2)
        ax[0].tick_params(axis='both', which='major', labelsize=LABELSIZE-4)
        ax[0].grid()
        # # plot Error vs k
        # ax[1].loglog(ks, errors, "-o", label="FTCS")
        # ax[1].loglog(ks, ks, "-k", label=r"$\mathcal{O}(k)$")
        # ax[1].set_xlabel(r"$k$", size=LABELSIZE)
        # ax[1].legend(fontsize=LABELSIZE-2)
        # ax[1].grid()
        fig.tight_layout()
        fig.savefig("figures/ex2_convergence.png")
        

# %% ==== Initilize object ====
test = Parabolic(M=101, IC_func=eta, BC_funcs=(gL, gR), eps=EPS, use_safe_k=True)

# %% ==== Plot heat map
test.plot_map()

# %% ==== PLot convergence ====
test.plot_convergence([2**i*40 for i in range(6)])

# %% ==== Plot solution from some frames ====

test.plot_sol([0, 40, -100])

# %% ==== Plot solution as animation ====
test.plot_sol_animation()
