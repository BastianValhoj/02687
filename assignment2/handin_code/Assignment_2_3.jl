using Plots
using Printf

function solve_advection_ftbs(a, L, T, nx, nt, init_func)
    # spatial grid
    dx = 2 * L / (nx-1)
    x = range(-L, L, length=nx)
    
    # time grid
    dt = T / (nt-1)
    t = range(0, T, length=nt)
    
    # Calculate Courant number (c is some other constant so we call it cr)
    cr = a * dt / dx
    
    # Check stability condition
    if cr > 1.0
        @warn "Courant number $cr > 1.0, scheme may be unstable"
    end
    
    # Initialize solution matrix
    u = zeros(nx, nt)
    
    # Set initial condition
    for i in 1:nx
        u[i, 1] = init_func(x[i])
    end
    
    # Time stepping using FTBS
    for j in 1:nt-1
        for i in 2:nx
            # FTBS scheme: u_i^{n+1} = u_i^n - cr * (u_i^n - u_{i-1}^n)
            u[i, j+1] = u[i, j] - cr * (u[i, j] - u[i-1, j])
        end
        
        # Apply periodic boundary condition
        u[1, j+1] = u[nx-1, j+1]  # u[1] = u[nx-1] (not u[nx] to avoid duplication at boundary)
    end
    
    return x, t, u
end

# now, as we knwo the u_exact from the exercise manual, we can calculate the error

function calculate_error(u_numerical, u_exact)
    diff = u_numerical - u_exact
    l2_error = sqrt(sum(diff.^2) / length(diff)) # The L2 error gives a good overall measure of how well the numerical solution approximates the exact solution across the entire domain
    linf_error = maximum(abs.(diff)) #  The L-infinity error tells you the maximum deviation between your numerical solution and the exact solution
    return l2_error, linf_error
end

"""
    advection_exact(x, t, a, init_func)

Compute the exact solution for advection equation with periodic boundaries.
"""

# Now, we'll define how we compute the exact solution for the advection equation.


function advection_exact(x, t, a, init_func)
    # Calculate the position where the current x came from, using teh wave propagation
    # speed, given by 'a', along with a time step 't'
    x_origin = x - a*t
    
    # Apply periodic boundary condition to keep x_origin in [-1, 1]
    # First normalize to [0, 2] range
    x_origin = mod(x_origin + 1, 2) - 1
    
    # Apply initial condition function to this origin point
    return init_func(x_origin)
end


# Parameters
a = 0.5         # advection speed
L = 1.0         # domain from -1 to 1
T_periods = 2   # number of periods to simulate

# Initial condition (as mentioned in the exercise)
init_func(x) = sin(2 * pi * x)

# Determine simulation time based on periods
wavelength = 2.0  # domain length
period = wavelength / a
T = T_periods * period

# Test with various grid resolutions
resolutions = [20,50, 100, 200, 400, 1000]
errors_l2 = Float64[]
errors_linf = Float64[]
dx_values = Float64[]

# Create a plot for the initial condition and exact solution 
# (will add all numerical solutions to this plot)
# Use a high-resolution grid for the exact solution
x_high_res = range(-L, L, length=1000)
u_exact_high_res = [advection_exact(x, T, a, init_func) for x in x_high_res]
u_init_high_res = [init_func(x) for x in x_high_res]

# Create the multi-resolution solution plot
p_multi = plot(x_high_res, u_init_high_res, label="Initial", 
                xlabel="x", ylabel="u", legend=:best)
plot!(p_multi, x_high_res, u_exact_high_res, label="Exact (t=$T)", 
      linestyle=:dash, linewidth=2, color=:black,dpi=300)

# Define a color palette for different resolutions
colors = [:blue, :red, :green, :purple, :orange, :cyan]

for (i, nx) in enumerate(resolutions)
    # Set number of time steps to maintain constant Courant number
    cr_target = 0.8  # as specified in the exercise
    dx = 2 * L / (nx-1) # spatial step size
    dt = cr_target * dx / a 
    nt = Int(ceil(T / dt)) + 1
    dt = T / (nt-1)  # Recalculate to ensure we end exactly at time T
    
    # Solve the advection equation
    x, t, u = solve_advection_ftbs(a, L, T, nx, nt, init_func)
    
    # Calculate exact solution at final time
    u_exact = zeros(nx)
    for j in 1:nx
        u_exact[j] = advection_exact(x[j], T, a, init_func)
    end
    
    # Calculate errors
    l2_err, linf_err = calculate_error(u[:, end], u_exact)
    push!(errors_l2, l2_err)
    push!(errors_linf, linf_err)
    push!(dx_values, dx)
    
    println("Resolution: $nx, dx: $dx, L2 error: $l2_err, Linf error: $linf_err")
    
    # Add this resolution to the multi-resolution plot
    plot!(p_multi, x, u[:, end], label="nx=$nx", 
          linewidth=1.5, alpha=0.7, color=colors[i])
end

# Save the multi-resolution plot
savefig(p_multi, "advection_multi_resolution.png")
display(p_multi)

# Calculate convergence rates
convergence_l2 = log.(errors_l2[1:end-1] ./ errors_l2[2:end]) ./ log.(dx_values[1:end-1] ./ dx_values[2:end])
convergence_linf = log.(errors_linf[1:end-1] ./ errors_linf[2:end]) ./ log.(dx_values[1:end-1] ./ dx_values[2:end])

println("\nConvergence rates (L2): ", convergence_l2)
println("Convergence rates (Linf): ", convergence_linf)

# Plot convergence
p2 = plot(dx_values, errors_l2, xaxis=:log, yaxis=:log, 
            xlabel="dx", ylabel="Error", label="L2 Error",  marker=:circle)
plot!(p2, dx_values, errors_linf, label="Linf Error", marker=:square)

# Add reference line with slope 1 (first-order convergence)
ref_line = 0.5 * dx_values
plot!(p2, dx_values, ref_line, label="First Order", linestyle=:dash, dpi=300)

display(p2)
savefig(p2, "advection_convergence.png")


#%% phase shift analysis



function ftbs_amplification(cr, kdx)
    xi = 1 - cr * (1 - exp(-im * kdx))
    return abs(xi), angle(xi)
end

steps_no = 100
kdx = 2 * pi / steps_no
xiabs, xiphase = ftbs_amplification(0.8, kdx)
nt = 40 * steps_no # 40 wave periods at 100 steps per period
amplitude_decay = xiabs^nt
phase_shift = nt * xiphase  # in radians
@printf "predicted amplitude ratio after 40 periods: %.5f\n" amplitude_decay
@printf "predicted phase shift after 40 periods: %.5f radians (%.3f wavelengths)\n" phase_shift (phase_shift / (2 * pi))

# visualize it???
