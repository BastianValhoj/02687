using Plots

# At first, we define the flame propagation equation from the problem
# y^2 - y^3
# y0 = delta
# 0 <_ t <_ 2/delta
#where y(t) represents the radius of the ball and δ is a small value. Try δ= 0:02 and
# se if you can reduce it to 0:0001 within a relative tolerance of 10􀀀4 (you may decrease
# this, but this requires more work). Results can be compared to a ODE solver provided in
# Python/Matlab. Can you apply the Picard-Lindelo  theorem to understand if a unique
# solution exists?

# at first we define the flame propagation function
function flame_propagation(t, y)
    return y^2 - y^3
end

# Butcher tableau for RK23 method (Bogacki-Shampine), defined from#
# https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method

# okay afterwards we do not this this. 

function rk23_step(f, tn, yn, h)
    # First stage
    eps1 = yn
    
    # Coefficients from the equation
    a21 = 1/2
    a31 = 0
    a32 = 3/4
    
    c2 = a21  # c2 = 1/2
    c3 = a31 + a32  # c3 = 3/4
    
    # Butcher tableau weights for the 3rd order solution
    b1 = 2/9
    b2 = 1/3
    b3 = 4/9
    
    # Butcher tableau weights for the 2nd order solution (for error estimation)
    d1 = 7/24
    d2 = 1/4
    d3 = 1/3
    
    # Second stage
    eps2 = yn + a21 * h * f(tn, eps1)
    
    # Third stage
    eps3 = yn + a31 * h * f(tn, eps1) + a32 * h * f(tn + c2*h, eps2)
    
    # Calculate next value (3rd order)
    yn_plus1 = yn + h * (b1 * f(tn, eps1) + b2 * f(tn + c2*h, eps2) + b3 * f(tn + c3*h, eps3))
    
    # Calculate error estimate
    en_plus1 = h * (d1 * f(tn, eps1) + d2 * f(tn + c2*h, eps2) + d3 * f(tn + c3*h, eps3))
    
    # Calculate second-order solution for error estimation
    yn_hat = yn + en_plus1
    
    # Error calculation
    error = abs(yn_plus1 - yn_hat)
    
    return yn_plus1, yn_hat, error
end

# Adaptive RK23 solver
function solve_adaptive_rk23(f, t0, y0, tend, reps, aeps; h_init=0.01, h_min=1e-10, h_max=0.1)
    t = t0 # init is 0
    y = y0 # init is delta
    h = h_init # init is 0.01
    
    # Store results
    ts = [t] # 1 element vecotrs
    ys = [y] # 1 element vectors
    hs = Float64[]  # Step sizes used, empty init
    
    while t < tend
        # This code is only for the last step, wo we dont overshoot the last time step. 
        if t + h > tend
            h = tend - t
        end
        # now we make the step with our fucntion above, computing 
        # y_new which is the new 3rd order solution, and 
        # y_hat which is the 2nd order solution, only used for error estimation
        y_new, y_hat, error = rk23_step(f, t, y, h)

        # now we have defined 
        # y_new which 
        # Compute tolerance based on combined absolute and relative error form the manual, and if the error is less than then tolerance, 
        # we accept and pushes it on the end of the vectors we defined above
        tol = reps * abs(y) + aeps
        if error <= tol # accept the error, as it is too small, and 
                        # proceede to add h to the time, and updazte the solution
            t = t + h
            y = y_new
            push!(ts, t)
            push!(ys, y)
            push!(hs, h)
            
            # Adjust step size for next step (with safety factor 0.9)
            # https://en.wikipedia.org/wiki/Adaptive_step_size
            safety_factor = 0.9
            h_new = safety_factor * h * min(max((tol/error)^(1/2), 0.2), 5.0)

            h = min(max(h_new, h_min), h_max)
        else # decline the error, and try again with a smaller stepsize
            
            h = max(0.5 * h * (tol/error)^(1/3), h_min)
        end
    end
    # afther this while loop, we have reached the end time, so we return the results as 

    return ts, ys, hs 
end

# now we define a function to save the results, given the reps, aeps, and delta, in order to test selveral valiues
function solve_flame_propagation(delta, reps, aeps)
    t0 = 0.0
    y0 = delta
    tend = 2.0 / delta
    ts, ys, hs = solve_adaptive_rk23(flame_propagation, t0, y0, tend, reps, aeps)
    return ts, ys, hs
end

# we test the solver with delta = 0.02
delta = 0.01
reps = 1e-4  # Relative tolerance
aeps = 1e-6  # Absolute tolerance

ts, ys, hs = solve_flame_propagation(delta, reps, aeps)
    # ts: time points
    # ys: solution
    # hs: step size, also being logged

println("Flame propagation results for delta = $delta:")
println("Total steps: $(length(ts)-1)")
println("Final time: $(ts[end])")
println("Final radius: $(ys[end])")


delta_small = 0.00001
ts_small, ys_small, hs_small = solve_flame_propagation(delta_small, reps, aeps)

println("\nFlame propagation results for delta = $delta_small:")
println("Total steps: $(length(ts_small)-1)")
println("Final time: $(ts_small[end])")
println("Final radius: $(ys_small[end])")

#plotting
# Create the initial plot with delta value in title
p1 = plot(ts, ys,
    title = "Time versus radius of ball (delta = $delta)",
    xlabel = "time [s]",
    ylabel = "radius [m]"
)

# Display the plot
display(p1)
#%%
p2 = plot(ts_small, ys_small,
    title = "Time versus radius of ball (delta = $delta_small)",
    xlabel = "time [s]",
    ylabel = "radius [m]"
)

# Display the plot
display(p2)


#%% we solve for 10 different deltas, and plot them all in the same plot
deltas = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
results = []
for delta in deltas
    ts, ys, hs = solve_flame_propagation(delta, reps, aeps)
    push!(results, (ts, ys))
end
# Plot all results
p3 = plot(title = "Time versus radius of ball for different deltas",
    xlabel = "time [s]",
    ylabel = "radius [m]"
)
# Plot each result with a different color
for (i, (ts, ys)) in enumerate(results)
    plot!(p3, ts, ys, label = "delta = $(deltas[i])")
end

plot!(p3, 
        legend = :topright,
        xlims=(0, 1000))
display(p3)