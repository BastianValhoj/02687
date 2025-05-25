using Plots

# Define the function: Re[(1 - c + c cos(θ) - i c sin(θ))^2]
f(θ, c) = real((1 - c + c * cos(θ) - im * c * sin(θ))^2)

# Create θ and c ranges
θ_vals = range(0, 2π, length=3000)
c_vals = range(0, 2, length=3000)

# Create a matrix of real values
z = [f(θ, c) for c in c_vals, θ in θ_vals]  # c on rows, θ on columns

# Create a binary matrix where 1 = values > 1, 0 = values <= 1
mask = zeros(size(z))
mask[z .> 1] .= 1

# Plot just the mask with red for 1 (values > 1) and transparent for 0
plt = heatmap(
    θ_vals, c_vals, mask,
    xlabel="θ",
    ylabel="c",
    title="Areas where Re[ξ(θ, c)^2] > 1",
    color = cgrad([:transparent, :red], [0, 1]),  # Transparent to red
    colorbar = false,  # No colorbar needed for binary data
    dpi = 300
)

display(plt)  # Display the plot
savefig("heatmap_areas_above_1.png")  # Save the figure as a PNG file