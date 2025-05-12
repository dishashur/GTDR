include("GraphTDA.jl")
include("utils.jl")



```Random walk without restart```
function rw_norestart(n, dim)
    points = []
    current = zeros(dim)  # Start at the origin
    for i in 1:n
        step = randn(dim)  # Random step in `dim` dimensions
        next = current .+ step  # Move to the next position
        push!(points, next)  # Store the new point with its color
        current = next  # Update position
    end
    return points
end
