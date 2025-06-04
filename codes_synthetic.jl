#holds the function to return all synthetic datasets and what aspect of 
#dimension reduction do they preserve --- also refer to https://jmlr.org/papers/volume22/20-1061/20-1061.pdf
"""datasets = 

"""

using Downloads, ZipFile, NPZ, FileIO, ImageIO, Images, MatrixNetworks, Statistics

```Random walk without restart - shows trajectory maintenance```
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

```Galaxies are concentric circle - shows local shape maintainence/distortion```


```synthetic branching data PHATE - shows local and global shape maintainence```

function synthetic_tree(n_dim=100, n_branch=20, branch_length=100, rand_multiplier=2, seed=37, sigma=4)
    Random.seed!(seed)
    M = cumsum(-1 .+ rand_multiplier .* rand(branch_length, n_dim), dims = 1)
    for i in range(1,n_branch-1)
        ind = rand(1:branch_length)
        new_branch = cumsum(-1 .+ rand_multiplier .* rand(branch_length, n_dim), dims = 1)
        M = vcat(M,hcat([new_branch[b,:] .+ M[ind,:] for b in range(1,branch_length)]...)')
    end
    noise = rand(Normal(0,sigma),size(M))
    M = M .+ noise
    # returns the group labels for each point to make it easier to visualize
    # embeddings
    C =([div(i,branch_length)+1 for i in range(0,n_branch*branch_length-1)])
    return M, C
end



```synthetic developmental data```


```parallel lines -> 2D --- local topologgy maintainence```
function createline4mpoints(p1,p2,n)
    d = length(p1)
    sigma = 0.1
    # form n points from p1 to p2 with noise
    xs = map(t->p1 + t*(p2-p1) + sigma*randn(d),range(start=0,stop=1,length=n))
    return xs
end

function create_hashtag_sign(n, d, num_lines, distance)
    # Create a random direction vector for the first line
    @assert length(distance) == num_lines-1
    dir1 = randn(d);
    ini1 = randn(d);
    # Create two points separated by 'distance' along dir1
    p1 = ini1/norm(ini1) .+ (-10/2)*(dir1/norm(dir1));
    p2 = ini1/norm(ini1) .+ (10/2)*(dir1/norm(dir1));
    @show "ensuring the same length" norm(p1 .- p2)

    # Create a random direction vector for the second line orthogonal to the first
    dir2 = nullspace(dir1')[:,2];
    ini2 = randn(d);

    points = []
    labels = []
    push!(points,createline4mpoints(p1,p2,n))
    append!(labels,[1 for _ in range(1,n)])

    @info "distance is " distance

    for i in range(2,num_lines)
         # Create two points separated by 'distance' along dir2
         #p_start = ini2/norm(ini2) .+ (-distance/2)*(dir2/norm(dir2));
         #p_end = ini2/norm(ini2) .+ (distance/2)*(dir2/norm(dir2));
         # Offset the points along the first direction to create the parallel lines
        p_ini = p1 .+ (distance[i-1]*dir2/norm(dir2))
        p_fin = p2 .+ (distance[i-1]*dir2/norm(dir2))
        @show "length" norm(p_ini .- p_fin)
        push!(points,createline4mpoints(p_ini,p_fin,n))
        append!(labels,[i for _ in range(1,n)])
    end
    perm = randperm(n*k)
    X = Matrix(hcat(hcat(points...)...)');
    X = X[perm,:]
    labels = labels[perm]
    return X, labels
end




```parallel clusters -> 2D --- local shape maintainence```

```lines separated in high dimensions -> 2D --- local (high points per line) 
and global shape (too many lines) maintainence```

function generate_lines(k::Int, d::Int, nye::Vector{Int64}, separation::Vector{Float64},line_length::Vector{Float64})
    points = zeros(sum(nye), d)  # Placeholder for all points
    labels = []
    for i in 1:k
        # Random initial starting point for each line, spaced apart by 'separation'
        start_point = rand(d) .* separation[i] + (i - 1) .* separation[i] * ones(d)

        # Random direction for each line
        direction = randn(d)
        direction = direction / norm(direction)  # Normalize to get a unit vector

        end_point = start_point + direction * line_length[i]

        for j in 1:nye[i]
            t = (j - 1) / (nye[i] - 1)  # Parameter t runs from 0 to 1 for uniform spacing
            point = (1 - t) * start_point + t * end_point  # Linear interpolation
            points[sum(nye[1:i-1]) + j, :] = point
        end
  
        @show norm(end_point-start_point)
        @show sum([temp !=0 for temp in points]) #should be i*d
        append!(labels, [i for _ in range(1,nye[i])])
    end
    
    return points,labels
end

```3 gaussians at different distance -> 2D --- point topology```
function generate_multivariate_gaussians(k, d, nye, sep;var = 0)
    orthogonal_means = zeros(d, k)
    @assert length(sep) == k-1
    if length(sep)>1
        orthogonal_means[:,2:end] = orthogonal_means[:,2:end] .+ (ones(d,1)*sep')
    else
        orthogonal_means[:,2] = orthogonal_means[:,2] .+  sep #(sep*sqrt(2 * log(k)))
    end
    @info "separation" sep
    @show "variance" var
    gaussians = zeros(d,sum(nye))
    labels = zeros(sum(nye))
    for i in 1:k
        mean_vector = orthogonal_means[:, i]
        if length(var) >1 
            @assert length(var) == k
            covariance_matrix = var[i]*I(d)
        else
            covariance_matrix = I(d)
        end
        dist = MvNormal(mean_vector, covariance_matrix) 
        start_idx = sum(nye[1:i-1])+ 1
        end_idx = sum(nye[1:i])
        gaussians[:,start_idx:end_idx] = rand(dist, nye[i])
        labels[start_idx:end_idx] .= i
        @show sum([temp !=0 for temp in gaussians]) #should be i*d*n
    end
    p = randperm(size(gaussians,2))
    return gaussians[:,p], labels[p]
end

```2 Gaussians with different variance -> 2D ---- point topology```

```Mammoth 3D -> 2D --- ```