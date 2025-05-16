

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
function galaxies()
    Random.seed!(42)
    n = [500, 500, 2000, 500, 1000] #1000 #different number of stars in each galxy
    X = randn(sum(n),3) / sum(n)
    labels = []
    rad = [3, 10 , 3, 15]
    X[1:n[1],1] += rad[1]*cos.([i for i in range(1,n[1])]*2*pi/n[1])
    X[1:n[1],2] += rad[1]*sin.([i for i in range(1,n[1])]*2*pi/n[1])
    append!(labels,[1 for _ in range(1,n[1])])
    X[n[1]+1:sum(n[1:2]),1] += 4 .+ cos.([i for i in range(1,n[2])]*2*pi/n[2])
    X[n[1]+1:sum(n[1:2]),2] += 4 .+ sin.([i for i in range(1,n[2])]*2*pi/n[2])
    append!(labels,[2 for _ in range(n[1]+1,sum(n[1:2]))]);
    X[sum(n[1:2])+1:sum(n[1:3]),1] += rad[2]*cos.([i for i in range(1,n[3])]*2*pi/n[3])
    X[sum(n[1:2])+1:sum(n[1:3]),2] += rad[2]*sin.([i for i in range(1,n[3])]*2*pi/n[3])
    append!(labels,[3 for _ in range(sum(n[1:2])+1,sum(n[1:3]))]);
    X[sum(n[1:3])+1:sum(n[1:4]),1]  +=  15 .+ rad[3]*cos.([i for i in range(1,n[4])]*2*pi/n[4])
    X[sum(n[1:3])+1:sum(n[1:4]),2] +=  15 .+ rad[3]*sin.([i for i in range(1,n[4])]*2*pi/n[4])
    append!(labels,[4 for _ in range(sum(n[1:3])+1,sum(n[1:4]))]);
    X[sum(n[1:4])+1:sum(n[1:5]),1]  += rad[4]*cos.([i for i in range(1,n[5])]*2*pi/n[5])
    X[sum(n[1:4])+1:sum(n[1:5]),2]  += rad[4]*sin.([i for i in range(1,n[5])]*2*pi/n[5])
    append!(labels,[5 for _ in range(sum(n[1:4])+1,sum(n[1:5]))])
    perm = randperm(sum(n))
    X = X[perm,:]
    labels = labels[perm]

    return X, labels
end

```synthetic branching data PHATE - shows local and global shape maintainence```


```parallel lines -> 2D --- local topologgy maintainence```

```parallel clusters -> 2D --- local shape maintainence```

```lines separated in high dimensions -> 2D --- local (high points per line) 
and global shape (too many lines) maintainence```

```3 gaussians at different distance -> 2D --- point topology```

```2 Gaussians with different variance -> 2D ---- point topology```

```Mammoth 3D -> 2D --- ```