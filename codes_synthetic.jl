include("/Users/alice/Documents/Research/RPs/mainidea.jl")

```tree data from PHATE```
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

n_dim = 200
n_branch = 10
branch_length = 300
rand_multiplier = 2
seed=37
sigma = 5
X,labels = synthetic_tree(n_dim, n_branch, branch_length, rand_multiplier, seed, sigma);
#phate_tree = npzread("../draft1/treedata.npz")
#X = phate_tree["data"];
#labels = phate_tree["labels"];
nodecolors = distinguishable_colors(n_branch, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,25,dims = 30);
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=50,min_component_group=1,verbose=false,overlap = 0.2,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node  
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.05,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
#nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);
#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw =0.1)
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")

gactual = SimpleGraph(G)
pos = spring_layout(gactual)
xyactual = hcat(pos...)
pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = labels,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")


super_plot = plot(t,p,layout = (1,2),size = (1200, 500),markerstrokewidth=0.5,titlefontsize = 10)

savefig(super_plot,"../draft1/phate_branches_$(n_branch)_$(branch_length)")

nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("phate_$(n_branch)_$(branch_length).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

``` hashtag sign ---back2basics ```

#for the hashtag sign
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

    @show distance

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

Random.seed!(123)
n = 500 #3000, 2000, 1000
d = 100 
k = 20 #3,5,10
sep = 10 # 5, 10
X,labels = create_hashtag_sign(n, d, k, [sep*i for i in range(1,k-1)]);

X = X .- mean(X);
X = X ./ std(X);

customcolors = cgrad(:ice, length(unique(labels)), categorical = true)

orig= scatter(X[:,1],X[:,2],markerstrokewidth=0.5,markersize=3.0,framestyle=:none,
axis_buffer=0.02,legend = :outertopleft, color = customcolors[labels], group = labels,legendfontsize = 10.0,
title = "$(n)-$(d)-$(sep)")

begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=200,min_component_group=1,verbose=false,overlap = 0.01,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node  
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/5e5),p,nodecolors,showlabels)
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = customcolors[labels], group = labels,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

savefig(super_plot,"../draft1/hashtag_$(n)_$(d)_$(k)_$(sep)")


nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("hashtag_$(n)_$(d)_$(k)_$(sep).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end


######################################################################################################################################################

```distill - two elongated gaussians```
function longGaussianData(n, dim)
    X = randn(n,dim)
    for i in range(1,dim)
        X[:,i] .= X[:,i] ./ (1+i)
    end
    return X
end

Random.seed!(123)
n = 3000
d = 100
X = longGaussianData(n,d)
X = X .- mean(X);
X = X ./ std(X);
labels = [norm(X[i,:]) for i in range(1,n)];
nodecolors = cgrad(:ice, Int(round(maximum(labels)) + 1), categorical = true)
finalcolors = [nodecolors[Int(round(i))] for i in labels] 

orig = scatter(X[:,1],X[:,2],color = finalcolors,markerstrokewidth=0.5,markersize=3.0,
framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)",label = "",group = labels)


begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining


gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=100,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node
preeb = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,
    framestyle=:none, linecolor=:black,linealpha=0.1,
    axis_buffer=0.02,labels = "");
reebcolors = [mean(labels[i]) for i in rc]
finalreebcolors = [nodecolors[Int(round(i))] for i in reebcolors] 
preeb = scatter!(preeb,xy[:,1],xy[:,2],color = finalreebcolors,title = "gtda_$(round(timereeb+tym1,sigdigits=4))",
label="",markerstrokewidth=0.1)


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.0,markersize=3.0,color = finalcolors,
       framestyle=:none,axis_buffer=0.02, title = "tsne$(round(timetsne,sigdigits=4))",label = "")

super_plot = plot(orig,t,preeb,layout = (1,3),size = (1600, 500),markerstrokewidth=0.0,titlefontsize = 10)

savefig(super_plot,"../draft1/Elongated$(n)_$(d).png")


nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("Elongated$(n)_$(d).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################

```Random walk from distill```
function randomWalk(n, dim)
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
Random.seed!(123)
points = randomWalk(3000,100);
X = Matrix(hcat(points...)');
X = X .- mean(X);
X = X ./ std(X);
labels = [norm(X[i,:]) for i in range(1,n)];
nodecolors = cgrad(:ice, Int(round(maximum(labels)) + 1), categorical = true)
finalcolors = [nodecolors[Int(round(i))] for i in labels] 

orig = scatter(X[:,1],X[:,2],color = finalcolors,markerstrokewidth=0.5,markersize=3.0,
framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)",label = "",group = labels)

begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining


gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=500,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node
preeb = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.6,
    framestyle=:none, linecolor=:black,linealpha=0.6,
    axis_buffer=0.02,labels = "");
reebcolors = [mean(labels[i]) for i in rc]
finalreebcolors = [nodecolors[Int(round(i))] for i in reebcolors] 
preeb = scatter!(preeb,xy[:,1],xy[:,2],color = finalreebcolors,label="",markerstrokewidth=0.1)
#title = "gtda_$(round(timereeb+tym1,sigdigits=4))",



embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.0,markersize=3.0,color = finalcolors,
       framestyle=:none,axis_buffer=0.02, title = "tsne$(round(timetsne,sigdigits=4))",label = "")

super_plot = plot(orig,t,preeb,layout = (1,3),size = (1600, 500),markerstrokewidth=0.0,titlefontsize = 10)

savefig(super_plot,"../draft1/DistillRW$(n)_$(d).png")

nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("DistillRW$(n)_$(d).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################


```Random walk```

function random_walk(N,d;p = 0.1)
    x0 = zeros(d)
    xi = copy(x0) 
    sigma = 0.1
    X = zeros(N, d) 
    for i=1:N
    xi = xi .+ sigma*randn(d) 
    if rand() < p
        xi = copy(x0) .+ sigma*randn(d)
    else 
        xi = xi .+ sigma*randn(d)   
    end
    X[i,:] = xi
    end 
    return X
end

Random.seed!(123)
d = 100
n = 1000
p = 0.7
X = random_walk(n,d, p=p);
X = X .- mean(X);
X = X ./ std(X);
colors = [norm(X[i,:]) for i in range(1,n)];
labels = colors;

orig = scatter(X[:,1],X[:,2],zcolor = colors,markerstrokewidth=0.5,markersize=3.0,
framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)_$(p)",label = "")

#= gtda parameters
d = 10, n = 100, prob = 0.1 - (10,50,0.1)
d = 100, n = 1000, prob = 0.1 - (100,500,0.1)
d = 100, n = 1000, prob = 0.7 - (100,500,0.1)
d = 10, n = 1000, prob = 0.1 - (100,500,0.1)
d = 10, n = 1000, prob = 0.7 - (100,500,0.1)
d = 100, n = 10000, prob = 0.1 - (1000, 7000, 0.05),(10,1000,0.2)
d = 100, n = 10000, prob = 0.7 - (1000,5000,0.1)
=#
begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining


gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
        max_split_size=700,min_component_group=1,verbose=false,overlap = 0.2,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node
preeb = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,
    framestyle=:none, linecolor=:black,linealpha=0.3,
    axis_buffer=0.02,labels = "");
nodecolors = [mean(colors[i]) for i in rc]
preeb = scatter!(preeb,xy[:,1],xy[:,2],zcolor = nodecolors,title = "gtda_$(round(timereeb+tym1,sigdigits=4))",
label="",markerstrokewidth=0.5)


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.0,markersize=3.0,zcolor= labels,
       framestyle=:none,axis_buffer=0.02, title = "tsne$(round(timetsne,sigdigits=4))",label = "")

super_plot = plot(orig,t,preeb,layout = (1,3),size = (1600, 500),markerstrokewidth=0.0,titlefontsize = 10)


savefig(super_plot,"../draft1/rw$(n)_$(d)_$(p).png")


nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("rw_$(n)_$(d)_$(p).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################


```
Gaussian
n[1000x3],sep[10,50]-3,600,0.5
```
function generate_multivariate_gaussians(k, d, nye, sep;var = 0)
    orthogonal_means = zeros(d, k)
    @assert length(sep) == k-1
    if length(sep)>1
        orthogonal_means[:,2:end] = orthogonal_means[:,2:end] .+ (ones(d,1)*sep')
    else
        orthogonal_means[:,2] = orthogonal_means[:,2] .+  sep #(sep*sqrt(2 * log(k)))
    end
    @show sep
    @show var
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
Random.seed!(123)

k = 3
d = 100
n = [1000, 1000, 1000]
sep = [10, 50]#[10*i for i in range(1,k-1)]
varnc = [1 for _ in range(1,k)]#[1, 1, 10]#
points, labels = generate_multivariate_gaussians(k,d,n,sep, var = varnc);
X = Matrix(points');
X = X .- mean(X);
X = X ./ std(X);


nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);

orig= scatter(X[:,1],X[:,2],markerstrokewidth=0.5,markersize=3.0,framestyle=:none,
axis_buffer=0.02,legend = :outertopleft, color = nodecolors[Int.(labels)], group = labels,
legendfontsize = 10.0,title = "$(n[1])-$(d)-$(sep)-$(varnc[1])")


begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=400,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0.0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node  
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.5,framestyle=:none, linecolor=:black,linealpha=1.0,axis_buffer=0.02,
labels = "", legend=:outertopleft);
#plot(framestyle=:none,axis_buffer=0.02,labels = "", legend=:outertopleft,linewidth=0.5,linealpha=0.05);
nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/5e5),p,nodecolors,showlabels,lw = 0.5)
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,
group= labels, color = nodecolors[Int.(labels)],framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

savefig(super_plot,"../draft1/gauss$(n[1])_$(d)_$(sep).png")


nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("gauss$(n[1])_$(d)_$(sep).json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################

```
lines
```

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
Random.seed!(12)
k, d, n, separation, line_length = 3, 100, [1000 for _ in range(1,3)], [5.0 for _ in range(1,3)], [10.0 for _ in range(1,3)]
X,labels = generate_lines(k, d, n, separation, line_length);
X = X .- mean(X);
X = X ./ std(X);
perm = randperm(size(X,1));
X = X[perm,:];
labels = labels[perm];

nodecolors= cgrad(:ice, length(unique(labels)), categorical = true);

orig= scatter(X[:,1],X[:,2],markerstrokewidth=0.5,markersize=3.0,framestyle=:none,
axis_buffer=0.02,legend = :outertopleft, color = nodecolors[labels], group = labels,legendfontsize = 10.0,
title = "$(n[1])-$(d)")

begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=200,min_component_group=1,verbose=false,overlap = 0.15,
        split_thd=0.001,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node  
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels)
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = labels,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

savefig(super_plot,"../draft1/lines3000_$(d).png")

nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("lines_3000_$(d)_5_10.json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################



``` TF IDF data```


#################################

```Spheres Dataset from TopoAE```
using Random, Distributions, LinearAlgebra

function dsphere(n::Int, d::Int, r::Float64)
    x = randn(n, d)  # Sample from normal distribution
    x ./= sqrt.(sum(x.^2, dims=2))  # Normalize to lie on the unit sphere
    return r * x  # Scale by radius r
end

function create_sphere_dataset(n_samples=200, d=100, n_spheres=6, r=5.0, plot=false, seed=42)
    # Rescaling variance by sqrt(d) to maintain structure
    variance = 10 / sqrt(d)

    # Shift matrix: random normal shifts for each sphere
    shift_matrix = randn(n_spheres, d) * variance

    spheres = []
    n_datapoints = 0

    for i in 1:(n_spheres-1)
        sphere = dsphere(n_samples, d, r)
        [sphere[r,:] = sphere[r,:] .+ shift_matrix[i,:] for r in range(1,size(sphere,1))]
        @show size(sphere)
        @show size(shift_matrix)
        push!(spheres, sphere)
        n_datapoints += n_samples
    end
    @show n_datapoints
    # Additional big surrounding sphere
    n_samples_big = 10 * n_samples
    big = dsphere(n_samples_big, d, r * 20)
    push!(spheres, big)
    n_datapoints += n_samples_big

    # Combine spheres into a single dataset
    dataset = vcat(spheres...)

    # Assign labels
    labels = zeros(Int, n_datapoints)
    label_index = 1
    for (index, data) in enumerate(spheres)
        n_sphere_samples = size(data, 1)
        labels[label_index:(label_index + n_sphere_samples - 1)] .= index
        label_index += n_sphere_samples
    end

    return dataset, labels
end

Random.seed!(123)
X,labels = create_sphere_dataset();
nodecolors= cgrad(:ice, length(unique(labels)), categorical = true);

begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=40,min_component_group=1,verbose=false,overlap = 0.2,
        split_thd=0.0,merge_thd = 0.01,labels = labels);
gtdaobj.G_reeb

@time begin
g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node;  
p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
#DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,framestyle=:none, linecolor=:black,linealpha=0.1,axis_buffer=0.02,labels = "");
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5)
end
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,
color = nodecolors[labels], group = labels,framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

super_plot = plot(t,p,layout = (1,2),size = (1200, 500),markerstrokewidth=0.5,titlefontsize = 10)


nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

open("CirclesTopoAE2000_100_20.json", "w") do io
    JSON.print(io, Dict("true_negative"=>true_negative,"true_positive"=>true_positive))
end

######################################################################################################################################################
```
Figure 5 from the neighboorhood embedding paper --- synthetic develomental data (the kind of data where tsne fails)

Gotta compare against umap and FA these are better at showing clusters in multiple resolution

```
#developmental data developed from python
#temp = JSON.parsefile("/Users/alice/Documents/Research/RPs/shifted_gaussians/developmentaldata.json")
#X = vcat(temp["gaussiandata"]'...)
#origlabels = vcat([[i for _ in range(1,1000)] for i in range(1,20)]...)
#pythontree = sparse(temp["nn_i"] .+ 1, temp["nn_j"] .+ 1, ones(length(temp["nn_i"])))
#lemap = vcat(temp["ledata"]'...)
#lemap = lemap[perm,:]
#le = scatter(lemap[:,1],lemap[:,2],group = labels,framestyle=:none,axis_buffer=0.02,legend =:outertopleft,title = "LEmap")


temp = npzread("/Users/alice/Documents/Research/RPs/shifted_gaussians/developmental_data_20.npz")
X = temp["data"]
X = X .- mean(X);
X = X ./ var(X);
origlabels = temp["label"] .+ 1;

labels = zeros(size(origlabels))
[labels[i] = 2 for i in range(1,length(labels)) if (origlabels[i] == 20.0)]
[labels[i] = 3 for i in range(1,length(labels)) if (origlabels[i] == 10.0)]
[labels[i] = 4 for i in range(1,length(labels)) if (origlabels[i] == 1.0)]
[labels[i] = 1 for i in range(1,length(labels)) if (origlabels[i] != 1.0 && origlabels[i] != 10.0 && origlabels[i] != 20.0)]
nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);
perm = randperm(length(origlabels))
X = X[perm,:]
labels = labels[perm];
origlabels = Int.(origlabels[perm])

orig = scatter(X[:,1],X[:,2],group = labels,framestyle=:none,axis_buffer=0.02,legend =:outertopleft, 
color = nodecolors[Int.(labels)], title = "Fig5_NE")


begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym = time() - begining
    
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=40,min_component_group=5,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);
        
g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node

p = plot(framestyle=:none,axis_buffer=0.02,labels = "");
#DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,
#    framestyle=:none, linecolor=:black,linealpha=0.6,
#    axis_buffer=0.02,labels = "");
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,nodecolors,showlabels,lw = 0.6)
end
title!(p,"GTDA_$(round((timereeb+tym),sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,group= labels,
       framestyle=:none,color = nodecolors[Int.(labels)],axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits = 4))")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

savefig(super_plot,"../draft1/NE_20.png")


######################################################################################################################################################


```
circle
```
Random.seed!(12)

n = 1000
X = randn(n,3) / n
X[:,1] += 50*cos.([i for i in range(1,n)]*2*pi/n)
X[:,2] += 50*sin.([i for i in range(1,n)]*2*pi/n)
X = X .- mean(X)
X = X ./ std(X)
colors = [i/n for i in range(1,n)];
orig = scatter(X[:,1],X[:,2],zcolor = colors,markerstrokewidth=0.5,markersize=3.0,
framestyle=:none,axis_buffer=0.02,legend = false,title = "$(n)_50")

labels = colors;

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=1.0,label="",zcolor = colors,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

begining = time()
Xnoisy,G,lens = topological_lens(X,3)
tym1 = time() - begining
 
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=50,
        max_split_size=200,min_component_group=1,verbose=false,overlap = 0.15,
        split_thd=0,merge_thd = 0.01,labels = labels);
  
g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,
    framestyle=:none, linecolor=:black,linealpha=0.6,
    axis_buffer=0.02,labels = "");
nodecolors = [sum(colors[i])/length(i) for i in rc]
p = scatter!(p,xy[:,1],xy[:,2],zcolor = nodecolors,title = "gtda_$(round(timereeb+tym1,sigdigits=4))",label="")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

super_plot[:plot_title] = "tsne$(round(tsne_error,sigdigits = 4))gtda$(round(gtda_error,sigdigits = 4))"
plot!(super_plot)


######################################################################################################################################################

```
concentric circles
```
Random.seed!(42)

n = [100, 200]#[1000, 1000, 1000, 1000, 1000]
rad = [5,6] #[3,10,40,70,80]
X = randn(sum(n),3) / sum(n)
labels = []
X[1:n[1],1] += rad[1]*cos.([i for i in range(1,n[1])]*2*pi/n[1]);
X[1:n[1],2] += rad[1]*sin.([i for i in range(1,n[1])]*2*pi/n[1]);
append!(labels,[1 for _ in range(1,n[1])]);
X[n[1]+1:sum(n[1:2]),1] +=  rad[2]*cos.([i for i in range(1,n[2])]*2*pi/n[2]);
X[n[1]+1:sum(n[1:2]),2] +=  rad[2]*sin.([i for i in range(1,n[2])]*2*pi/n[2]);
append!(labels,[2 for _ in range(n[1]+1,sum(n[1:2]))]);
X[sum(n[1:2])+1:sum(n[1:3]),1] +=  rad[3]*cos.([i for i in range(1,n[3])]*2*pi/n[3]);
X[sum(n[1:2])+1:sum(n[1:3]),2] +=  rad[3]*sin.([i for i in range(1,n[3])]*2*pi/n[3]);
append!(labels,[3 for _ in range(sum(n[1:2])+1,sum(n[1:3]))]);
X[sum(n[1:3])+1:sum(n[1:4]),1] +=  rad[4]*cos.([i for i in range(1,n[4])]*2*pi/n[4]);
X[sum(n[1:3])+1:sum(n[1:4]),2] +=  rad[4]*sin.([i for i in range(1,n[4])]*2*pi/n[4]);
append!(labels,[4 for _ in range(sum(n[1:3])+1,sum(n[1:4]))]);
X[sum(n[1:4])+1:sum(n[1:5]),1] +=  rad[5]*cos.([i for i in range(1,n[5])]*2*pi/n[5]);
X[sum(n[1:4])+1:sum(n[1:5]),2] +=  rad[5]*sin.([i for i in range(1,n[5])]*2*pi/n[5]);
append!(labels,[5 for _ in range(sum(n[1:4])+1,sum(n[1:5]))]);
X = X .- mean(X)
X = X ./ std(X)
perm = randperm(sum(n))
X = X[perm,:];
labels = labels[perm];
customcolors = cgrad(:ice, length(unique(labels)), categorical = true);

orig = scatter(X[:,1],X[:,2],color = customcolors[labels],
markerstrokewidth=0.5,markersize=3.0,framestyle=:none,axis_buffer=0.02, group = labels, title = "$(n)-$(rad)",legend = false)


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=false,color = customcolors[labels],
group= labels, framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits = 4))")

begining = time()
Xnoisy,G,lens = topological_lens(X,3)
tym = time() - begining
        
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=50,min_component_group=1,verbose=false,overlap = 0.1,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node

p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,
    framestyle=:none, linecolor=:black,linealpha=0.6,
    axis_buffer=0.02,labels = "");
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,customcolors,showlabels)
end
title!(p,"GTDA_$(round((timereeb+tym),sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)",legend = :outertopleft)


super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)

errortime = time()
tsne_graph = make_graph(embedding)
tsne_dist = get_graph_distance(tsne_graph)
gtda_dist = get_graph_distance(gtdaobj.G_reeb)
n = size(embedding,1)
rc = gtdaobj.reeb2node
gtda_projected_dist = Inf*ones(n,n)
for r in range(1,length(rc))
    for j in range(1, length(rc))
        gtda_projected_dist[rc[r],rc[r]] .= 0
        gtda_projected_dist[rc[r],rc[j]] .= min.(gtda_dist[r,j],gtda_projected_dist[rc[r],rc[j]])
    end
end


orig_graph = make_graph(X)
orig_graph_distance = get_graph_distance(orig_graph)


tsne_dist .= ifelse.(tsne_dist .== Inf, 10^10, tsne_dist)
gtda_projected_dist .= ifelse.(gtda_projected_dist .== Inf, 10^10, gtda_projected_dist)
orig_graph_distance .= ifelse.(orig_graph_distance .== Inf, 10^10, orig_graph_distance)


tsne_pairs = topo_error.calculate_error(tsne_dist);
gtda_pairs = topo_error.calculate_error(gtda_projected_dist);
orig_pairs = topo_error.calculate_error(orig_graph_distance);

tsne_pairs = tsne_pairs .+ 1
gtda_pairs = gtda_pairs .+ 1
orig_pairs = orig_pairs .+ 1


@show tsne_verity = norm(
    [(orig_graph_distance[orig_pairs[i,1],orig_pairs[i,2]] - tsne_dist[orig_pairs[i,1],orig_pairs[i,2]]) for i in range(1,size(orig_pairs,1))]
    )

@show gtda_verity = norm(
[(orig_graph_distance[orig_pairs[i,1],orig_pairs[i,2]] - gtda_projected_dist[orig_pairs[i,1],orig_pairs[i,2]]) for i in range(1,size(orig_pairs,1))]
) 

@show tsne_error = norm(tsne_dist[tsne_pairs[:,1],:] - tsne_dist[tsne_pairs[:,2],:])
@show gtda_error = norm(gtda_projected_dist[gtda_pairs[:,1],:] - gtda_projected_dist[gtda_pairs[:,2],:])
@show orig_error = norm(orig_graph_distance[orig_pairs[:,1],:] - orig_graph_distance[orig_pairs[:,2],:])

@show tsne_seclusion =  abs(tsne_error - orig_error)/orig_error
@show gtda_seclusion =  abs(gtda_error - orig_error)/orig_error

@show time() - errortime

super_plot[:plot_title] = "tsne$(round(tsne_error,sigdigits = 4))gtda$(round(gtda_error,sigdigits = 4))"
plot!(super_plot)


######################################################################################################################################################

```galaxy part 2```

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

X = X .- mean(X)
X = X ./ std(X)
perm = randperm(sum(n))
X = X[perm,:]
labels = labels[perm]

orig = scatter(X[:,1],X[:,2],group = labels,framestyle=:none,axis_buffer=0.02,legend=:outertopleft, title = "$(n)-$(rad)")



```galaxy part 1```
Random.seed!(42)
n = [200, 200, 1000, 200, 1000] #1000 #different number of stars in each galxy
X = randn(sum(n),3) / sum(n)
labels = []
rad = [3,3,15,5] #[3, 10 , 3, 15]
X[1:n[1],1] += -3 .+ rad[1]*cos.([i for i in range(1,n[1])]*2*pi/n[1])
X[1:n[1],2] += -3 .+ rad[1]*sin.([i for i in range(1,n[1])]*2*pi/n[1])
append!(labels,[1 for _ in range(1,n[1])])
X[n[1]+1:sum(n[1:2]),1] += 5 .+ cos.([i for i in range(1,n[2])]*2*pi/n[2])
X[n[1]+1:sum(n[1:2]),2] += 5 .+ sin.([i for i in range(1,n[2])]*2*pi/n[2])
append!(labels,[2 for _ in range(n[1]+1,sum(n[1:2]))]);
X[sum(n[1:2])+1:sum(n[1:3]),1] += 10 .+ rad[2]*cos.([i for i in range(1,n[3])]*2*pi/n[3])
X[sum(n[1:2])+1:sum(n[1:3]),2] += 10 .+ rad[2]*sin.([i for i in range(1,n[3])]*2*pi/n[3])
append!(labels,[3 for _ in range(sum(n[1:2])+1,sum(n[1:3]))]);
X[sum(n[1:3])+1:sum(n[1:4]),1] +=  rad[3]*cos.([i for i in range(1,n[4])]*2*pi/n[4])
X[sum(n[1:3])+1:sum(n[1:4]),2] +=  rad[3]*sin.([i for i in range(1,n[4])]*2*pi/n[4])
append!(labels,[4 for _ in range(sum(n[1:3])+1,sum(n[1:4]))]);
X[sum(n[1:4])+1:sum(n[1:5]),1] += -3 .+ rad[4]*cos.([i for i in range(1,n[5])]*2*pi/n[5])
X[sum(n[1:4])+1:sum(n[1:5]),2] += -3 .+ rad[4]*sin.([i for i in range(1,n[5])]*2*pi/n[5])
append!(labels,[5 for _ in range(sum(n[1:4])+1,sum(n[1:5]))])

X = X .- mean(X)
X = X ./ std(X)
perm = randperm(sum(n))
X = X[perm,:]
labels = labels[perm]

orig = scatter(X[:,1],X[:,2],group = labels,framestyle=:none,axis_buffer=0.02,legend=:outertopleft, title = "$(n)-$(rad)")



embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,group= labels,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits = 4))")

begining = time()
Xnoisy,G,lens = topological_lens(X,3)
tym = time() - begining
        
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
        max_split_size=70,min_component_group=1,verbose=false,overlap = 0.15,
        split_thd=0,merge_thd = 0.01,labels = labels);
        
g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g)
xy = hcat(pos...)
rc = gtdaobj.reeb2node

p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=1.5,
    framestyle=:none, linecolor=:black,linealpha=0.6,
    axis_buffer=0.02,labels = "");
nodecolors = cgrad(:ice, length(unique(labels)), categorical = true);
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels)
end
title!(p,"GTDA_$(round((timereeb+tym),sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")

super_plot = plot(orig,t,p,layout = (1,3),size = (1600, 500),markerstrokewidth=0.5,titlefontsize = 10)


super_plot[:plot_title] = "tsne$(round(tsne_error,sigdigits = 4))gtda$(round(gtda_error,sigdigits = 4))"
plot!(super_plot)





```
distill - Two long, linear clusters in 2D.
```
Random.seed!(42)
n = 1000
l1 = []
l2 = []
for i in range(1,n)
    push!(l1,[i+randn(), i+randn()])
    push!(l2,[i+randn()+(n), i+randn()-(n)])
end
line1 = Matrix(hcat(l1...)')
line2 = Matrix(hcat(l2...)')
labels = []
append!(labels, [1 for _ in range(1,1000)])
append!(labels, [2 for _ in range(1,1000)])
Xnoisy = vcat(line1,line2);
Xnoisy = Xnoisy .- mean(Xnoisy)
Xnoisy = Xnoisy ./ std(Xnoisy)
lens = Xnoisy[:,2:end]
lens = lens ./norm(lens)
G = GraphTDA.canonicalize_graph(Xnoisy)
@show unique(scomponents(G).map)
spy(G)
gtdaobj = GraphTDA.analyzepredictions(lens,G = G,min_group_size=1,
               max_split_size=1,min_component_group=1,verbose=false,overlap = 0.1,
               split_thd=0.01,initial_clustering_threshold = 0,merge_thd = 0.01,labels = labels);





#= save the code for quickly dumping the lens and the graphj into a json file
using JSON
i,j,_ = findnz(G)
f = open("Glens500.json","w")
JSON.print(f,Dict("i"=>i,"j"=>j,"lens"=>lens))
close(f)

using JSON
f = JSON.parsefile("/Users/alice/Documents/Research/RPs/shifted_gaussians/TDA300python.json")
tdareeb = sparse(f["gi"] .+ 1,f["gj"] .+ 1,ones(length(f["gi"])))
_, xy = igraph_layout(tdareeb,"fr")
tdacomponents = f["gcomponents"];
tdacomponents = Dict([parse(Int64,k)+1=>v .+ 1 for (k,v) in tdacomponents])
tdacomponents = sort(tdacomponents)
rc = [val for val in values(tdacomponents)]
p = DiffusionTools.draw_graph(tdareeb,xy; linewidth=1.5,
framestyle=:none, linecolor=:black,linealpha=0.6,
axis_buffer=0.02,labels = "");
nodecolors = cgrad(:ice, length(unique(labels)), categorical = true)
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],3*sqrt(length(rc[i])/size(G,1)),p,nodecolors,showlabels)
end
plot!(p,title = "Mapper_10_0.25")
savefig("Mapper_10_0-25.png")
=#


_,tym_error,_,_ = @timed begin
tsne_graph = make_graph(embedding)
tsne_dist = get_graph_distance(tsne_graph)
gtda_dist = get_graph_distance(gtdaobj.G_reeb)
n = size(embedding,1)
gtda_projected_dist = Inf*ones(n,n)
for r in range(1,size(gtda_dist,1))
    #if it's in the same reebnode you make it 0, 
    #otherwise find the path length to the other reebnodes
    for i in range(1,size(gtda_dist,2))
        for j in gtdaobj.reeb2node[i]
        for k in gtdaobj.reeb2node[r]
            if i==r
                gtda_projected_dist[k,j] = 0
            else
                gtda_projected_dist[k,j] = min(gtda_projected_dist[k,j],gtda_dist[r,i])
            end
        end
        end
    end
end

orig_graph = make_graph(X)
orig_graph_distance = get_graph_distance(orig_graph)


tsne_dist .= ifelse.(tsne_dist .== Inf, 10^10, tsne_dist)
gtda_projected_dist .= ifelse.(gtda_projected_dist .== Inf, 10^10, gtda_projected_dist)
orig_graph_distance .= ifelse.(orig_graph_distance .== Inf, 10^10, orig_graph_distance)


tsne_pairs = topo_error.calculate_error(tsne_dist)
gtda_pairs = topo_error.calculate_error(gtda_projected_dist)
orig_pairs = topo_error.calculate_error(orig_graph_distance)

tsne_pairs = tsne_pairs .+ 1
gtda_pairs = gtda_pairs .+ 1
orig_pairs = orig_pairs .+ 1

tsne_error = norm(tsne_dist[tsne_pairs[:,1],:] - tsne_dist[tsne_pairs[:,2],:])
gtda_error = norm(gtda_projected_dist[gtda_pairs[:,1],:] - gtda_projected_dist[gtda_pairs[:,2],:])
orig_error = norm(orig_graph_distance[orig_pairs[:,1],:] - orig_graph_distance[orig_pairs[:,2],:])
@show rel_tsne = abs(tsne_error - orig_error)/orig_error
@show rel_gtda = abs(gtda_error - orig_error)/orig_error
end
@show tym_error

lemap = Float64.(lemap)
le_graph = make_graph(lemap)
le_dist = get_graph_distance(le_graph)
le_dist .= ifelse.(le_dist .== Inf, 10^10, le_dist)
@time le_pairs = topo_error.calculate_error(le_dist)
le_pairs = le_pairs .+ 1;
le_error = norm(le_dist[le_pairs[:,1],:] - le_dist[le_pairs[:,2],:])

super_plot[:plot_title] = "tsne$(round(rel_tsne,sigdigits = 4))gtda$(round(rel_gtda,sigdigits = 4))"
plot!(super_plot)
savefig




###measures to validate GTDA representation

#4. same <=> same ... different <=> same


    
################# IGNORE THIS #############
once = UpperTriangular(gtda_projected_dist);
    gtda_edges = []
    for i in range(1,size(once,1))
        for j in range(i+1,size(once,1))
            if once[i,j] != Inf
                push!(gtda_edges,[i, j])
            end
        end
    end
    tsne_graph = make_graph(tsne_emb, num_nn = nn);
    tsne_dist = get_graph_distance(tsne_graph);
    orig_graph = make_graph(orig, num_nn = nn);
    orig_edges = [[i, j] for (i,j) in zip(findnz(triu(orig_graph))[1],findnz(triu(orig_graph))[2])]
    tsne_edges = [[i, j] for (i,j) in zip(findnz(triu(tsne_graph))[1],findnz(triu(tsne_graph))[2])]
    orig_tsne = intersect(orig_edges,tsne_edges)
    orig_gtda = intersect(orig_edges,gtda_edges)
    push!(common_edges["gtda"],length(orig_gtda)/length(gtda_edges))
    push!(common_edges["tsne"],length(orig_tsne)/length(tsne_edges))
    orig_not_tsne = setdiff(orig_edges,tsne_edges)
    orig_not_gtda = setdiff(orig_edges,gtda_edges)
    vals = [tsne_dist[i[1],i[2]] for i in orig_not_tsne if tsne_dist[i[1],i[2]] != Inf]
    push!(true_negative["tsne"],[vals==[] ? 0 : mean(vals), vals==[] ? 0 : var(vals)])
    vals = [gtda_projected_dist[i[1],i[2]] for i in orig_not_gtda if gtda_projected_dist[i[1],i[2]] != Inf]
    push!(true_negative["gtda"],[vals==[] ? 0 : mean(vals), vals==[] ? 0 : var(vals)])
    tsne_not_orig = setdiff(tsne_edges,orig_edges)
    gtda_not_orig = setdiff(gtda_edges,orig_edges)
    vals = [norm(orig[i[1],:]-orig[i[2],:]) for i in tsne_not_orig]
    push!(false_positive["tsne"],[mean(vals), var(vals)])
    vals = [norm(orig[i[1],:]-orig[i[2],:]) for i in gtda_not_orig]
    push!(false_positive["gtda"],[mean(vals), var(vals)])
using DataStructures
@time begin 
    uf = IntDisjointSets(n)
    temp = findnz(triu(orig_graph))[1:2]
    for (i,j) in zip(temp[1],temp[2])
        a = find_root(uf,i)
        b = find_root(uf,j)
        if a==b
            continue
        else
            root_union!(uf,a,b)
        end
    end
    [find_root(uf, i) for i in range(1,n)]
end
################# IGNORE THIS #############


#2. geodesic distance

#3. "stress"

#1.silhouette index
fcf = sort(gtdaobj.final_components_filtered)
sil_index = -5*ones(size(G,1))
rnsizes = [length(i) for i in fcf] 
for rn in keys(fcf)
    suma = 0
    sumb = 0
    for i in fcf[rn]
        suma = (1/(length(fcf[rn])-1))*sum([norm(Xnoisy[i,:] - Xnoisy[j,:]) for j in fcf[rn]]) 
        otherdist = [(1/length(fcf[on]))*sum([norm(Xnoisy[i,:] - Xnoisy[k,:]) for k in fcf[on]]) for on in keys(fcf) if on!=rn]
        sumb  = minimum(otherdist)
        temp = (sumb - suma)/max(suma,sumb)
        if temp > sil_index[i]
            sil_index[i]= temp
        end
    end
end
temp = minimum(sil_index)
[sil_index[i] = temp for i in range(1,length(sil_index)) if sil_index[i] == -5]


function false_errors_and_accuracies(tsne_emb, G_reeb, rc, xy, orig; nns_to_test = [6])
    all_tsne_dist = []
    all_gtda_projected_dist = []
    all_orig_graph_distance = []
    all_tsne_pairs = []
    all_gtda_pairs = []
    all_orig_pairs = []
    n = size(tsne_emb,1)
    verity = zeros(size(nns_to_test,1),2)
    seclusion = zeros(size(nns_to_test,1),2)
    inc_nodes = []
    for i in rc
        append!(inc_nodes,i)
    end
    inc_nodes = unique(inc_nodes);
    tsne_emb = tsne_emb[inc_nodes,:];
    orig = orig[inc_nodes,:];
    for (i,nn) in enumerate(nns_to_test)
        tsne_graph = make_graph(tsne_emb, num_nn = nn);
        tsne_dist = get_graph_distance(tsne_graph);
        gtda_dist = get_graph_distance(G_reeb);
        gtda_projected_dist = Inf*ones(n,n);
        for r in range(1,length(rc))
            for j in range(1, length(rc))
                gtda_projected_dist[rc[r],rc[r]] .= 0
                gtda_projected_dist[rc[r],rc[j]] .= min.(gtda_dist[r,j],gtda_projected_dist[rc[r],rc[j]])
            end
        end
        @assert gtda_projected_dist == gtda_projected_dist' 
        gtda_projected_dist = gtda_projected_dist[inc_nodes,inc_nodes];

        orig_graph = make_graph(orig, num_nn = nn);
        orig_graph_distance = get_graph_distance(orig_graph);


        tsne_dist .= ifelse.(tsne_dist .== Inf, 10^10, tsne_dist);
        gtda_projected_dist .= ifelse.(gtda_projected_dist .== Inf, 10^10, gtda_projected_dist);
        orig_graph_distance .= ifelse.(orig_graph_distance .== Inf, 10^10, orig_graph_distance);
        
        orig_dist = zeros(size(orig,1),size(orig,1))
        for i in range(1,size(orig,1))
        for j in range(i+1,size(orig,1))
        orig_dist[i,j] = norm(orig[i,:] - orig[j,:])
        end
        end
        orig_dist = max.(orig_dist, orig_dist')
        [orig_graph_distance[i] = orig_dist[i] for i in findall(j->j==Inf,orig_graph_distance)];

        gtda_dist = zeros(size(xy))
        for i in range(1,size(xy,1))
        for j in range(i+1,size(xy,1))
        gtda_dist[i,j] = norm(xy[i,:] - xy[j,:])
        end
        end
        gtda_dist = max.(gtda_dist, gtda_dist')


        @time tsne_pairs = topo_error.calculate_error(tsne_dist) .+ 1;
        @time gtda_pairs = topo_error.calculate_error(gtda_projected_dist) .+ 1;
        @time orig_pairs = topo_error.calculate_error(orig_graph_distance) .+ 1;
        
        
        push!(all_tsne_dist,tsne_dist)
        push!(all_gtda_projected_dist,gtda_projected_dist)
        push!(all_orig_graph_distance,orig_graph_distance)
        push!(all_tsne_pairs,tsne_pairs)
        push!(all_gtda_pairs,gtda_pairs)
        push!(all_orig_pairs,orig_pairs)
    end
    return all_tsne_dist, all_gtda_projected_dist, all_orig_graph_distance, all_tsne_pairs, all_gtda_pairs, all_orig_pairs
end
    
#= #do this the first time
ENV["PYTHON"] = "/usr/bin/python3" 
import Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
=#

using PyCall
println(PyCall.python)
pushfirst!(pyimport("sys")."path","/Users/alice/Documents/Research/RPs/shifted_gaussians")

topo_error = pyimport("topological_error")
    
    