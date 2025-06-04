include("GraphTDA.jl")
include("../utils.jl") #getlayout
include("centralcode.jl") 
include("candidates.jl") #getemall
include("codes_synthetic.jl") 

include("plottingGTDR.jl")

"""script for generating pictures using GTDR and all other methods"""

#name ="synthdev" --
function synthdev()
    @info "Synthetic developmental data"
    name ="synthdev"

    temp = npzread("/p/mnt/homes/dshur/topo_dim_red/synthetic_data/developmental_data_20.npz")
    X = temp["data"]
    X = X .- mean(X);
    X = X ./ var(X);
    origlabels = temp["label"] .+ 1;

    labels = zeros(size(origlabels))
    [labels[i] = 2 for i in range(1,length(labels)) if (origlabels[i] == 20.0)]
    [labels[i] = 3 for i in range(1,length(labels)) if (origlabels[i] == 10.0)]
    [labels[i] = 4 for i in range(1,length(labels)) if (origlabels[i] == 1.0)]
    [labels[i] = 1 for i in range(1,length(labels)) if (origlabels[i] != 1.0 && origlabels[i] != 10.0 && origlabels[i] != 20.0)]
   
    perm = randperm(length(origlabels))
    X = X[perm,:]
    labels = labels[perm];
    
    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=3
    max_split_size=40
    min_component_group=5
    overlap = 0.3

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=40,min_component_group=5,verbose=false,overlap = 0.3,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()

end

#name = "galaxies" --
function galaxies()
    @info "Synthetic galaxies"
    name = "galaxies"

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


    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,3,dims = 3)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)
    
    min_group_size=10
    max_split_size=70
    min_component_group=1
    overlap = 0.15

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
            max_split_size=70,min_component_group=1,verbose=false,overlap = 0.15,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()


end

#name = "synthbranches" ---
function synthbranches()
    @info "Synthetic branches as in PHATE"
    name = "synthbranches"
    n_dim = 200
    n_branch = 10
    branch_length = 300
    rand_multiplier = 2
    seed=37
    sigma = 5
    X,labels = synthetic_tree(n_dim, n_branch, branch_length, rand_multiplier, seed, sigma); #fn defined in codes_synthetic

    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,25,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=3
    max_split_size=50
    min_component_group=1
    overlap = 0.2

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=50,min_component_group=1,overlap = 0.2,verbose=false,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()

end

#name = "parallelines" ---
function parallelines()
    name = "parallelines"
    Random.seed!(123)
    n = 2000
    d = 100 
    k = 5
    sep = 10 
    X,labels = create_hashtag_sign(n, d, k, [sep*i for i in range(1,k-1)]); #fn in codes_synthetic

    X = X .- mean(X);
    X = X ./ std(X);

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=3
    max_split_size=200
    min_component_group=1
    overlap = 0.01

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=200,min_component_group=1,overlap = 0.01,verbose=false,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()

end

#name = "kDlines" ---
function kDlines()
    name = "kDlines"
    Random.seed!(12)
    k, d, n, separation, line_length = 3, 100, [1000 for _ in range(1,3)], [5.0 for _ in range(1,3)], [10.0 for _ in range(1,3)]
    X,labels = generate_lines(k, d, n, separation, line_length); #fn defined in codes_synthetic
    X = X .- mean(X);
    X = X ./ std(X);
    perm = randperm(size(X,1));
    X = X[perm,:]
    labels = labels[perm]


    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=5
    max_split_size=200
    min_component_group=1
    overlap = 0.15

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
            max_split_size=200,min_component_group=1,overlap = 0.15,verbose=false,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()


end

#name = "seprngauss" --
function seprngauss()
    name = "seprngauss"

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

    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=3
    max_split_size=400
    min_component_group=1
    overlap = 0.3

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=400,min_component_group=1,overlap = 0.3,verbose=false,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()

end

#name = "varncgauss" ---
function varncgauss()
    name = "varncgauss"

    Random.seed!(123)
    k = 3
    d = 100
    n = [1000, 1000, 1000]
    sep = [10*i for i in range(1,k-1)]
    varnc = [1, 1, 10]
    points, labels = generate_multivariate_gaussians(k,d,n,sep, var = varnc);
    X = Matrix(points');
    X = X .- mean(X);
    X = X ./ std(X);

    @info "got data of size" size(X,1)
    
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    @info "graph of size" size(Xgraph)
    @info "lens of size" size(lens)

    min_group_size=3
    max_split_size=400
    min_component_group=1
    overlap = 0.3

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=400,min_component_group=1,overlap = 0.3,verbose=false,
            split_thd=0,merge_thd = 0.01,labels = labels);

    @show size(gtdaobj.G_reeb)

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name).json","w")
    JSON.print(f,combo_dict) 
    close(f)
        
    println("doing different methods")
    getemall(X,name,num_nn = 10)
    println("done")

    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()

end

#parallelines()
#kDlines()
#galaxies()