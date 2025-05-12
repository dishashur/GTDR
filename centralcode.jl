include("GraphTDA.jl")


function segregated_topological_lens(X,k;dims = 10,seed = 10, usefull = false)
    """
    This function is meant to analyze the effect of the overlap of the projected columns 
    """
    Random.seed!(seed)
    A = randn(size(X,2),dims)
    X = X .- mean(X)
    X = X ./ std(X)
    Xnoisy = X*A
    Xnoisy = Xnoisy .- mean(Xnoisy)
    Xnoisy = Xnoisy ./ std(Xnoisy)
    Xnoisy = Xnoisy[:,randperm(size(Xnoisy,2))]
    if usefull_graph
        Xgraph = Xnoisy
    else
        Xgraph = Xnoisy[:,1:dims-k]
    end
    kdtree = KDTree(Xgraph'; leafsize = 25)
    idxs, dists = NearestNeighbors.knn(kdtree, Xgraph', 10, true);
    ei = []
    ej = []
    for i in idxs
        for j in range(2,length(i))
            append!(ei,i[1])
            append!(ej,i[j])
        end
    end
    G = sparse(ei,ej,ones(length(ei)),size(Xgraph,1),size(Xgraph,1))
    G = max.(G,G')
    lens = Xnoisy[:,end-k+1:end]
    lens = lens .- mean(lens)
    lens = lens ./ norm(lens)
    @show norm(lens)
    @show var(lens)
    return Xnoisy, Xgraph, G, lens
end

function topological_lens(X,k;dims = 10,seed = 10)
    """
    This function is the main function used where full columns are used for developing the graph and any of the 
    k columns are used for lens
    """
    Random.seed!(seed)
    A = randn(size(X,2),dims)
    X = X .- mean(X)
    X = X ./ std(X)
    Xnoisy = X*A
    Xnoisy = Xnoisy .- mean(Xnoisy)
    Xnoisy = Xnoisy ./ std(Xnoisy)
    Xgraph = Xnoisy
    kdtree = KDTree(Xgraph'; leafsize = 25)
    idxs, dists = NearestNeighbors.knn(kdtree, Xgraph', 10, true);
    ei = []
    ej = []
    for i in idxs
        for j in range(2,length(i))
            append!(ei,i[1])
            append!(ej,i[j])
        end
    end
    G = sparse(ei,ej,ones(length(ei)),size(Xgraph,1),size(Xgraph,1))
    G = max.(G,G')
    lens = Xnoisy[:,randperm[size(Xnoisy,2)]][:,end-k+1:end]
    lens = lens .- mean(lens)
    lens = lens ./ norm(lens)
    @show norm(lens)
    @show var(lens)
    return Xnoisy, Xgraph, G, lens
end