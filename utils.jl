
using DelimitedFiles,Random, Statistics, Distributions, LinearAlgebra, StatsBase, NearestNeighbors, 
      SparseArrays, DataFrames, JSON, JLD2, FileIO, MatrixNetworks, Graphs, GraphPlot, NPZ


function draw_pie(dist_dict, xpos, ypos, msize, ax, color_map, showlabels;lw = 2.0, actuallabels = [])
    dist = [v for (k,v) in dist_dict]
    distkey = [Int(k) for (k,v) in dist_dict]
    if actuallabels==[]
        actuallabels = distkey
    end
    temp = cumsum(dist)
    temp = temp ./ temp[end]
    pie = [0;temp]
    for (r1, r2, dkey) in zip(pie[1:end-1], pie[2:end], distkey)
        angles = LinRange(2 * π * r1, 2 * π * r2, 50)
        x = msize .* [0;cos.(angles)]
        y = msize .* [0;sin.(angles)]
        points = hcat(x .+ xpos, y .+ ypos)
        if dkey in showlabels
            c = findfirst(i->distkey[i]==dkey,range(1,length(distkey)))
            if actuallabels != distkey
                labelstoplot = actuallabels[distkey[c]]
            else
                labelstoplot = actuallabels[c]
            end
            poly = PyPlot.matplotlib.patches.Polygon(points, closed=true,
                        facecolor=color_map[dkey], edgecolor="none")#, label=string(labelstoplot))
        else
            poly = PyPlot.matplotlib.patches.Polygon(points, closed=true,
                facecolor=color_map[dkey], edgecolor="none")
        end
        ax.add_patch(poly)
    end
    angles = LinRange(0, 2 * π, 100)
    x = msize .* cos.(angles)
    y = msize .* sin.(angles)
    points = hcat(x .+ xpos,y .+ ypos)
    poly = PyPlot.matplotlib.patches.Polygon(points,
        closed=true,alpha=0.01, edgecolor="black", linewidth=lw)
    ax.add_patch(poly)
    return ax
end


function getlayout(A::SparseMatrixCSC)
    """
    takes as input the reeb graph adjacency matrix and returns xy coordinates for all the considered layout candidates
    g = SimpleGraph(A,layout)
    """
    nx = pyimport("networkx")
    layout_cands = Dict(nx.spring_layout=>[],nx.spectral_layout=>[],
			nx.kamada_kawai_layout=>[])#,nx.planar_layout=>[],nx.random_layout=>[])
    g = nx.Graph()
    edgelist = hcat(findnz(A)[1:2]...)
    for i in range(1,size(edgelist,1))
        g.add_edge(edgelist[i,1],edgelist[i,2])
    end
    [push!(layout_cands[i],zeros(size(A,1),2)) for i in keys(layout_cands)]
    for i in keys(layout_cands)
        @show i
        dict_cords = i(g)
        [layout_cands[i][1][j,:] = dict_cords[j] for (j,v) in dict_cords]
    end
    return layout_cands
end

function make_graph(X;leafsize = 25,num_nn = 6)
    @show num_nn
    @show size(X)
    kdtree = KDTree(X'; leafsize = leafsize)
    idxs, dists = NearestNeighbors.knn(kdtree, X', num_nn, true);
    ei = []
    ej = []
    for i in idxs
        for j in range(2,length(i))
            append!(ei,i[1])
            append!(ej,i[j])
        end
    end
    G = sparse(ei,ej,ones(length(ei)),size(X,1),size(X,1))
    G = max.(G,G')
    return G
end

function get_graph_distance(reps_graph)
    #the input here is the graph on which you want to calculate the distance metric
    dist_matrix = zeros(size(reps_graph));
    for i in 1:size(reps_graph,1)
        # Get shortest paths from node i to all other nodes
        path_length,_ = MatrixNetworks.dijkstra(reps_graph, i)
        dist_matrix[i, :] = path_length
    end
    return dist_matrix
end

function draw_graph_segments(A::SparseMatrixCSC, xy)
    ei, ej = findnz(triu(A,1))[1:2]
    linesx = Vector{Vector{Float64}}()
    linesy = Vector{Vector{Float64}}()
    for nz in 1:length(ei)
        src, dst = ei[nz], ej[nz]
        push!(linesx, [xy[src,1], xy[dst,1]])
        push!(linesy, [xy[src,2], xy[dst,2]])
    end
    return linesx, linesy
end


function triplet_accuracy(G,cand_G;nodes = [],sample_size = 3000, ntrials = 3)
    #https://github.com/danchern97/RTD_AE
    #inputs are the distance graphs, nodes = unique([v for v in gtdaobj.reeb2node])
    acc = []
    n = size(G,1)
    if length(nodes) == 0
        nodes = collect(range(1,n))
    else
        G = G[nodes,nodes]
        n = length(nodes)
    end
    for _ in range(1,ntrials)
        triplets = collect([rand(1:n,3) for _ in range(1,min(sample_size,n))]) #collect(combinations(1:length(nodes), 3))
        is = [i[1] for i in triplets];
        js = [i[2] for i in triplets];
        ks = [i[3] for i in triplets];
        append!(acc,sum(((G[is,js] .< G[js,ks]) .- (cand_G[is,js] .< cand_G[js,ks])) .== 0)/length(is)^2)
    end
    return mean(acc)
end

function errors_and_accuracies(G_orig, G_cand; cand_name = "GTDA", rc=[], rns=[])
    #candidates = ["umap", "pca", "tsne", "pacmap", "localmap", "trimap", "mds", "paga"] 
    """for calculating spearman correlation
    take input as candidate graph G_cand and original graph G_orig 
    if candidate is GTDA then check for reebnodes components nodes and nodes part of reebnodes
    if candidate is others do calculations on the graph
    """
    acc, err = [],[]
    n = size(G_orig,1)
    orig_graph_distance = get_graph_distance(G_orig);
    cand_dist = zeros(n,n);
    if cand_name == "GTDA"
        gtda_dist = get_graph_distance(G_cand);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n) 
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                cand_dist[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        cand_dist = max.(cand_dist,cand_dist');
        cand_dist = cand_dist - I;
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        ta = triplet_accuracy(G_orig,cand_dist,nodes = inc_nodes)
    else
        cand_dist = get_graph_distance(G_cand);
        ta = triplet_accuracy(G_orig, G_cand)
    end
    sc_orig = scomponents(G_orig);
    for cs in range(1,sc_orig.number)
        inodes = findall(j->j==cs,sc_orig.map);
        if cand_name == "GTDA"
            rnodes = unique(vcat(reebnodes[inodes]...));
            rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_cand[rnodes,rnodes])[2]);
            rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
            rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
            temp = [StatsBase.corspearman(cand_dist[c,rlvnt_nodes],orig_graph_distance[c,rlvnt_nodes]) for c in rlvnt_nodes]
            temp = length(temp) == 0 ? 0 : temp
            push!(acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
            ir_nodes = setdiff(inodes, rlvnt_nodes)
            temp = [StatsBase.corspearman(cand_dist[c,ir_nodes],orig_graph_distance[c,ir_nodes]) for c in ir_nodes]
            temp = length(temp) == 0 ? 0 : temp
            push!(err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
        else
            can_rlvnt_nodes = findall(x->x==1,largest_component(G_cand[inodes,inodes])[2]);
            can_rlvnt_nodes = inodes[can_rlvnt_nodes];
            temp= [StatsBase.corspearman(cand_dist[c,can_rlvnt_nodes],orig_graph_distance[c,can_rlvnt_nodes]) for c in can_rlvnt_nodes]
            temp = length(temp) == 0 ? 0 : temp
            push!(acc, [mean(temp), var(temp),length(can_rlvnt_nodes)/length(inodes)])
            can_ir_nodes = setdiff(inodes,can_rlvnt_nodes);
            temp= [StatsBase.corspearman(cand_dist[c,can_ir_nodes],orig_graph_distance[c,can_ir_nodes]) for c in can_ir_nodes]
            temp = length(temp) == 0 ? 0 : temp
            push!(err, [mean(temp), var(temp),length(can_ir_nodes)/length(inodes)])
        end
    end
    avg_acc = [mean([c[1] for c in acc]), mean([c[2] for c in acc]), mean([c[3] for c in acc])]
    avg_err = [mean([c[1] for c in err]), mean([c[2] for c in err]), mean([c[3] for c in err])]

    return avg_acc, avg_err, ta
end


function errors_overall(X_orig,dataname;nns_to_test = [6,10,15,20],G_cand = nothing,cand_embs =nothing, rc = nothing,rns =nothing)
    #all toegther to ensure comparision with the same G_orig, 
    # and cand embs is one huge matrix
    candyd8s = ["umap", "pca", "tsne", "pacmap", "localmap", "trimap", "mds", "paga"] 
    all_errors = Dict([k=>[] for k in nns_to_test])
    for (i,num_nn) in enumerate(nns_to_test)
        G_orig = make_graph(X_orig, num_nn = num_nn);
        cand_errors = Dict([k=>[] for k in candyd8s])
        for cand_name in candyd8s
            X_cand = allothermethods[cand_name][1][dataname][1]
            #methods returning embedding
            G_cand =  make_graph(X_cand, num_nn = nn);
            acc, err, ta = errors_and_accuracies(G_orig, G_cand, cand_name = cand_name)
            push!(cand_errors[cand_name],Dict("acc"=>acc, "err"=>err, "ta"=>ta))
        end
        #gtda - methods returning graphs
        cand_name = "GTDA"
        acc, err, ta = errors_and_accuracies(G_orig, G_cand, cand_name = cand_name, rc = rc, rns = rns)
        push!(cand_errors[cand_name],Dict("acc"=>acc, "err"=>err, "ta"=>ta))
        push!(all_errors[num_nn],cand_errors)
    end
    return all_errors
end   





#=
using Ripserer
function rtd(x_dist, z_dist)
    #https://github.com/danchern97/RTD_AE
    function lp_loss(a, b, p=2)
        return sum(abs.(a .- b) .^ p)
    end

    # Function to extract indices from Rips complex
    function get_indices(DX, rc, dim, card)
        dgm = rc[dim+1]  # Persistence diagram for given dimension
        indices, pers = [], []

        for (birth, death, simplex) in dgm
            if length(simplex) == dim+1
                i1 = argmax(DX[simplex, simplex])  # Get max pairwise distance
                indices = vcat(indices, simplex[i1])
                pers = vcat(pers, death - birth)
            end
        end

        perm = sortperm(pers, rev=true)
        indices = indices[perm][1:min(4*card, length(indices))]
        return vcat(indices, zeros(Int, max(0, 4*card - length(indices))))
    end

    # Function to compute Rips persistence
    function Rips(DX, dim, card)
        dim = max(dim, 1)
        rc = ripserer(DX, dim_max = dim)
        all_indices = [get_indices(DX, rc, d, card) for d in 1:dim]
        return all_indices
    end

    #Persistence calculation
    function RTD_Differentiable(Dr1,Dr2,dim = 1,card = 50)
        Dzz = rand(0.1:0.5,size(Dr1));
        Dr12 =  min.(Dr1, Dr2);# : max.(Dr1, Dr2)
        DX = [Dr1 Dr1'; Dr1 Dr12];
        #make symmetric
        DX = (DX .+ DX') / 2.0 ;
        DX = DX .- diagm(diag(DX));
        @show all_ids = Rips(DX, dim, card)
        all_dgms = []
        for ids in all_ids
            tmp_idx = reshape(ids, (2*card, 2))
            dgm = hcat(reshape(DX[tmp_idx[1:2:end, 1], tmp_idx[1:2:end, 2]], (card, 1)),
           reshape(DX[tmp_idx[2:2:end, 1], tmp_idx[2:2:end, 2]], (card, 1)))
            push!(all_dgms, dgm)
        end

        return all_dgms
    end

    @show rtd_xz = RTD_Differentiable(x_dist, z_dist)
    loss_xz = 0
    for (d, rtd) in enumerate(rtd_xz)
        loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=1.0)
    end
    return loss_xz
end

=#




