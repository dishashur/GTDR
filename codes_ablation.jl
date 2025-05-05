include("/Users/alice/Documents/Research/RPs/mainidea.jl")



function phate()
    name = "PHATE"
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
            max_split_size=50,min_component_group=1,verbose=false,overlap = 0.3,
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

    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[], "projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=50,min_component_group=1,verbose=false,overlap = 0.3,
            split_thd=0.001,merge_thd = 0.01,labels = labels);
        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n)
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                dists[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G)
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                @show length(gtda_acc)
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(gtda_err)
                
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                @show length(projected_acc)
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(projected_err)
            end
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end

    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:star,label = "included_nodes")
    
    title!(pos,"$(name)-$(n_branch)-$(n_dim)-$(branch_length)positive")


    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n_branch)-$(n_dim)-$(branch_length)negative")

    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n_branch)-$(n_dim)-$(branch_length)triplet_accuracy")
    
    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend =:outertopright)
    
    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n_branch)-$(n_dim)-$(branch_length)")


    open("$(name)-$(n_branch)-$(n_dim)-$(branch_length).json", "w") do io
        JSON.print(io, soln)
    end

end


function lines()
    name = "Lines"
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
    k, d, n, separation, line_length = 3, 200, [2000 for _ in range(1,3)], [5.0 for _ in range(1,3)], [10.0 for _ in range(1,3)]
    X,labels = generate_lines(k, d, n, separation, line_length);
    X = X .- mean(X);
    X = X ./ std(X);
    perm = randperm(size(X,1));
    X = X[perm,:];
    labels = labels[perm];
    nodecolors = distinguishable_colors(n_branch, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    orig= scatter(X[:,1],X[:,2],markerstrokewidth=0.5,markersize=3.0,framestyle=:none,
    axis_buffer=0.02,legend = :outertopleft, color = nodecolors[labels], group = labels,legendfontsize = 10.0,
    title = "$(n[1])-$(d)")
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining
    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
            max_split_size=300,min_component_group=1,verbose=false,overlap = 0.15,
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
    gactual = SimpleGraph(G)
    pos = spring_layout(gactual)
    xyactual = hcat(pos...)
    pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
    linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")
    scatter!(pactual,xyactual[:,1],xyactual[:,2],group = labels)
    
    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end
    
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[],"projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
            max_split_size=500,min_component_group=1,verbose=false,overlap = 0.15,
            split_thd=0.001,merge_thd = 0.01,labels = labels);
        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n)
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                dists[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G)
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                @show length(gtda_acc)
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(gtda_err)
                
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                @show length(projected_acc)
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(projected_err)
            end
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end
    
    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:xcross,label = "included_nodes")
    title!(pos,"$(name)-$(n[1])-$(d)positive")
    
    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n[1])-$(d)negative")
    
    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n[1])-$(d)triplet_accuracy")
    
    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend =:outertopright)
    
    

    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n[1])-$(d)")
    
    
    open("$(name)-$(n[1])-$(d).json", "w") do io
        JSON.print(io, soln)
    end
    
end


function hashtag()
    name = "hashtag"
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
    d = 150 
    k = 10 #3,5,10
    sep = 5 # 5, 10
    X,labels = create_hashtag_sign(n, d, k, [sep*i for i in range(1,k-1)]);

    X = X .- mean(X);
    X = X ./ std(X);
    nodecolors = distinguishable_colors(length(unique(labels)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    orig= scatter(X[:,1],X[:,2],markerstrokewidth=0.5,markersize=3.0,framestyle=:none,
    axis_buffer=0.02,legend = :outertopleft, color = nodecolors[labels], group = labels,legendfontsize = 10.0,
    title = "$(n)-$(d)-$(sep)")

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,Int(0.7*30),dims = 30)
    tym1 = time() - begining

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=200,min_component_group=1,verbose=false,overlap = 0.3,
            split_thd=0,merge_thd = 0.01,labels = labels);

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
    title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)",legend=:outertopleft)
    
    gactual = SimpleGraph(G)
    pos = spring_layout(gactual)
    xyactual = hcat(pos...)
    pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
    linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")
    scatter!(pactual,xyactual[:,1],xyactual[:,2],group = labels)

    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end

    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[],"projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=200,min_component_group=1,verbose=false,overlap = 0.3,
            split_thd=0,merge_thd = 0.01,labels = labels);
        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n)
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                dists[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G);
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
              
            end
            @show fracs[i]
            @show nn
            @show sc_orig.number
            @show [c[3] for c in gtda_acc]
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end
    
    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:xcross,label = "included_nodes")
    title!(pos,"$(name)-$(n[1])-$(d)positive",legend=:outertopleft)

    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n[1])-$(d)negative",legend=:outertopleft)

    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n[1])-$(d)triplet_accuracy")
    
    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend =:outertopright)

    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n[1])-$(d)")


    open("$(name)-$(n[1])-$(d).json", "w") do io
        JSON.print(io, soln)
    end
end


function rw_distill()
    name = "rw_distill"
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
    n = 3000
    d = 100
    Random.seed!(123)
    points = randomWalk(n,d);
    X = Matrix(hcat(points...)');
    X = X .- mean(X);
    X = X ./ std(X);
    labels = [norm(X[i,:]) for i in range(1,n)];
    nodecolors = cgrad(:ice, Int(round(maximum(labels)) + 1), categorical = true)
    finalcolors = [nodecolors[Int(round(i))] for i in labels] 

    orig = scatter(X[:,1],X[:,2],color = finalcolors,markerstrokewidth=0.5,markersize=3.0,
    framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)",label = "",group = labels)
    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
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
    gactual = SimpleGraph(G)
    pos = spring_layout(gactual)
    xyactual = hcat(pos...)
    pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
    linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")
    scatter!(pactual,xyactual[:,1],xyactual[:,2])

    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[],"projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=500,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);
        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for ii in range(1,n)
        if reebnodes[ii] != []
            for jj in range(ii,n)
            if reebnodes[jj] != []
                dists[ii,jj] = 1+minimum([gtda_dist[a,b] for a in reebnodes[ii], b in reebnodes[jj]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G)
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                @show length(gtda_acc)
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(gtda_err)
                
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                @show length(projected_acc)
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(projected_err)
            end
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end

    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:star,label = "included_nodes")
    title!(pos,"$(name)-$(n[1])-$(d)positive")


    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n[1])-$(d)negative")

    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n[1])-$(d)triplet_accuracy")


    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend=:outertopright)
    
    
    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n[1])-$(d)")


    open("$(name)-$(n[1])-$(d).json", "w") do io
        JSON.print(io, soln)
    end


end

function galaxy()
    name = "galaxy"
    Random.seed!(42)
    n = [500, 500, 2000, 500, 2000] #1000 #different number of stars in each galxy
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
    nodecolors = distinguishable_colors(length(n), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    orig = scatter(X[:,1],X[:,2],group = labels,color = nodecolors[labels],framestyle=:none,axis_buffer=0.02,legend=:outertopleft, title = "$(n)-$(rad)")

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym = time() - begining
            
    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
            max_split_size=200,min_component_group=1,verbose=false,overlap = 0.15,
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
        p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels)
    end
    title!(p,"GTDA_$(round((timereeb+tym),sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)")
    gactual = SimpleGraph(G)
    pos = spring_layout(gactual)
    xyactual = hcat(pos...)
    pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
    linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")
    scatter!(pactual,xyactual[:,1],xyactual[:,2],group = labels)

    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[],"projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
            max_split_size=200,min_component_group=1,verbose=false,overlap = 0.15,
            split_thd=0,merge_thd = 0.01,labels = labels);
        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n)
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                dists[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G)
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                @show length(gtda_acc)
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(gtda_err)
                
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                @show length(projected_acc)
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(projected_err)
            end
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end

    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:star,label = "included_nodes")
    title!(pos,"$(name)-$(n[1])-$(d)positive")


    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n[1])-$(d)negative")

    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n[1])-$(d)triplet_accuracy")


    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend=:outertopright)
    
    
    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n[1])-$(d)")


    open("$(name)-$(n[1])-$(d).json", "w") do io
        JSON.print(io, soln)
    end


end


function randomwalk()
    name = "Random walk"
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
    n = 5000
    p = 0.1
    X = random_walk(n,d, p=p);
    X = X .- mean(X);
    X = X ./ std(X);
    colors = [norm(X[i,:]) for i in range(1,n)];
    labels = colors;

    orig = scatter(X[:,1],X[:,2],zcolor = colors,markerstrokewidth=0.5,markersize=3.0,
    framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)_$(p)",label = "")

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym = time() - begining
            
    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=700,min_component_group=3,verbose=false,overlap = 0.2,
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
    
    
    gactual = SimpleGraph(G)
    pos = spring_layout(gactual)
    xyactual = hcat(pos...)
    pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
    linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")
    scatter!(pactual,xyactual[:,1],xyactual[:,2],zcolor = colors,markerstrokewidth=0.5)

    nns_to_test = [6, 10, 15, 20]
    orig_graph = []
    orig_graph_distance = []
    for nn in nns_to_test
        push!(orig_graph,make_graph(X, num_nn = nn))
        @show length(orig_graph)
        push!(orig_graph_distance,get_graph_distance(orig_graph[end])) 
        @show length(orig_graph_distance)
    end

    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    soln = [Dict("inc_nodes"=>[],"projected_ta"=>[],"gtda_ta"=>[],"true_positive_gtda"=>[],"true_negative_gtda"=>[],"true_positive_projected"=>[],"true_negative_projected"=>[]) for _ in range(1,length(fracs))]
    for i in range(1,length(fracs))
        begining = time()
        num_cols = Int(fracs[i]*30);
        Xnoisy,Xgraph,G,lens = topological_lens(X,num_cols,dims = 30)
        @show size(Xgraph)
        @show size(lens)
        tym1 = time() - begining
        
        gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
            max_split_size=700,min_component_group=3,verbose=false,overlap = 0.2,
            split_thd=0,merge_thd = 0.01,labels = labels);

        G_reeb = gtdaobj.G_reeb
        rns = gtdaobj.node2reeb
        rc = gtdaobj.reeb2node
        inc_nodes = []
        for i in rc
            append!(inc_nodes,i)
        end
        inc_nodes = unique(inc_nodes);
        @show length(inc_nodes)
        n = size(X,1)
        gtda_dist = get_graph_distance(G_reeb);
        dists = zeros(n,n);
        reebnodes = [[] for _ in range(1,n)];
        [reebnodes[k] = Int.(v) for (k,v) in rns];
        t = time()
        for i in range(1,n)
        if reebnodes[i] != []
            for j in range(i,n)
            if reebnodes[j] != []
                dists[i,j] = 1+minimum([gtda_dist[a,b] for a in reebnodes[i], b in reebnodes[j]])
            end
            end
        end
        end
        tover = time() - begining
        @show tover
        dists = max.(dists,dists');
        dists = dists - I;  
        sc_gtda = scomponents(G_reeb);
        noisy_graph_distance = get_graph_distance(G);
        for (j,nn) in enumerate(nns_to_test)
            sc_orig = scomponents(orig_graph[j]);
            @show "projected"
            projected_ta = triplet_accuracy(orig_graph_distance[j],noisy_graph_distance,nodes = inc_nodes)
            @show "gtda"
            gtda_ta = triplet_accuracy(orig_graph_distance[j],dists,nodes = inc_nodes)
            gtda_acc, gtda_err, projected_err, projected_acc =[],[],[],[]
            for cs in range(1,sc_orig.number)
                inodes = findall(j->j==cs,sc_orig.map);
                rnodes = unique(vcat(reebnodes[inodes]...));
                rlvnt_reeb_nodes =  findall(x->x==1,largest_component(G_reeb[rnodes,rnodes])[2]);
                rlvnt_reeb_nodes = rnodes[rlvnt_reeb_nodes];
                rlvnt_nodes = unique(vcat(rc[rlvnt_reeb_nodes]...));
                temp = [StatsBase.corspearman(dists[c,rlvnt_nodes],orig_graph_distance[j][c,rlvnt_nodes]) for c in rlvnt_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_acc,[mean(temp), var(temp),length(rlvnt_nodes)/length(inodes)])
                @show length(gtda_acc)
                ir_nodes = setdiff(inodes, rlvnt_nodes)
                temp = [StatsBase.corspearman(dists[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(gtda_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(gtda_err)
                
                rlvnt_noisy_nodes = findall(x->x==1,largest_component(G[inodes,inodes])[2]);
                rlvnt_noisy_nodes = inodes[rlvnt_noisy_nodes];
                temp= [StatsBase.corspearman(noisy_graph_distance[c,rlvnt_noisy_nodes],orig_graph_distance[j][c,rlvnt_noisy_nodes]) for c in rlvnt_noisy_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_acc, [mean(temp), var(temp),length(rlvnt_noisy_nodes)/length(inodes)])
                @show length(projected_acc)
                ir_nodes = setdiff(inodes, rlvnt_noisy_nodes)
                temp = [StatsBase.corspearman(noisy_graph_distance[c,ir_nodes],orig_graph_distance[j][c,ir_nodes]) for c in ir_nodes]
                temp = length(temp) == 0 ? 0 : temp
                push!(projected_err,[mean(temp), var(temp),length(ir_nodes)/length(inodes)])
                @show length(projected_err)
            end
            push!(soln[i]["true_positive_gtda"],[mean([c[3] for c in gtda_acc]), mean([c[1]*c[3] for c in gtda_acc]), std([c[1]*c[3] for c in gtda_acc])])
            push!(soln[i]["true_negative_gtda"],[mean([c[3] for c in gtda_err]), mean([c[1]*c[3] for c in gtda_err]), std([c[1]*c[3] for c in gtda_err])])
            push!(soln[i]["true_positive_projected"],[mean([c[3] for c in projected_acc]), mean([c[1]*c[3] for c in projected_acc]), std([c[1]*c[3] for c in projected_acc])])
            push!(soln[i]["true_negative_projected"],[mean([c[3] for c in projected_err]), mean([c[1]*c[3] for c in projected_err]), std([c[1]*c[3] for c in projected_err])])
            push!(soln[i]["projected_ta"],projected_ta)
            push!(soln[i]["gtda_ta"],gtda_ta)
        end
        soln[i]["inc_nodes"] = inc_nodes
    end

    pos = plot()
    for j in range(1,length(nns_to_test))
    plot!(pos,fracs,[soln[i]["true_positive_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
    plot!(pos,fracs,[soln[i]["true_positive_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    plot!(pos,fracs,[length(soln[i]["inc_nodes"])/size(X,1) for i in range(1,length(fracs))],markershape=:star,label = "included_nodes")
    title!(pos,"$(name)-$(n[1])-$(d)positive")


    neg = plot()
    for j in range(1,length(nns_to_test))
        plot!(neg,fracs,[soln[i]["true_negative_gtda"][j][2] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(neg,fracs,[soln[i]["true_negative_projected"][j][2] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")
    end
    title!(neg,"$(name)-$(n[1])-$(d)negative")

    ta = plot()
    for j in range(1,length(nns_to_test))
        plot!(ta,fracs,[soln[i]["gtda_ta"][j] for i in range(1,length(fracs))],markershape=:star,label = "gtda_$(nns_to_test[j])")
        plot!(ta,fracs,[soln[i]["projected_ta"][j] for i in range(1,length(fracs))],markershape=:circle,label = "proj_$(nns_to_test[j])")    
    end
    title!(ta,"$(name)-$(n[1])-$(d)triplet_accuracy")


    plot(pos,neg,ta,layout = (1,3),size = (1600, 500),legend=:outertopright)
    
    
    savefig("/Users/alice/Documents/Research/RPs/ablation/$(name)-$(n[1])-$(d)")


    open("$(name)-$(n[1])-$(d).json", "w") do io
        JSON.print(io, soln)
    end
end


