include("codes_synthetic.jl")
include("codes_real.jl")
include("utils.jl")

"""
this script is for experiments examining the differnet aspects of dimension reduction
shape distortion, local structure, global structure, maintainence of structure
for downstream in inference, 6 points of a dimension reduction methods
"""


#1a. datasets for 2D->2D(shape distortion) - galaxy(2D), clusters 2D, shape 2D
dataname = "galaxies"
X, labels = galaxies();
X = X .- mean(X)
X = X ./ std(X)
println("got data")
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,3,dims = 3)
tym1 = time() - begining
println("got lens")
min_group_size = 10
max_split_size =70
min_component_group = 1
overlap = 0.15
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=10,
        max_split_size=70,min_component_group=1,verbose=false,overlap = 0.15,
        split_thd=0,merge_thd = 0.01,labels = labels);

println("going to get diffrent layouts")
coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords


combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap),"orig_data"=>X,"orig_labels"=>labels)
f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(dataname).json","w")
JSON.print(f,combo_dict)
close(f)

println("doing different methods")

getemall(X,dataname,num_nn = 10)
println("done")

#need error for all of them




#2a. Datasets for 3D->2D (mainatin shape) -Mammoth, Meridians, Nested spheres, RTD spheres



#4a. Path consistent (PHATE tree, NE)

X,labels =  synthetic_tree()
#n_dim = 200,n_branch = 10,branch_length = 300,rand_multiplier = 2,seed=37,sigma = 5
nodecolors = distinguishable_colors(n_branch, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,25,dims = 30);
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=50,min_component_group=1,verbose=false,overlap = 0.2,
        split_thd=0,merge_thd = 0.01,labels = labels);

layouts = getlayout(gtdaobj.G_reeb)
labelcounter = zeros(length(unique(labels)))
for (lo,v) in layouts
    xy = v
    for i in range(1,size(xy,1))
        dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
        labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
        showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
        p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw =0.1)
    end
    title!(p,"GTDA+$(lo.__name__)")    
end

nns_to_test = [6, 10, 15, 20]
@time true_positive,true_negative = errors_and_accuracies(embedding, gtdaobj.G_reeb, gtdaobj.reeb2node, gtdaobj.node2reeb, X, nns_to_test = nns_to_test)

$(round(timereeb+tym1,sigdigits=4))
$(scomponents(gtdaobj.G_reeb).sizes)