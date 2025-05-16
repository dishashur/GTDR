include("GraphTDA.jl")
include("../utils.jl") #getlayout
include("centralcode.jl") 
include("candidates.jl") #getemall
include("codes_synthetic.jl")
include("codes_real.jl")

"""script for generating the first example"""

function firstexample_part1()
    dataname = "rw_norestart"
    Random.seed!(123)
    n = 3000
    d = 100
    points = rw_norestart(n,d);
    X = Matrix(hcat(points...)');
    X = X .- mean(X);
    X = X ./ std(X);
    labels = [norm(X[i,:]) for i in range(1,n)];
    f = open("synthetic_data/$(dataname).json","w")
    JSON.print(f,Dict("data"=>X,"labels"=>labels)) 
    close(f)


    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
            max_split_size=500,min_component_group=1,verbose=false,overlap = 0.3,
            split_thd=0,merge_thd = 0.01,labels = labels);

    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords


    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
        "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
        "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
        "min_component_group"=>min_component_group,"overlap"=>overlap))
    f = open("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(dataname).json","w")
    JSON.print(f,combo_dict) 
    close(f)

    other_methods = getemall(X,dataname,num_nn = 10)
    println("done")

end


function firstexample_part2()
    dataname = "mnist"
    _ , _, X, labels, _, _ = get_mnist()

    println("got data")

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining

    println("got lens")

    min_group_size = 5
    max_split_size =30
    min_component_group = 20
    overlap = 0.1
    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
            max_split_size=max_split_size,min_component_group=20,verbose=false,overlap = 0.1,
            split_thd=0,merge_thd = 0.01,labels = labels);

    println("going to get diffrent layouts")
    coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords

    println("done w layouts")

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

end


firstexample_part2()


#firstexample_part2()