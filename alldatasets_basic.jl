include("GraphTDA.jl")
include("../utils.jl") #getlayout
include("centralcode.jl") 
include("candidates.jl") #getemall
include("codes_synthetic.jl") 
include("codes_real.jl")
include("../plottingGTDR.jl")

"""script for generating pictures using GTDR and all other methods"""

#make a parallel version of this so that for each data picture can be genrated in parallel
#for this basic expt we only need the representation pictures

real_data = Dict("mnist"=>[5,30,20,0.1],
"fmnist" => [5,300,5,0.1],
"coil20"=>[3,10,1,0.3],
"melanoma"=>[1,10,5,0.3],
"humandevelopmental"=> [5,50,5,0.1],
"zfishembryo"=>[5,50,5,0.1],
"mousestem"=> [1,15,5,0.2],
#"20NG"=>[],
#"usps"=>[]
)



for (name,params) in real_data
    @info name
    func_name = Symbol("get_", name)
    func = getfield(Main, func_name)
    X,labels = func()

    println("got data")

    begining = time()
    Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 20)
    tym1 = time() - begining
    
    @show size(Xgraph)
    @show size(lens)
    println("got lens")
    
    min_group_size = params[1]
    max_split_size = params[2]
    min_component_group = params[3]
    overlap = params[4]  


    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
            max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
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
     
    println("now plotting")
    func_name = Symbol("plot_", name)
    func = getfield(Main, func_name)
    func()
    
    println("doing different methods")
    getemall(X,dataname,num_nn = 10)
    println("done")

end

#synthetic_data = []

#for name in synthetic_data
#end