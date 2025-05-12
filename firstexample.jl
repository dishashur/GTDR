include("GraphTDA.jl")
include("utils.jl")
include("centralcode.jl")
include("candidates.jl")


"""script for generating the first example"""

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
nodecolors = cgrad(:ice, Int(round(maximum(labels)) + 1), categorical = true)
finalcolors = [nodecolors[Int(round(i))] for i in labels] 

begining = time()
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining

gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=5,
        max_split_size=500,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);

coords_dict =  getlayout(gtdaobj.G_reeb) #dictionary of 5 layout coords


combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
    "rn"=>gtdaobj.node2reeb, "all_coords"=>coords_dict, "time"=>round((timereeb+tym1),sigdigits=4),
    "params"=>Dict("min_group_size"=>min_group_size,"max_split_size"=>max_split_size,
    "min_component_group"=>min_component_group,"overlap"=>overlap))
f = open("/stored_embeddings/$(dataname).json","w")
JSON.print(f,combo_dict) 
close(f)

