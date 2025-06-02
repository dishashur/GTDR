

#program to read feature matrices and save their knns
using Distributed
addprocs(8)  # or however many cores you want

@everywhere begin
    using SparseArrays
    using NearestNeighbors
    include("utils_parallel.jl")  # make sure `make_graph` is defined
end

@everywhere function build_graph_entry(X, num_nn)
    G_orig = make_graph(X, num_nn = num_nn)
    i, j = findnz(G_orig)[1:2]
    return num_nn => Dict("i" => i, "j" => j)
end

# Inputs (must be available on all workers)
dataname = "mnist"
X = hcat(gtdarecord["orig_data"]...)
labels = gtdarecord["orig_labels"][1]

# Broadcast X to workers (optional but efficient for large X)
@everywhere const SHARED_X = $X

# Run parallel job
nns_to_test = [6, 10, 15, 20]
results = pmap(num_nn -> build_graph_entry(SHARED_X, num_nn), nns_to_test)

# Assemble final dictionary
G_dicts = Dict(results)


f = open("stored_knns/$(dataname).json","w")
JSON.print(f,G_dicts) 
close(f)


#ENV["PYTHON"] = "/homes/dshur/miniconda3/envs/topo_red/bin/python"   # Replace with the environment containing the dimension reduction stuff
#using Pkg
#Pkg.build("PyCall")

using Distributed
addprocs(4)
@everywhere include("utils_parallel.jl")

#read the stored embedings -- DO NOT run them again, takes a lot of time
#1. read GTDA for each of [kk and spectral layout] and each of the other 8 layouts 
#2. compare them all of original and stor errors and accuracies in the dictionary
#3. Hence the dictionary will have 10 keys (2 for gtdr and 8 others), 
#each with 2 further keys, 

dataname  = "mnist"
gtdarecord = JSON.parsefile("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(dataname).json");
G_reeb = sparse(gtdarecord["G_reeb"][1],gtdarecord["G_reeb"][2],ones(length(gtdarecord["G_reeb"][2])));
rc = gtdarecord["rc"];
rns = gtdarecord["rn"];
#for name in ["spring","kamada_kawai"]
X = hcat(gtdarecord["orig_data"]...);
labels = gtdarecord["orig_labels"][1];

all_other_methods = load_object("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(dataname).jld2");
# Load precomputed G_dicts from JSON
G_dicts = JSON.parsefile("/p/mnt/homes/dshur/topo_dim_red/stored_knns/mnist.json")

@show nworkers()

all_errors = errors_with_G(G_dicts,dataname;nns_to_test = [6,10,15,20],G_cand_given = G_reeb,cand_embs = all_other_methods, rc = rc,rns=rns)


rmprocs(workers())
@show nworkers()






