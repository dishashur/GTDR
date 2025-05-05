include("/Users/alice/Documents/Research/RPs/mainidea.jl")
using DelimitedFiles, Downloads


#=datasets from papers RTD_AE
- MNIST, FMNIST, 
- Mammoth(3D->2D)
- coil20
 - scRNA mice  
 - scRNA melanoma
 -spheres 3D->3D
 - circle, 2 clusters, 3 clusters, random (2D->2D distortions)
 - meridians 3D -> 2D

PHATE S7, S8, S9
size of n, vs time taken, cluster coherence if present
1. scRNAseq 
2. mass cytometry
3. gut microbiome
4. Supplemental figure s3 in Phate paper
5. 1.3 million mouse brain cell datase 
6. facebook data
7. chromatin structure

things to show about structure - 
1. simple non-biological datasets
2. structures in high dimensional dataset
3. structures in connectivity dataset

tsne datasets 
the MNIST data set, the Olivetti faces data set, (3) the COIL-20 data set, (4) the word-features data set, and (5) the Netflix data set.

=#

function runthis(X, labels, name;min_group_size = 3, max_split_size = 30,min_component_group = 1, overlap = 0.1,tym1 = 0)
    begining = time()
    Random.seed!(123)
    Xnoisy,G,lens = topological_lens(X,10,dims = 30)
    tym1 = time() - begining
    @show tym1
    gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
            max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
            split_thd=0,merge_thd = 0.01,labels = labels);
    
    g = SimpleGraph(gtdaobj.G_reeb)
    pos = spring_layout(g) 
    xy = hcat(pos...)

    combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
    "rn"=>gtdaobj.node2reeb, "xy"=>xy, "time"=>round(timereeb,sigdigits=4))
    f = open("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).json","w")
    JSON.print(f,combo_dict) 
    close(f)

    return gtdaobj, combo_dict
end

##EXPERIMENT 9 TF edited file

indices = data1["matrix/indices"];
indptr = data1["matrix/indptr"];
vals = data1["matrix/data"];
n_genes, n_cells = data1["matrix/shape"];

#getting i,j from colptr
jind = []
for j in range(1,n_cells)
    [push!(jind,j) for _ in range(indptr[j],indptr[j+1]-1)]
end
X = Matrix(sparse(indices .+ 1, jind, vals, n_genes,n_cells))
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,1,dims = 3);
tym1 = time() - begining


##EXPERIMENT 8 meridians

##EXPERIMENT 7 Mammoth
name = "Mammoth"
using JSON
data = JSON.parsefile("/Users/alice/Documents/Research/RPs/data/mammoth/mammoth_3d.json")
label_data = JSON.parsefile("/Users/alice/Documents/Research/RPs/data/mammoth/mammoth_10k_encoded.json")
labels = []
[append!(labels,i) for (i,k) in enumerate(label_data["labelOffsets"]) for kk in range(1,k)]

X = Matrix(hcat(data...)');
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,1,dims = 3);
tym1 = time() - begining

min_group_size = 1
max_split_size = 5
overlap = 0.3
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=1,
        max_split_size=5,min_component_group=5,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);
gtdaobj.G_reeb

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node 
#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
nodecolors = distinguishable_colors(length(unique(labels)) + 1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5,actuallabels = [])
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))",legend=:outertopleft)


##EXPERIMENT 6 - scRNA melanoma
name = "scRNA_melanoma"
data = readdlm("/Users/alice/Documents/Research/RPs/data/RNAseq_melanoma/GSE72056_melanoma_single_cell_revised_v2.txt")
temp = readdlm("data/RNAseq_melanoma/GSE72056_series_matrix.txt")
labels = ones(size(data,2)-1) #0 is unresolved
[labels[i-1] = 8 for i in range(2, size(data,2)) if data[3,i]==2] #malignant
[labels[i-1] = data[4,i] + 1 for i in range(2, size(data,2)) if data[3,i]==1] #non-malignant type
X = Matrix(data[5:end,2:end]')
labelnames = ["unresolved","T","B","Macro","Endo","CAF","NK","Malignant"]
begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,5,dims = 30);
tym1 = time() - begining

min_group_size = 1
max_split_size = 10
overlap = 0.3
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=1,
        max_split_size=10,min_component_group=5,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);
gtdaobj.G_reeb


g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node 
#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
nodecolors = distinguishable_colors(length(unique(labels)) + 1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,nodecolors,showlabels,lw = 0.5,actuallabels = labelnames)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))_5",legend=:outertopleft)


gactual = SimpleGraph(G)
pos = spring_layout(gactual)
xyactual = hcat(pos...)
pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")

tot = plot(p,pactual,layout = (1,2),size = (1200, 500),markerstrokewidth=0.0,titlefontsize = 10);

savefig(tot,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(overlap).png")


embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = labelnames,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")



##EXPERIMENT 4 - scRNA mice(RTD_AE) - 25392 genes times 1402 cells
name = "scRNA_mice"
X = float.(Matrix(readdlm("/Users/alice/Documents/Research/RPs/data/RNAseq_mice/DATA_MATRIX_LOG_TPM.txt")[2:end,2:end]'));
nuclei_clusters =  readdlm("/Users/alice/Documents/Research/RPs/data/RNAseq_mice/CLUSTER_AND_SUBCLUSTER_INDEX.txt")[3:end,2];
nuclei_subclusters = readdlm("/Users/alice/Documents/Research/RPs/data/RNAseq_mice/CLUSTER_AND_SUBCLUSTER_INDEX.txt")[3:end,3];
uclusters = unique(nuclei_clusters)#unique(nuclei_subclusters)
labels = zeros(size(nuclei_clusters,1)) #zeros(size(nuclei_subclusters,1))
for i in range(1,length(uclusters))
    labels[findall(x->x==uclusters[i],nuclei_clusters)] .= i
end
labels = Int64.(labels)

begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30);
tym1 = time() - begining

min_group_size = 1
max_split_size = 20
overlap = 0.3
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=1,
        max_split_size=20,min_component_group=5,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);
gtdaobj.G_reeb


g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node 
#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
nodecolors = distinguishable_colors(length(uclusters) + 1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,nodecolors,showlabels,lw = 0.5,actuallabels = uclusters)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))",legend=:outertopleft)

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = nuclei_clusters,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")

plot(t,p,layout = (1,2),size = (1200, 500),markerstrokewidth=0.0,titlefontsize = 10)

savefig("/Users/alice/Documents/Research/RPs/real/$(name)_clusters$(min_group_size)_$(max_split_size)_$(overlap).png")

#labelling according to nuclei_subclusters
uclusters = [i for i in unique(nuclei_subclusters) if occursin("DG",i)]
labels = zeros(size(nuclei_subclusters,1)) 
for i in range(1,length(uclusters))
    labels[findall(x->x==uclusters[i],nuclei_subclusters)] .= i
end
labels = Int64.(labels) .+ 1
nodecolors = distinguishable_colors(length(uclusters) + 1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node 

labelcounter = zeros(length(unique(labels)))
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.1,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,nodecolors,showlabels,lw = 0.5,actuallabels = nuclei_subclusters)
end
title!(p,"GTDA_DG_$(round(timereeb,sigdigits=4))",legend=:outertopleft)


t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = nuclei_subclusters,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")



#EXPERIMENT 3 - coil20
url = "https://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
zip_path = "data/coil-20.zip"
Downloads.download(url, zip_path)


using FileIO, ImageIO, Images
image_dir = "data/coil-20"
image_files = filter(x -> endswith(x, ".png"), readdir(image_dir))
# Load all images into an array
images = [load(joinpath(image_dir, file)) for file in image_files]
image_size = size(images[1])  # Assuming all images are the same size
image_matrix = hcat([vec(Float64.(Gray.(img))) for img in images]...)';
X = (image_matrix .- mean(image_matrix)) ./ std(image_matrix);
labels = [parse(Int64,split(split(i,"__")[1],"obj")[2]) for i in readdir(image_dir)];


begining = time()
Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30);
tym1 = time() - begining

min_group_size = 3
max_split_size = 10
overlap = 0.3
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=3,
        max_split_size=10,min_component_group=1,verbose=false,overlap = 0.3,
        split_thd=0,merge_thd = 0.01,labels = labels);

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
rc = gtdaobj.reeb2node  
p = DiffusionTools.draw_graph(gtdaobj.G_reeb,xy; linewidth=0.05,framestyle=:none, linecolor=:black,linealpha=0.6,axis_buffer=0.02,labels = "");
nodecolors = distinguishable_colors(length(unique(labels))+1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/5e4),p,nodecolors,showlabels,lw =0.1)
end
title!(p,"GTDA$(round(timereeb+tym1,sigdigits=4))_$(scomponents(gtdaobj.G_reeb).sizes)",legend=:outertopright)

gactual = SimpleGraph(G)
pos = spring_layout(gactual)
xyactual = hcat(pos...)
pactual = DiffusionTools.draw_graph(G,xyactual; linewidth=0.5,framestyle=:none,
linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "")

fig3 = plot(images[findall(i->i==3,labels)[1]],framestyle=:none,size = (200,200))
fig6 = plot(images[findall(i->i==6,labels)[1]],framestyle=:none,size = (200,200))

plot(p,pactual,fig3,fig6,layout = (2,2),size = (1200, 1200))


savefig("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(overlap).png")


## EXPERIMENT NO. 1 MNIST - download directly from url to maintain uniformity

# URLs for MNIST CSV files
name = "MNIST"
url_train = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"
file_train = "data/mnist_train.csv"

# Download the file
Downloads.download(url_train, file_train)

# Load dataset
data = readdlm(file_train, ',', header=true)

# Convert to numeric matrix
X = Float32.(data[1][:, 1:end-1])
y = Int.(data[1][:, end]) .+ 1
X_train = X[1:60000,:];
y_train = y[1:60000,:];
X_test = X[60001:end,:];
y_test = y[60001:end,:];

min_group_size = 5
max_split_size =30
min_component_group = 20
overlap = 0.1
@time mnist_reeb, mnist_dict = runthis(X_train,y_train, name,min_group_size = min_group_size, max_split_size = max_split_size, min_component_group = min_component_group, overlap = 0.01);
#mnist_dict = JSON.parsefile("/Users/alice/Documents/Research/RPs/real/$(name).json")


#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
p = DiffusionTools.draw_graph(G_reeb,xy; linewidth=0.5,framestyle=:none,
linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "");
nodecolors = distinguishable_colors(length(unique(labels)) + 1, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))
mnistlegend = ["0","1","2","3","4","5","6","7","8","9"]
for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5,actuallabels = mnistlegend)
end
title!(p,"GTDA$(round(tym1,sigdigits=4))")

savefig(p,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).png")




##EXPERIMENT NO. 2 FMNIST
name = "FMNIST"
using MLDatasets
temp = FashionMNIST(; Tx=Float32, split=:train, dir=nothing)
X = Matrix(reshape(temp.features,(28*28,60000))');
labels = temp.targets .+ 1;
fmnistlegend  = temp.metadata["class_names"]

begining = time()
Random.seed!(123)
Xnoisy,Xgraph,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining
@show tym1



min_group_size = 5
max_split_size =300
min_component_group = 5
overlap = 0.1
#@time fmnist_reeb, fmnist_dict = runthis(X,labels, name,min_group_size = min_group_size, max_split_size = max_split_size, min_component_group = min_component_group, overlap = 0.01);
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
        max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
        split_thd=0,merge_thd = 0.01,labels = labels);
G_reeb = gtdaobj.G_reeb
rc = gtdaobj.reeb2node;

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
#p = DiffusionTools.draw_graph(G_reeb,xy; linewidth=0.5,framestyle=:none,
#linecolor=:black,linealpha=0.5,axis_buffer=0.02,labels = "");

nodecolors = distinguishable_colors(length(unique(labels)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))

for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5,actuallabels = fmnistlegend)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))")



savefig(p,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).png")

combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
"rn"=>gtdaobj.node2reeb, "xy"=>xy, "time"=>round(timereeb,sigdigits=4))
f = open("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).json","w")
JSON.print(f,combo_dict) 
close(f)

embedding,timetsne,_,_ = @timed tsne(X, 2, 0, 1000, 30.0)
t = scatter(embedding[:,1],embedding[:,2],markerstrokewidth=0.5,markersize=3.0,legend=:outertopleft,color = nodecolors[labels], group = labels,
       framestyle=:none,axis_buffer=0.02,legendfontsize = 10.0, title = "tsne$(round(timetsne,sigdigits=4))")


##EXPERIMENT NO. 3 Time stamped human brain oragnoid data
name = "BrainOrganoid"
using NPZ, MatrixNetworks
data = npzread("/Users/alice/Documents/gtda/gtda_sca/humanbrain/human-409b2.data.npy");
labels = readlines("/Users/alice/Documents/gtda/gtda_sca/humanbrain/labels409b2.csv");

num_labels = zeros(length(labels))
ulabels = unique(labels)
for i in range(1,length(ulabels))
    num_labels[findall(j->j==ulabels[i],labels)] .= i
end
labels = Int.(num_labels)


begining = time()
Random.seed!(123)
Xnoisy,G,lens = topological_lens(data,10,dims = 30)
tym1 = time() - begining
@show tym1


min_group_size = 5
max_split_size =50
min_component_group = 5
overlap = 0.1
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
        max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
        split_thd=0,merge_thd = 0.01,labels = labels);
G_reeb = gtdaobj.G_reeb
rc = gtdaobj.reeb2node;

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
#p = DiffusionTools.draw_graph(G_reeb,xy; linewidth=0.5,framestyle=:none,
#linecolor=:black,linealpha=0.06,axis_buffer=0.02,labels = "");

nodecolors = distinguishable_colors(length(unique(labels)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))

for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5,actuallabels = ulabels)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))")



savefig(p,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).png")

combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
"rn"=>gtdaobj.node2reeb, "xy"=>xy, "time"=>round(timereeb,sigdigits=4))
f = open("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).json","w")
JSON.print(f,combo_dict) 
close(f)



##EXPERIMENT NO. 4 zebrafish data

name = "zebrafish"
data = npzread("/Users/alice/Documents/gtda/gtda_sca/zfish/ne_processed_data.npy");
labels = JSON.parsefile("/Users/alice/Documents/gtda/gtda_sca/zfish/ne_processed_labels.json");
times = labels["labels"];
tissues = labels["altlabels"];
utiem = unique(tissues) #tissues #times
num_labels = zeros(length(tissues)) #times
for i in range(1,length(utiem))
    num_labels[findall(j->j==utiem[i],tissues)] .= i
end
labels = Int.(num_labels)


begining = time()
Random.seed!(123)
Xnoisy,G,lens = topological_lens(data,10,dims = 30)
tym1 = time() - begining
@show tym1


min_group_size = 5
max_split_size =50
min_component_group = 5
overlap = 0.1
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
        max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
        split_thd=0,merge_thd = 0.01,labels = labels);
G_reeb = gtdaobj.G_reeb
rc = gtdaobj.reeb2node;

g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
#p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
p = DiffusionTools.draw_graph(G_reeb,xy; linewidth=0.5,framestyle=:none,
linecolor=:black,linealpha=0.2,axis_buffer=0.02,labels = "");

nodecolors = distinguishable_colors(length(unique(labels)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))

for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),p,nodecolors,showlabels,lw = 0.5,actuallabels = utissues)#utiem)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))",legend = :outertopleft)

savefig(p,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap)_organs.png")

combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
"rn"=>gtdaobj.node2reeb, "xy"=>xy, "time"=>round(timereeb,sigdigits=4))
f = open("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).json","w")
JSON.print(f,combo_dict) 
close(f)



##EXPERIMENT NO. 5 paul15
name = "MurineHematopiesis"
origdata = JSON.parsefile("/Users/alice/Documents/Research/RPs/paul15orig.json");
#GraphTDA/paul15data/data/paul15/paul15.h5
#JSON.parsefile("../GraphTDA/paul15data/paul15orig.json");

#X = npzread("/Users/alice/Documents/Research/RPs/mouse_pca.npy")
X = Matrix(hcat(origdata["X"]...)')
clusters = origdata["clusters"]
uclusters = unique(clusters)
num_labels = zeros(length(clusters)) 
for i in range(1,length(uclusters))
    num_labels[findall(j->j==uclusters[i],clusters)] .= i
end
labels = Int.(num_labels)


begining = time()
Random.seed!(123)
Xnoisy,G,lens = topological_lens(X,10,dims = 30)
tym1 = time() - begining
@show tym1

min_group_size = 1
max_split_size =15
min_component_group = 5
overlap = 0.2
gtdaobj,timereeb,_,_ = @timed GraphTDA.analyzepredictions(lens,G = G,min_group_size=min_group_size,
        max_split_size=max_split_size,min_component_group=min_component_group,verbose=false,overlap = overlap,
        split_thd=0,merge_thd = 0.01,labels = labels);
G_reeb = gtdaobj.G_reeb

rc = gtdaobj.reeb2node;
g = SimpleGraph(gtdaobj.G_reeb)
pos = spring_layout(g) 
xy = hcat(pos...)
p = plot(framestyle=:none,axis_buffer=0.02,labels = "")
#p = DiffusionTools.draw_graph(G_reeb,xy; linewidth=0.5,framestyle=:none,
#linecolor=:black,linealpha=0.3,axis_buffer=0.02,labels = "");

nodecolors = distinguishable_colors(length(unique(labels)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
labelcounter = zeros(length(unique(labels)))

for i in range(1,size(xy,1))
    dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
    labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
    showlabels = [k for k in range(1,length(labelcounter)) if labelcounter[k]==1]
    p = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e4),p,nodecolors,showlabels,lw = 0.5,actuallabels = uclusters)
end
title!(p,"GTDA$(round(timereeb,sigdigits=4))",legend = :outertopleft)

savefig(p,"/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).png")
combo_dict = Dict("G_reeb"=>findnz(gtdaobj.G_reeb)[1:2],"rc"=>gtdaobj.reeb2node,
"rn"=>gtdaobj.node2reeb, "xy"=>xy, "time"=>round(timereeb,sigdigits=4))
f = open("/Users/alice/Documents/Research/RPs/real/$(name)_$(min_group_size)_$(max_split_size)_$(min_component_group)_$(overlap).json","w")
JSON.print(f,combo_dict) 
close(f)
