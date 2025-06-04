
#holds the function to define all the real datasets
"""datasets = ["mnist","fmnist","coil20",
"Paga-paul15","Paga/NEs-zfish","NEs-human409b2","rtdae-melanoma", //NEs-hydra","NEs-AMC",
"localmap-usps","localmap-20ng",//"localmap-kang","localmap-seurat","localmap-humancortex","localmap-cbmc",
"scgae-hspc","scgae-pancreas",
"googlenews-word2vec-umap","wikiword-largeviz","csauthor-largeviz",
"transformer-data-activation"]

"Phate-RNA","Phate-cytometry", "Phate-micorbiome", Supplemental figure s3 in Phate paper,
1.3 million mouse brain cell datase, facebook data, chromatin structure]

"""

using Downloads, ZipFile, NPZ, FileIO, ImageIO, Images, MatrixNetworks, Statistics

function get_mnist()
    name = "mnist"
    url_train = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"
    file_train = "real_data/mnist_train.csv"

    Downloads.download(url_train, file_train)

    data = readdlm(file_train, ',', header=true)

    X = Float32.(data[1][:, 1:end-1])
    y = Int.(data[1][:, end]) .+ 1
    X_train = X[1:60000,:];
    y_train = y[1:60000,:];
    X_test = X[60001:end,:];
    y_test = y[60001:end,:];
    return X_train, y_train
end

function get_fmnist()
    @info "Fashon-MNIST"
    file_train = "real_data/fmnist_train.csv"
    #Downloads.download("https://github.com/fpleoni/fashion_mnist/raw/refs/heads/master/fashion-mnist_train.csv",file_train)
    
    data = readdlm(file_train, ',', header=true) 
    
   
    y = Int.(data[1][:, 1]) .+ 1
    X = Float32.(data[1][:,2:end])
    return X,y
end

function get_coil20()
    #url = "https://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    #zip_path = "real_data/coil-20.zip"
    #Downloads.download(url, zip_path)

    @info "COIL-20"
    image_dir = "real_data/coil20"
    image_files = filter(x -> endswith(x, ".png"), readdir(image_dir))
    # Load all images into an array
    images = [load(joinpath(image_dir, file)) for file in image_files]
    image_size = size(images[1])  # Assuming all images are the same size
    image_matrix = hcat([vec(Float64.(Gray.(img))) for img in images]...)';
    X = (image_matrix .- mean(image_matrix)) ./ std(image_matrix);
    labels = [parse(Int64,split(split(i,"__")[1],"obj")[2]) for i in readdir(image_dir)];

    return X, labels

end


function get_melanoma()
    @info "metastatic melanoma"

    data = readdlm("real_data/GSE72056_melanoma_single_cell_revised_v2.txt")
    temp = readdlm("real_data/GSE72056_series_matrix.txt")
    labels = ones(size(data,2)-1) #0 is unresolved
    [labels[i-1] = 8 for i in range(2, size(data,2)) if data[3,i]==2] #malignant
    [labels[i-1] = data[4,i] + 1 for i in range(2, size(data,2)) if data[3,i]==1] #non-malignant type
    X = Matrix(data[5:end,2:end]')
    return X, labels
end

function get_humandevelopmental()
    @info "human brain organoid developmental data"

    data = npzread("real_data/human-409b2.data.npy");
    labels = readlines("real_data/labels409b2.csv");

    num_labels = zeros(length(labels))
    ulabels = unique(labels)
    for i in range(1,length(ulabels))
        num_labels[findall(j->j==ulabels[i],labels)] .= i
    end
    actual_labels = Int.(num_labels)
    return data,Dict("actual_labels"=>actual_labels,"true_labels"=>labels)
end

function get_zfishembryo()
    @info "zebrafish embryo developmental data"
    data = npzread("real_data/zfish_data.npy");
    labels = JSON.parsefile("real_data/zfish_labels.json");
    times = labels["labels"];
    utiem = unique(times) 
    num_labels = zeros(length(times)) 
    for i in range(1,length(utiem))
        num_labels[findall(j->j==utiem[i],times)] .= i
    end
    time_labels = Int.(num_labels)

    tissues = labels["altlabels"];
    utiem = unique(tissues) 
    num_labels = zeros(length(tissues)) 
    for i in range(1,length(utiem))
        num_labels[findall(j->j==utiem[i],tissues)] .= i
    end
    tissue_labels = Int.(num_labels)

    return data, Dict("true_labels"=>times, "actual_labels"=>time_labels)

end

function get_mousestem()
    @info "MARS sequencing of mouse stem cell differentiation"
    #this is the one used in PAGA from paul15
    origdata = JSON.parsefile("real_data/paul15orig.json");
    X = Matrix(hcat(origdata["X"]...)')
    clusters = origdata["clusters"]
    uclusters = unique(clusters)
    num_labels = zeros(length(clusters)) 
    for i in range(1,length(uclusters))
        num_labels[findall(j->j==uclusters[i],clusters)] .= i
    end
    labels = Int.(num_labels)

    return X, Dict("actual_labels"=>labels,"true_labels"=>clusters)
end

############################################# the following are ot compleetd yet ##########
function get_20NG()
    #from LocalMAP/data/
    @info "nodes are articles colours are news groups"
    url = "https://github.com/williamsyy/LocalMAP/raw/refs/heads/experiments/data/20NG.npy"
    zip_path = "real_data/20NG_images.npz.zip"
    Downloads.download(url, zip_path)
    ZipFile.Reader(zip_path) do archive
        for file in archive.files
            @info "Extracting: $(file.name)"
            npzwrite(file.name, read(file))
        end
    end
    X = npzread("real_data/20ng_images.npz")
    url = "https://github.com/williamsyy/LocalMAP/raw/refs/heads/experiments/data/20NG_labels.npy"
    zip_path = "real_data/20NG_labels.npz.zip"
    Downloads.download(url, zip_path)
    ZipFile.Reader(zip_path) do archive
        for file in archive.files
            @info "Extracting: $(file.name)"
            npzrite(file.name, read(file))
        end
    end
    y = npzread("real_data/20ng_labels.npz") 

    return X,y
end

function get_usps()
    #from LocalMAP/data/
    @info "USPS"
    url = "https://github.com/williamsyy/LocalMAP/raw/refs/heads/experiments/data/USPS.npy"
    zip_path = "real_data/usps_images.npz.zip"
    Downloads.download(url, zip_path)
    ZipFile.Reader(zip_path) do archive
        for file in archive.files
            @info "Extracting: $(file.name)"
            npzwrite(file.name, read(file))
        end
    end
    X = npzread("real_data/usps_images.npz")
    url = "https://github.com/williamsyy/LocalMAP/raw/refs/heads/experiments/data/USPS_labels.npy"

    zip_path = "real_data/usps_labels.npz.zip"
    Downloads.download(url, zip_path)
    ZipFile.Reader(zip_path) do archive
        for file in archive.files
            @info "Extracting: $(file.name)"
            npzrite(file.name, read(file))
        end
    end
    y = npzread("real_data/usps_labels.npz") 
    @info "done"
    return X,y
end


function get_hydra()
    @info "cell cluster data"
end

function get_mousecortex()
    @info "adult mouse cortex data"
    #this is the one  used in the NE spectrum paper
end


function get_glove(dims = 100)
    @info "$(dims) dimensional word embeddings from GloVe from among [50,100,200,300]"
    @info "Needs to be completed"
    #=wget https://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip 
    =#
    embeddings = Dict{String, Vector{Float32}}()
    filepath = "/u/subspace_s4/dshur/dishas_data/glove/glove.6B.$(dims)d.txt"
    open(filepath, "r") do f
        for line in eachline(f)
            parts = split(line)
            word = parts[1]
            vec = Float32.(parse.(Float64, parts[2:end]))
            embeddings[word] = vec
        end
    end
    return embeddings

    #need to return matrix embeddings and labels
end

function get_wikidoc()
    @warn "Not implemented"
end


function get_googlenewsvec()
     @warn "Not implemented"
end



function get_csuathor()
     @warn "Not implemented"
end

function get_kang()
    @warn "Not implemented"
end

function get_seurat()
     @warn "Not implemented"
end

function get_humancortex()
     @warn "Not implemented"
end

function get_cbmc()
     @warn "Not implemented"
end

function get_flowcyto()
     @warn "Not implemented"
end




