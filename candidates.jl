"""
this script defines all the candidate embedding techniques which will be accessed for 
different experiments like ablation, aspects of dimensiomnality reduction, embedding errors,
"""
#ENV["PYTHON"] = "/homes/dshur/miniconda3/envs/topo_red/bin/python"   # Replace with the environment containing the dimension reduction stuff
#using Pkg
#Pkg.build("PyCall")
using PyCall, NPZ

#all of them together - embeddings once produced are stored in stored_embeddings/method/datasetname.json
function getemall(X,dataname;num_nn = 3,n_comp = 2)
    candyd8 = ["umap", "pca", "tsne", "pacmap", "localmap", "trimap", "mds", "paga"] 
    allofem = Dict(k=>[] for k in candyd8)
    for name in candyd8
        println(name)
        if name =="umap"
            emb = get_umap_coordinates(X;num_nn = num_nn,n_comp = n_comp)
        elseif name == "pca"
            emb = get_pca_coordinates(X; n_comp=n_comp)
        elseif name == "tsne"
            emb = get_tsne_coordinates(X; n_comp=n_comp, perplexity=num_nn*3)
        elseif name == "pacmap"
            emb = get_pacmap_coordinates(X;num_nn = num_nn,n_comp=n_comp)
        elseif name == "localmap"
            emb = get_localmap_coordinates(X;num_nn = num_nn,n_comp=n_comp)
        elseif name == "trimap"
            emb = get_trimap_coordinates(X;num_nn = num_nn,n_comp=n_comp)
        elseif name == "mds"
            emb = get_mds_coordinates(X;n_comp=n_comp)
        else
            emb = get_paga_coordinates(X;n_comp = n_comp,num_nn = num_nn)
        end

        if !isdir("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name)_emb")
            mkdir("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name)_emb")
        end
        npzwrite("/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(name)_emb/$(dataname).npy",emb)
        push!(allofem["$(name)"],Dict("$(dataname)"=>emb))
    end
    @save "/p/mnt/homes/dshur/topo_dim_red/stored_embeddings/$(dataname).jld2" allofem
    return allofem
end

# 1. UMAP
function get_umap_coordinates(X;num_nn = 3,n_comp = 2)
    umap = pyimport("umap").UMAP
    model = umap(n_neighbors=num_nn, n_components=n_comp)
    return model.fit_transform(X)
end

# 2. PCA
function get_pca_coordinates(X; n_comp=2)
    pca = pyimport("sklearn.decomposition").PCA
    model = pca(n_components=n_comp)
    return model.fit_transform(X)
end

# 3. t-SNE
function get_tsne_coordinates(X; n_comp=2, perplexity=30.0)
    @assert n_comp < 4
    @assert perplexity < size(X,1)
    tsne = pyimport("sklearn.manifold").TSNE
    model = tsne(n_components=n_comp, perplexity=perplexity)
    return model.fit_transform(X)
end

# 4. PaCMAP
function get_pacmap_coordinates(X;num_nn = 3,n_comp=2)
    pacmap = pyimport("pacmap").PaCMAP
    n_neighbors = min(n_neighbors, size(X, 1) - 1)
    model = pacmap(n_neighbors=num_nn, n_components=n_comp)
    return model.fit_transform(X)
end


# 5. LocalMAP 
function get_localmap_coordinates(X;num_nn = 3,n_comp=2)
    pacmap = pyimport("pacmap").LocalMAP 
    n_neighbors = min(n_neighbors, size(X, 1) - 1)
    model = pacmap(n_neighbors=num_nn, n_components=n_comp)
    return model.fit_transform(X)
end

# 6. TriMAP
function get_trimap_coordinates(X;num_nn = 3,n_comp=2)
    trimap = pyimport("trimap").TRIMAP
    num_nn = min(num_nn, size(X, 1) - 1)
    model = trimap(n_inliers = num_nn,n_dims=n_comp)
    return model.fit_transform(X)
end

# 7. MDS
function get_mds_coordinates(X;n_comp=2)
    mds = pyimport("sklearn.manifold").MDS
    model = mds(n_components = n_comp)
    return model.fit_transform(X)
end

# 8. PAGA (from scanpy)
function get_paga_coordinates(X;n_comp = 5,num_nn = 3)
    #paga = pyimport("scanpy").tools.paga
    n_comp = min(n_comp,size(X,1))-1
    sc = pyimport("scanpy")
    anndata = pyimport("anndata")
    adata = anndata.AnnData(X)
    sc.pp.pca(adata, n_comps=n_comp)
    sc.pp.neighbors(adata, n_neighbors=num_nn, use_rep="X_pca")
    sc.tl.louvain(adata)
    sc.tl.paga(adata, groups="louvain")
    paga_graph = adata.uns["paga"]
    sc.pl.paga(adata, show=false)
    sc.tl.umap(adata, init_pos="paga")
    return adata.obsm.__getitem__("X_umap"), paga_graph
end

# 9. RTD-AE 
# rtd_ae = pyimport("rtd_ae") 

# 10. scGAE 
# scgae = pyimport("scgae")  # or define custom model loading

 # 10. LargeVis (often accessed via `openTSNE` or standalone repo)
# Might require pyimport("largevis") or use RPyCall to call R LargeVis
#function get_largevis_coordinates(X;num_nn = 3)
#    lv = pyimport("LargeVis").LargeVis
 
#end

# 11. iVis
#ivis = pyimport("ivis").IVIS
#issues with tensorflow dependency

# 12. PHATE
#phate = pyimport("phate").PHATE
#pandas dependency could bot be resolved

