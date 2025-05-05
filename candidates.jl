"""
this script defines all the candidate embedding techniques which will be accessed for 
different experiments like ablation, aspects of dimensiomnality reduction, embedding errors,
"""
using PyCall

# 1. UMAP
umap = pyimport("umap").UMAP

# 2. PCA
pca = pyimport("sklearn.decomposition").PCA

# 3. t-SNE
tsne = pyimport("sklearn.manifold").TSNE

# 4. LargeVis (often accessed via `openTSNE` or standalone repo)
# Might require pyimport("largevis") or use RPyCall to call R LargeVis
pyimport("LargeVis")

# 5. PaCMAP
pacmap = pyimport("pacmap").PaCMAP

# 6. LocalMAP (less standard; check if it's installed as "localmap")
localmap = pyimport("pacmap").LocalMAP 

# 7. TriMAP
trimap = pyimport("trimap").TRIMAP

# 8. scGAE (typically part of deep learning libraries; might require keras/torch)
# scgae = pyimport("scgae")  # or define custom model loading

# 9. RTD-AE (same — may be in torch-based repo)
# rtd_ae = pyimport("rtd_ae") 

# 10. iVis
#ivis = pyimport("ivis").IVIS: issues with tensorflow dependency


# 11. PHATE
phate = pyimport("phate").PHATE

# 12. PAGA (from scanpy)
paga = pyimport("scanpy").tools.paga

# 13. MDS
mds = pyimport("sklearn.manifold").MDS


