#this script is for reading the coordinates and actuall generating the plotsg = gtdaobj.G_reeb
using JSON, SparseArrays
include("utils.jl")


#step 1 read all the layout keys from different layouts of gtda
gtdarecord = JSON.parsefile("stored_embeddings/rw_norestart_5_500_1_0.3.json")
G_reeb = sparse(gtdarecord["G_reeb"][1],gtdarecord["G_reeb"][2],ones(length(gtdarecord["G_reeb"][2])))
rc = gtdarecord["rc"]
rn = gtdarecord["rn"]
gtdatime = gtdarecord["time"] 
coords_dict = gtdarecord["all_coords"]
ncols = length(coords_dict)+1
nrows = 1
fig = CairoMakie.Figure(resolution = (4400, 800))
layout_keys = collect(keys(coords_dict))

ax = Axis(fig[1])
orig = scatter(ax,X[:,1],X[:,2],color = finalcolors,markerstrokewidth=0.5,markersize=3.0,
framestyle=:none,axis_buffer=0.02,title = "$(n)_$(d)",label = "",group = labels);

for (i, key) in enumerate(layout_keys)
    ax = Axis(fig[i+1])
    layout_coords = coords_dict[key]

    # Convert coordinates to format suitable for scatter
    xy = coords_dict[key]
   
    # Plot nodes
    linesx, linesy = draw_graph(G_reeb,xy)
    preeb = lines!(ax, linesx, linesy, linewidth=0.6, framestyle=:none, linecolor=:black,linealpha=0.6,
    axis_buffer=0.02,labels = "");
    reebcolors = [mean(labels[i]) for i in rc]
    finalreebcolors = [nodecolors[Int(round(i))] for i in reebcolors] 
    preeb = scatter!(preeb,xy[:,1],xy[:,2],color = finalreebcolors,label="",markerstrokewidth=0.1,markersize = 4)
    ax.title = key
    hidespines!(ax)
    hidedecorations!(ax)
end
fig[1, :] = Label(fig, "GTDR for mnist", fontsize=24)
#display(fig)34
#save("stored_embeddings/$(name).png",fig)
