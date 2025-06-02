#this script is for reading the coordinates and actuall generating the plotsg = gtdaobj.G_reeb
using JSON, SparseArrays, JLD2, Statistics
include("utils.jl")
using PyPlot






function firstexample_part1()
    name = "rw_norestart"
    gtdarecord = JSON.parsefile("/Users/alice/Documents/Research/RPs/GTDR/stored_embeddings/rw_norestart_5_500_1_0.3.json")
    G_reeb = sparse(gtdarecord["G_reeb"][1],gtdarecord["G_reeb"][2],ones(length(gtdarecord["G_reeb"][2])))
    rc = gtdarecord["rc"]
    rn = gtdarecord["rn"]
    gtdatime = gtdarecord["time"] 
    coords_dict = gtdarecord["all_coords"]
    #for plotting do occursin(name,k) for k in keys(coords_dict) 
    #for name in ["spectral","random","spring","kamada_kawai","planar"]

    original_data = JSON.parsefile("/Users/alice/Documents/Research/RPs/GTDR/synthetic_data/rw_norestart.json")
    X = hcat(original_data["data"]...)
    labels = original_data["labels"]
    min_c, max_c = minimum(labels), maximum(labels)
    normalized_colors = [(c - min_c) / (max_c - min_c) for c in labels]
    cmap = PyPlot.cm.get_cmap("viridis")  # or "plasma", "Blues", etc.
    nodecolors = [cmap(c) for c in normalized_colors]

    ncols = length(coords_dict)+1
    nrows = 1 
    fig, axs = subplots(nrows, ncols, figsize=(60, 10))
    layout_keys = collect(keys(coords_dict))
    n,d = size(X)
    axs[1].scatter(X[:,1],X[:,2],c = nodecolors,s=9,edgecolors="black",linewidths=0.5,label = "");
    axs[1].set_title("Original $(n)_$(d)")
    axs[1].axis("off")
    axs[1].margins(0.02)

    for (i, key) in enumerate(layout_keys)
        ax = axs[i+1]
        # Convert coordinates to format suitable for scatter
        xy = hcat(hcat(coords_dict[key]...)...)
        linesx, linesy = draw_graph_segments(G_reeb,xy)
        for (x, y) in zip(linesx, linesy)
            ax.plot(x, y, color="black", linewidth=1.6, alpha=1.0, zorder=1)
        end
        reebcolors = [mean(labels[i]) for i in rc]
        min_c, max_c = minimum(reebcolors), maximum(reebcolors)
        normalized_colors = [(c - min_c) / (max_c - min_c) for c in reebcolors]
        
        finalreebcolors = [cmap(c) for c in normalized_colors]
        
        ax.scatter(xy[:,1], xy[:,2], c=finalreebcolors, s=500, edgecolors="black", linewidths=0.5, zorder = 2)
        ax.set_title("GTDR-$(match(r" (\w+)_layout",key).captures[1])")
        ax.axis("off")
    end
    tight_layout()
    fig.savefig("draft2/$(name).png")
    time_taken = Dict()
    push!(time_taken,"gtdr"=>gtdarecord["time"])

    allothermethods = load_object("/Users/alice/Documents/Research/RPs/GTDR/stored_embeddings/rw_norestart.jld2")
    ncols = div(length(allothermethods),2)
    nrows = 2
    fig, axs = subplots(nrows, ncols, figsize=(40, 10))
    layout_keys = collect(keys(coords_dict))

    for (i,k) in enumerate(keys(allothermethods))
        if i<5
            ax = axs[1,i]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        else
            ax = axs[2,i-4]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        end
    end 

    fig.suptitle("$(name)-other methods", fontsize=20)
    tight_layout()
    fig.savefig("draft2/$(name)-other_methods.png")
end


#for plotting gtdr do not only consider spring, spectral and kk
function firstexamplepart2()
    name = "mnist"
    gtdarecord = JSON.parsefile("stored_embeddings/mnist.json")
    G_reeb = sparse(gtdarecord["G_reeb"][1],gtdarecord["G_reeb"][2],ones(length(gtdarecord["G_reeb"][2])))
    rc = gtdarecord["rc"]
    rn = gtdarecord["rn"]
    gtdatime = gtdarecord["time"]
    coords_dict = gtdarecord["all_coords"]
    #for name in ["spectral","spring","kamada_kawai"]
    X = hcat(gtdarecord["orig_data"]...)
    labels = gtdarecord["orig_labels"][1]

    labels_unique = unique(labels)
    cmap = PyPlot.cm.get_cmap("tab20")  # 20 colors
    color_map = Dict(lbl => cmap(mod1(i, 20) - 1) for (i, lbl) in enumerate(labels_unique))
    nodecolors = [color_map[lbl] for lbl in labels]
    labelcounter = zeros(length(labels_unique))

    println("starting to plot")
    ncols = length(coords_dict)
    nrows = 1
    fig, axs = subplots(nrows, ncols, figsize=(35, 10))
    layout_keys = collect(keys(coords_dict))
    n,d = size(X)
    mnistlegend = ["0","1","2","3","4","5","6","7","8","9"]
    for (i, key) in enumerate(layout_keys)
        ax = axs[i]
        xy = hcat(hcat(coords_dict[key]...)...)
        linesx, linesy = draw_graph_segments(G_reeb,xy)
        for (x, y) in zip(linesx, linesy)
            ax.plot(x, y, color="black", linewidth=0.1, alpha=0.1, zorder=1)
        end
        seen = Set{Int}()
        for i in range(1,size(xy,1))
            dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
            distkey = [Int(k) for (k,v) in dist_dict]
            #labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
            showlabels = [k for k in distkey if !(k in seen)]
            union!(seen, distkey)
            ax = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),ax,color_map,showlabels,lw = 0.5,actuallabels = mnistlegend)
        end
        ax.set_title("GTDR-$(match(r" (\w+)_layout",key).captures[1])")
        ax.axis("off")
        for lbl in labels_unique
        ax.scatter([], [], c=[color_map[lbl]], label=mnistlegend[lbl])
        end
        ax.legend()
        println("$(key) done")
    end
    tight_layout()
    fig.savefig("draft2/$(name)2.png")
    time_taken = Dict()
    push!(time_taken,"gtdr"=>gtdarecord["time"])



    allothermethods = load_object("/Users/alice/Documents/Research/RPs/GTDR/stored_embeddings/mnist.jld2")
    ncols = div(length(allothermethods),2)
    nrows = 2
    fig, axs = subplots(nrows, ncols, figsize=(40, 10))

    for (i,k) in enumerate(keys(allothermethods))
        if i<5
            ax = axs[1,i]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        else
            ax = axs[2,i-4]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        end
    end 

    fig.suptitle("$(name)-other methods", fontsize=20)
    tight_layout()
    fig.savefig("draft2/$(name)-other_methods.png")

end


function plot_fmnist()
    name = "fmnist"
    gtdarecord = JSON.parsefile("stored_embeddings/fmnist.json")
    G_reeb = sparse(gtdarecord["G_reeb"][1],gtdarecord["G_reeb"][2],ones(length(gtdarecord["G_reeb"][2])))
    rc = gtdarecord["rc"]
    rn = gtdarecord["rn"]
    gtdatime = gtdarecord["time"]
    coords_dict = gtdarecord["all_coords"]
    #for name in ["spectral","spring","kamada_kawai"]
    X = hcat(gtdarecord["orig_data"]...)
    labels = gtdarecord["orig_labels"][1]

    labels_unique = unique(labels)
    cmap = PyPlot.cm.get_cmap("tab20")  # 20 colors
    color_map = Dict(lbl => cmap(mod1(i, 20) - 1) for (i, lbl) in enumerate(labels_unique))
    nodecolors = [color_map[lbl] for lbl in labels]
    labelcounter = zeros(length(labels_unique))

    println("starting to plot")
    ncols = length(coords_dict)
    nrows = 1
    fig, axs = subplots(nrows, ncols, figsize=(35, 10))
    layout_keys = collect(keys(coords_dict))
    n,d = size(X)
    mnistlegend = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankleboot"]

    for (i, key) in enumerate(layout_keys)
        ax = axs[i]
        xy = hcat(hcat(coords_dict[key]...)...)
        linesx, linesy = draw_graph_segments(G_reeb,xy)
        for (x, y) in zip(linesx, linesy)
            ax.plot(x, y, color="black", linewidth=0.1, alpha=0.1, zorder=1)
        end
        seen = Set{Int}()
        for i in range(1,size(xy,1))
            dist_dict = sort(StatsBase.countmap(labels[rc[i]]))
            distkey = [Int(k) for (k,v) in dist_dict]
            #labelcounter[[Int(k) for k in keys(dist_dict)]] = labelcounter[[Int(k) for k in keys(dist_dict)]] .+ 1
            showlabels = [k for k in distkey if !(k in seen)]
            union!(seen, distkey)
            ax = draw_pie(dist_dict,xy[i,1],xy[i,2],sqrt(length(rc[i])/1e5),ax,color_map,showlabels,lw = 0.5,actuallabels = mnistlegend)
        end
        ax.set_title("GTDR-$(match(r" (\w+)_layout",key).captures[1])")
        ax.axis("off")
        for lbl in labels_unique
        ax.scatter([], [], c=[color_map[lbl]], label=mnistlegend[lbl])
        end
        ax.legend()
        println("$(key) done")
    end
    tight_layout()
    fig.savefig("draft2/$(name).png")
    time_taken = Dict()
    push!(time_taken,"gtdr"=>gtdarecord["time"])

    allothermethods = load_object("/Users/alice/Documents/Research/RPs/GTDR/stored_embeddings/fmnist.jld2")
    ncols = div(length(allothermethods),2)
    nrows = 2
    fig, axs = subplots(nrows, ncols, figsize=(40, 10))

    for (i,k) in enumerate(keys(allothermethods))
        if i<5
            ax = axs[1,i]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        else
            ax = axs[2,i-4]
            xy = allothermethods[k][1][name][1]
            ax.scatter(xy[:,1], xy[:,2], c=nodecolors, s=9,edgecolors="black",linewidths=0.5,label = "")
            ax.set_title(k)
            ax.axis("off")
            push!(time_taken,k=>allothermethods[k][1][name][2])
        end
    end 

    fig.suptitle("$(name)-other methods", fontsize=20)
    tight_layout()
    fig.savefig("draft2/$(name)-other_methods.png")

end

function plot_coil20()
end

function plot_humandevelopmental()
end

function plot_zfishembryo()
end

function plot_mousestem()
end