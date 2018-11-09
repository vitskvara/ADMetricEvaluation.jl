#using Plots
using CSV
using DataFrames
#using StatPlots

dataset = ARGS[1]
#dataset = "iris"
isubd = (length(ARGS) > 1) ? isubd = Int(Meta.parse(ARGS[2])) : 0
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"

loaddata(dataset::String, path = data_path) = 
	map(x->CSV.read(joinpath(path, dataset, x)), readdir(joinpath(path, dataset)))
subdatasets(dataframe_list::Array{DataFrame,1}) = unique([df[:dataset][1] for df in dataframe_list]) 
mergeds(df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	vcat(map(x->x[vcat([:dataset, :model], metrics)], df_list)...)
mergesubd(name::String, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1]==name,df_list), metrics)
mergesubd(names::Array{String,1}, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1] in names,df_list), metrics)

dfs = loaddata(dataset)
subd_list = subdatasets(dfs)
subd = subd_list[isubd]
metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5]
models = ["kNN", "IF", "LOF", "OCSVM"]

merged_df = mergesubd(subd, dfs, metrics)
for metric in metrics
	merged_df[metric] = Float64.(merged_df[metric])
end

using PyPlot
Nm = length(metrics)
figure()
global n = 0
for (j, m2) in enumerate(metrics)
	for (i, m1) in enumerate(metrics)
		n += 1
		subplot(Nm, Nm, n)
		if j == Nm xlabel(String(m1)) end
		if i == 1 ylabel(String(m2)) end

		if i == j
			for model in models
				mdf = filter(x->x[:model]==model, merged_df)
				PyPlot.plt[:hist](mdf[m1], 20, alpha = 1.0, label = model, density = true,
					histtype = "step")
			end
			if i==j==Nm legend() end
		elseif j>i
			for model in models
				mdf = filter(x->x[:model]==model, merged_df)
				PyPlot.scatter(mdf[m1], mdf[m2], s = 15, alpha = 0.2)
			end
		else
			for model in models
				mdf = filter(x->x[:model]==model, merged_df)
				PyPlot.scatter(mdf[m1], mdf[m2], s = 15, alpha = 0.2)
			end
			#mdf = filter(x->x[:model] in models, merged_df)
			#PyPlot.plt[:hist2d](mdf[m1], mdf[m2], 20, cmap = "hot_r")
		end
	end
end
suptitle(subd)
show()
