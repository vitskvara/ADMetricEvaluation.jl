using CSV
using DataFrames
using PyPlot
using Statistics

loaddata(dataset::String, path = data_path) = 
	map(x->CSV.read(joinpath(path, dataset, x)), readdir(joinpath(path, dataset)))
subdatasets(dataframe_list::Array{DataFrame,1}) = unique([df[:dataset][1] for df in dataframe_list]) 
mergeds(df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	vcat(map(x->x[vcat([:dataset, :model], metrics)], df_list)...)
mergesubd(name::String, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1]==name,df_list), metrics)
mergesubd(names::Array{String,1}, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1] in names,df_list), metrics)
function linear_fit(x,y)
	# y = Xb, where X = [ones', x']'
	s = size(x)
	n = s[1]
	X = hcat(ones(n), x)
	Xt = Array(transpose(X))
	b = inv(Xt*X)*Xt*y
	return b
end
function plot_linear_fit(x,y)
	b = linear_fit(x,y)
	_x = [minimum(x), maximum(x)]
	_y = hcat(ones(2), _x)*b
	plot(_x, _y, c="k", lw=1)
end
function correlation_grid_plot(dataset, isubd, data_path;
		metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
		models = ["kNN", "IF", "LOF", "OCSVM"])
	dfs = loaddata(dataset)
	subd_list = subdatasets(dfs)
	isubd = min(isubd, length(subd_list))
	subd = subd_list[isubd]

	merged_df = mergesubd(subd, dfs, metrics)
	for metric in metrics
		merged_df[metric] = Float64.(merged_df[metric])
	end

	Nm = length(metrics)
	f = figure(figsize=(15,10))
	global n = 0
	for (j, m2) in enumerate(metrics)
		for (i, m1) in enumerate(metrics)
			n += 1
			subplot(Nm, Nm, n)
			# first, create a dummy scatter plot to get the x axis limits
			for model in models
				mdf = filter(x->x[:model]==model, merged_df)
				PyPlot.scatter(mdf[m1], mdf[m2], c="w")
			end
			ax = plt[:gca]()
			_xlim = ax[:get_ylim]()
			#ax[:cla]()

			if i == j
				for model in models
					mdf = filter(x->x[:model]==model, merged_df)
					PyPlot.plt[:hist](mdf[m1], 20, alpha = 1.0, label = model, density = true,
						histtype = "step")
				end
				xlim(_xlim)
				if i==j==1 legend() end
			elseif j>i
				_x = []
				_y = []
				for model in models
					mdf = filter(x->x[:model]==model, merged_df)
					PyPlot.scatter(mdf[m1], mdf[m2], s = 15, alpha = 0.2)
					push!(_x, mdf[m1])
					push!(_y, mdf[m2])
				end
				# add the fit line
				try
					plot_linear_fit(vcat(_x...), vcat(_y...))
				catch
					nothing
				end
				# add the correlation coefficient
				r = round(Statistics.cor(vcat(_x...), vcat(_y...)),digits=2)
				#text(0.1,0.9,"R=$r", size=8)
				_line = plt[:Line2D]([1], [1],color="w")
				legend([_line],["R=$r"], frameon=false)
			else
				for model in models
					mdf = filter(x->x[:model]==model, merged_df)
					PyPlot.scatter(mdf[m1], mdf[m2], s = 15, alpha = 0.2)
				end
				#mdf = filter(x->x[:model] in models, merged_df)
				#PyPlot.plt[:hist2d](mdf[m1], mdf[m2], 20, cmap = "hot_r")
			end
			# axis formatting
			if j == Nm xlabel(String(m1)) end
			if i == 1 ylabel(String(m2)) end
			if j != Nm ax[:set_xticklabels]([]) end
		end
	end
	tight_layout()
	f[:subplots_adjust](hspace=0)
	suptitle(subd)
	return f
end

### MAIN ###
if basename(PROGRAM_FILE) == basename(@__FILE__)

	dataset = ARGS[1]
	#dataset = "iris"
	isubd = (length(ARGS) > 1) ? isubd = Int(Meta.parse(ARGS[2])) : 1
	data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
	correlation_grid_plot(dataset, isubd, data_path)
	show()
end