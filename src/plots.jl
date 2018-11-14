loaddata(dataset::String, path) = 
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
		models = ["kNN", "IF", "LOF", "OCSVM"],
		dataset_info = "")
	dfs = loaddata(dataset, data_path)
	subd_list = subdatasets(dfs)
	isubd = min(isubd, length(subd_list))
	subd = subd_list[isubd]
	merged_df = mergesubd(subd, dfs, metrics)
	for metric in metrics
		merged_df[metric] = Float64.(merged_df[metric])
	end

	# if required, load information on subdatasets
	if dataset_info != ""
		info_df = CSV.read(dataset_info)
		# now all the lines for a subdataset are the same but it might not be the case later
		# if characteristics for individual training/testing splits are computed
		info_df = info_df[info_df[:dataset].==subd,:][1,:]
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
	st = subd
	if dataset_info != ""
		st = st*"\n"
		for info in [:anomalous_p, :clusterdness, :N, :norm_vol, :anomal_vol, :n_clusters]
			#st *= "$info=$(round(info_df[info][1],digits=3)), "
			x = info_df[info][1]
			if abs(x) < 1e-2
				sx = Printf.@sprintf("%.2e",x)
			else
				sx = "$(round(x,digits=2))"
			end
			st *= "$info=$sx  "
		end
	end
	suptitle(st)
	tight_layout(rect=[0, 0.03, 1, 0.95])
	f[:subplots_adjust](hspace=0)
	return f
end