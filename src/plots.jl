loaddata(dataset::String, path) = 
	map(x->CSV.read(joinpath(path, dataset, x)), readdir(joinpath(path, dataset)))
subdatasets(dataframe_list::Array{DataFrame,1}) = unique([df[:dataset][1] for df in dataframe_list]) 
mergeds(df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	vcat(map(x->x[vcat([:dataset, :model], metrics)], df_list)...)
mergesubd(name::String, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1]==name,df_list), metrics)
mergesubd(names::Array{String,1}, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[:dataset][1] in names,df_list), metrics)
filter_string_by_beginning(x::String, master::String) = (length(x) < length(master)) ? false : (x[1:length(master)]==master)
filter_string_by_beginnings(x::String, masters::Array{String,1}) = any(map(y->filter_string_by_beginning(x,y),masters))
function load_all_by_model(data_path, model)
	datasets = readdir(data_path)
	dfs = []
	for dataset in datasets
		_dfs = loaddata(dataset, data_path)
		_dfs = filter(x->x[:model][1]==model, _dfs)
		push!(dfs, vcat(_dfs...))
	end
	dfs = filter(x->size(x,1)!=0, dfs)
	alldf = vcat(dfs...)
end
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
#metrics = [:iteration, :auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, 
#			:vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]
function correlation_grid_plot(df::DataFrame, metrics::Array{Symbol,1}, 
	models::Array{String,1}, sup_title::String; correlation::String="kendall",
	plot_histograms=true)
	Nm = length(metrics)
	f = figure(figsize=(15,10))
	global n = 0
	for (j, m2) in enumerate(metrics)
		for (i, m1) in enumerate(metrics)
			n += 1
			subplot(Nm, Nm, n)
			# first, create a dummy scatter plot to get the x axis limits
			for model in models
				mdf = filter(x->x[:model]==model, df)
				PyPlot.scatter(mdf[m1], mdf[m2], c="w")
			end
			ax = plt[:gca]()
			_xlim = ax[:get_ylim]()
			#ax[:cla]()

			if i == j
				if plot_histograms
					for model in models
						mdf = filter(x->x[:model]==model, df)
						PyPlot.plt[:hist](mdf[m1], 20, alpha = 1.0, label = model, density = true,
							histtype = "step")
					end
				end
				xlim(_xlim)
				if i==j==1 legend() end
			elseif j>i
				_x = []
				_y = []
				for model in models
					mdf = filter(x->x[:model]==model, df)
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
				r = NaN
				_leg = ""
				if correlation == "kendall"
					r = round(StatsBase.corkendall(vcat(_x...), vcat(_y...)),digits=2)
					_leg = "τ=$r"
				elseif correlation == "pearson"
					r = round(Statistics.cor(vcat(_x...), vcat(_y...)),digits=2)
					_leg = "R=$r"
				end
				_line = plt[:Line2D]([1], [1],color="w")
				legend([_line],[_leg], frameon=false)
			else
				for model in models
					mdf = filter(x->x[:model]==model, df)
					PyPlot.scatter(mdf[m1], mdf[m2], s = 15, alpha = 0.2)
				end
				#mdf = filter(x->x[:model] in models, df)
				#PyPlot.plt[:hist2d](mdf[m1], mdf[m2], 20, cmap = "hot_r")
			end
			# axis formatting
			if j == Nm xlabel(String(m1)) end
			if i == 1 ylabel(String(m2)) end
			if j != Nm ax[:set_xticklabels]([]) end
		end
	end
	suptitle(sup_title)
	tight_layout(rect=[0, 0.03, 1, 0.95])
	f[:subplots_adjust](hspace=0)
	return f
end

function single_dataset_corr_grid(dataset, isubd, data_path;
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

	# create the suptitle
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
	
	# now call the plotting function
	return correlation_grid_plot(merged_df, metrics, models, st)
end

function collect_all_data(data_path; aggreg_f = nothing,
	metrics= [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, 
			:vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1])
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		df = mergeds(dfs, vcat([:iteration], metrics)) 
		if aggreg_f == nothing
			push!(res, df)
		else
			push!(res, aggregate(df, [:dataset, :model], aggreg_f))
		end
	end
	return vcat(res...)
end

function join_with_info(all_data::DataFrame, dataset_info::String)
	infodf = CSV.read(dataset_info)
	# drop iterations
	infodf = infodf[infodf[:iteration].==1,filter(x->x!=:iteration,names(infodf))]
	# do the join
	return join(all_data, infodf, on=:dataset)
end

"""
Plots scatter plots and correlation for data collected across all datasets.
"""
function correlation_grid_datasets(data_path::String, dataset_info::String; 
	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
	models = ["kNN", "IF", "LOF", "OCSVM"],
	datasets = nothing,
	aggreg_f = Statistics.mean,
	filters = []
	)
	# first get the aggregated collection of all data
	alldf = join_with_info(collect_all_data(data_path; aggreg_f=aggreg_f, metrics = metrics), dataset_info)
	# filter out only some datasets
	if datasets != nothing
		alldf = alldf[map(x->filter_string_by_beginnings(x,datasets),alldf[:dataset]),:]
	end
	
	# filter out the new metric names - after aggregation they are different
	metrics = names(alldf)[map(x->filter_string_by_beginnings(x,string.(metrics)), string.(names(alldf)))]
	Nm = length(metrics)

	# create the suptitle
	st = ""

	# filter out by some columns
	if filters != []
		for _filter in filters
			fstring = repr(alldf[_filter[1]]) * _filter[2]
			alldf = alldf[eval(Meta.parse(fstring)),:]
			st *= string(_filter[1])*_filter[2]*"  " 
		end
	end

	# now call the plotting function
	f = correlation_grid_plot(alldf, metrics, models, st)

#	return alldf, metrics
	return f
end
function scatter_grid_plot(df::DataFrame, metrics::Array{Symbol,1}, 
	data_chars::Array{Symbol,1},
	models::Array{String,1}, sup_title::String; 
	correlation::String="kendall")
	Nm = length(metrics)
	Nc = length(data_chars)
	f = figure(figsize=(15,10))
	global n = 0
	for (j, m) in enumerate(metrics)
		for (i, c) in enumerate(data_chars)
			n += 1
			subplot(Nm, Nc, n)
			_x = []
			_y = []
			for model in models
				mdf = filter(x->x[:model]==model, df)
				PyPlot.scatter(mdf[c], mdf[m], s = 15, alpha = 0.2)
				push!(_x, mdf[c])
				push!(_y, mdf[m])
			end
			# add the fit line
			try
				plot_linear_fit(vcat(_x...), vcat(_y...))
			catch
				nothing
			end
			# add the correlation coefficient
			r = NaN
			_leg = ""
			if correlation == "kendall"
				r = round(StatsBase.corkendall(vcat(_x...), vcat(_y...)),digits=2)
				_leg = "τ=$r"
			elseif correlation == "pearson"
				r = round(Statistics.cor(vcat(_x...), vcat(_y...)),digits=2)
				_leg = "R=$r"
			end
			_line = plt[:Line2D]([1], [1],color="w")
			legend([_line],[_leg], frameon=false)

			# axis formatting
			ax = plt[:gca]()
			if j == Nm xlabel(String(c)) end
			if i == 1 ylabel(String(m)) end
			if j != Nm ax[:set_xticklabels]([]) end
			if i != 1 ax[:set_yticklabels]([]) end
		end
	end
	suptitle(sup_title)
	tight_layout(rect=[0, 0.03, 1, 0.95])
	f[:subplots_adjust](hspace=0, wspace=0)
	return f
end

function scatter_grid_metrics_datachars(data_path::String, dataset_info::String; 
	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
	data_chars = [:anomalous_p, :clusterdness, :norm_vol, :anomal_vol, :n_clusters],
	models = ["kNN", "IF", "LOF", "OCSVM"],
	datasets = nothing,
	aggreg_f = Statistics.mean,
	filters = []
	)
	# first get the aggregated collection of all data
	alldf = join_with_info(collect_all_data(data_path; aggreg_f=aggreg_f, metrics = metrics), dataset_info)
	# filter out only some datasets
	if datasets != nothing
		alldf = alldf[map(x->filter_string_by_beginnings(x,datasets),alldf[:dataset]),:]
	end

	# log clusterdness
	alldf[:log_clusterdness] = log.(alldf[:clusterdness])

	# filter out the new metric names - after aggregation they are different
	metrics = names(alldf)[map(x->filter_string_by_beginnings(x,string.(metrics)), string.(names(alldf)))]
	
	# create the suptitle
	st = ""

	# filter out by some columns
	if filters != []
		for _filter in filters
			fstring = repr(alldf[_filter[1]]) * _filter[2]
			alldf = alldf[eval(Meta.parse(fstring)),:]
			st *= string(_filter[1])*_filter[2]*"  " 
		end
	end

	# now call the plotting function
	return scatter_grid_plot(alldf, metrics, data_chars, models, st)
#	return alldf, metrics
end
