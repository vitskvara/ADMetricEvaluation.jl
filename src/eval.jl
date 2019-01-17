function average_over_folds(df)
	if df[:model][1] == "IF"
		return aggregate(df, [:dataset, :model, :num_estimators], Statistics.mean)
	elseif df[:model][1] == "LOF"
		return aggregate(df, [:dataset, :model, :num_neighbors], Statistics.mean)
	elseif df[:model][1] == "OCSVM"
		return aggregate(df, [:dataset, :model, :gamma], Statistics.mean)
	elseif df[:model][1] == "kNN"
		return aggregate(df, [:dataset, :model, :metric, :k], Statistics.mean)
	end
end

function drop_cols!(df)
	if df[:model][1] == "IF"
		deletecols!(df, :num_estimators)
	elseif df[:model][1] == "LOF"
		deletecols!(df, :num_neighbors)
	elseif df[:model][1] == "OCSVM"
		deletecols!(df, :gamma)
	elseif df[:model][1] == "kNN"
		deletecols!(df, :metric)
		deletecols!(df, :k)	
	end
end

function merge_param_cols!(df)
	if df[:model][1] == "IF"
		col = "num_estimators=".*string.(df[:num_estimators])
	elseif df[:model][1] == "LOF"
		col = "num_neighbors=".*string.(df[:num_neighbors])
	elseif df[:model][1] == "OCSVM"
		col = "gamma=".*string.(df[:gamma])
	elseif df[:model][1] == "kNN"
		col = "metric=".*string.(df[:metric])
		col = col.*" k=".*string.(df[:k])
	end
	insertcols!(df, 3, :params=>col)
end

function pareto_optimal_params(df, metrics)
	X = convert(Array,df[metrics])'
	weight_mask = fill(false, length(metrics))
	weight_mask[findall(x->x == :auc_weighted_mean, metrics)[1]] == true
	pareto_optimal_i = MultiObjective.pareto_best_index(X, weight_mask)
	return df[pareto_optimal_i,:] 
end

function rank_models(data_path::String; 
	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
	models = ["kNN", "IF", "LOF", "OCSVM"],
	pareto_optimal = false
	)
	# first get the aggregated collection of all data
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		aggregdfs = []
		for df in dfs
			_df = average_over_folds(df)
			if pareto_optimal
				_df = pareto_optimal_params(_df, map(x->Symbol(string(x)*"_mean"), metrics))
			end
			merge_param_cols!(_df)
			drop_cols!(_df)
			push!(aggregdfs, _df)
		end
		push!(res, vcat(aggregdfs...))
	end
	if !pareto_optimal
		alldf = aggregate(vcat(res...), [:dataset, :model], maximum)
	else
		alldf = vcat(res...)
	end

	datasets = unique(alldf[:dataset])
	rankdf = DataFrame(:dataset=>String[], :metric=>Any[], :kNN=>Float64[], :LOF=>Float64[],
		:IF=>Float64[], :OCSVM=>Float64[])
	for dataset in datasets
		if pareto_optimal
			aggmetrics = map(x->Symbol(string(x)*"_mean"), metrics)
		else
			aggmetrics = map(x->Symbol(string(x)*"_mean_maximum"), metrics)
		end
		for metric in aggmetrics  
			vals = alldf[alldf[:dataset].==dataset, [:model, metric]]
			vals[:rank]=rankvals(vals[metric])
			try
				push!(rankdf, [dataset, metric, 
					vals[:rank][vals[:model].=="kNN"][1],
					vals[:rank][vals[:model].=="LOF"][1],
					vals[:rank][vals[:model].=="IF"][1],
					vals[:rank][vals[:model].=="OCSVM"][1]
					])
			catch
				nothing
			end
		end 
	end
	return rankdf, alldf
end

function rankvals(x::Vector, rev=true)
    j = 1
    tiec = 0 # tie counter
    y = Float64[]
    # create ranks
    sx = sort(x, rev=rev)
    is = sortperm(x, rev=rev)
    for _x in sx
        # this decides ties
        nties = size(x[x.==_x],1) - 1
        if nties > 0
            push!(y, (sum((j-tiec):(j+nties-tiec)))/(nties+1))
            tiec +=1
            # restart tie counter
            if tiec > nties
                tiec = 0
            end
        else
            push!(y,j)
        end
        j+=1
    end
    return y[sortperm(is)]
end


function model_ranks_stats(data_path, metrics=[:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1,
:tpr_at_1, :vol_at_1])
	rankdf, alldf = rank_models(data_path, metrics = metrics)

	Nm = length(metrics)

	ranks_mean = DataFrame(:metric=>String[],:kNN=>Float64[], :LOF=>Float64[], :IF=>Float64[], :OCSVM=>Float64[])
	ranks_sd = DataFrame(:metric=>String[],:kNN=>Float64[], :LOF=>Float64[], :IF=>Float64[], :OCSVM=>Float64[])
	figure(figsize=(10,5))
	global ind = 1
	for metric in (map(x->Symbol(string(x)*"_mean_maximum"), metrics))
		subplot(1,Nm,ind)
		mus = []
		sds = []
		for model in [:kNN, :LOF, :IF, :OCSVM]
			x = rankdf[model][rankdf[:metric].==metric]
			mu=Statistics.mean(x)
			std=Statistics.std(x)
			push!(mus, mu)
			push!(sds, std)
			plt[:hist](x, 20, label=string(model), alpha=1, histtype="step")
		end
		push!(ranks_mean, vcat([string(metric)], mus))
		push!(ranks_sd, vcat([string(metric)], sds))
		legend()
		xlabel(metric)
		global ind+=1
	end
	return ranks_mean, ranks_sd
end

stripnans(x) = x[.!isnan.(x)]
# now create a list of dataframes that contain rows from the alldf
# where a given measure is maximum and the rest is left as is
function collect_rows(alldf, metric, metrics)
	datasets = unique(alldf[:dataset])
	models = unique(alldf[:model])
	df = DataFrame(:dataset=>String[], :model=>String[])
	for m in metrics
		df[m] = Float64[]
	end
	for dataset in datasets
		for model in models
			subdf = alldf[(alldf[:dataset].==dataset) .& (alldf[:model].==model), :]
			if size(subdf, 1) > 0
				imax = argmax(subdf[metric])
				push!(df, subdf[imax, vcat([:dataset, :model], metrics)])
			end
		end
	end
	return df
end

function compare_measures(data_path, metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1];
		pareto_optimal=false)
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = ADMetricEvaluation.loaddata(dataset, data_path)
		aggregdfs = []
		for df in dfs
			_df = ADMetricEvaluation.average_over_folds(df)
			if pareto_optimal
				_df = ADMetricEvaluation.pareto_optimal_params(_df, map(x->Symbol(string(x)*"_mean"), metrics))
			end
			ADMetricEvaluation.merge_param_cols!(_df)
			ADMetricEvaluation.drop_cols!(_df)
			push!(aggregdfs, _df)
		end
		push!(res, vcat(aggregdfs...))
	end
	alldf = vcat(res...)
	for name in names(alldf)
		new_name = Symbol(split(string(name), "_mean")[1])
		rename!(alldf, name => new_name	)
	end
	measure_dict = Dict(zip(metrics, map(x->collect_rows(alldf,x,metrics),metrics)))
	
	# now create a set of tables that represent the mean loss and its variance in measure values 
	# when maximising by a another measure
	means = Dict(zip(metrics, map(x->Statistics.mean(measure_dict[x][x][.!isnan.(measure_dict[x][x])]), metrics)))
	mean_diff = DataFrame(:measure=>Symbol[])
	map(x->mean_diff[x] = Float64[], metrics)
	sd_diff = deepcopy(mean_diff)
	for metric_row in metrics
		mean_row = Array{Any,1}()
		sd_row = Array{Any,1}()
		push!(mean_row, metric_row)
		push!(sd_row, metric_row)
		for metric_column in metrics
			x1 = measure_dict[metric_row][metric_column] # available value = row
			x2 = measure_dict[metric_column][metric_column] # true maximum = column
			push!(mean_row, Statistics.mean(stripnans(x2.-x1)))
			push!(sd_row, Statistics.std(stripnans(x2.-x1)))
		end
		push!(mean_diff, mean_row)
		push!(sd_diff, sd_row)
	end
	# compute the relative losses
	rel_mean_diff = deepcopy(mean_diff)
	map(x->rel_mean_diff[x]=rel_mean_diff[x]/means[x],metrics)
	rel_sd_diff = deepcopy(sd_diff)
	map(x->rel_sd_diff[x]=rel_sd_diff[x]/means[x],metrics)
 
	# histogram plots
	nm = length(metrics)
	i=0 # i se pridava v radku, postupne se pridavaji radky
	for metric_row in metrics
		for metric_column in metrics
			x1 = measure_dict[metric_row][metric_column] # available value = row
			x2 = measure_dict[metric_column][metric_column] # true maximum = column
			i+=1
			ax = subplot(nm,nm,i)
			y = x2-x1
			plt[:hist](y[.!isnan.(y)],50)
			if i>nm*(nm-1)
				xlabel(string(metric_column))
			end
			if (i-1)%nm==0
				ylabel(string(metric_row))
			end
			
		end
	end
	return mean_diff, sd_diff, rel_mean_diff, rel_sd_diff	
end

select_hyperparams(df, subclass, measure, models) = map(x->argmax(df[measure][df[:model].==x]), models)

function sensitivity_df(data_path, dataset, measure)
	# dataset = "pendigits"
	dfs = ADMetricEvaluation.loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = ADMetricEvaluation.average_over_folds(df)
		ADMetricEvaluation.merge_param_cols!(_df)
		ADMetricEvaluation.drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	aggregdf = vcat(aggregdfs...)
	map(x->rename!(aggregdf, x=>Symbol(split(string(x),"_mean")[1])), names(aggregdf)[4:end])
	# now create the subclass df
	insertcols!(aggregdf, 2, :subclass=>map(x->split(x, dataset)[2][2:end], aggregdf[:dataset]))
	subclasses = unique(aggregdf[:subclass])
	sensdf = DataFrame(Symbol("max/loss")=>String[])
	for subclass in subclasses
		sensdf[Symbol(subclass)] = Float64[]
	end
	models = unique(aggregdf[:model])

	for rowclass in subclasses
		global rowvec = Array{Any,1}()
		push!(rowvec, string(rowclass))
		for columnclass in subclasses
			# we select hyperparams on one subclass
			rowsubdf = aggregdf[aggregdf[:subclass].==rowclass,:]
			hyperparams_inds = select_hyperparams(rowsubdf, rowclass, measure, models)
			columnsubdf = aggregdf[aggregdf[:subclass].==columnclass,:]
			# and evaluate the difference between the measure on row and column df
			rowmean = Statistics.mean(map(x->rowsubdf[measure][rowsubdf[:model].==x[1]][x[2]] ,zip(models,hyperparams_inds)))
			columnmean = Statistics.mean(map(x->columnsubdf[measure][columnsubdf[:model].==x[1]][x[2]] ,zip(models,hyperparams_inds)))
			push!(rowvec, abs(rowmean-columnmean)/columnmean*100.0)
		end
		push!(sensdf, rowvec)
	end
	return sensdf
end


function measures_correlations(data_path::String, dataset_info::String; 
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
