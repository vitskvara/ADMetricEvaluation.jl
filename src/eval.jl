nan_mean(x) = Statistics.mean(x[.!isnan.(x)])
matrix_col_nan_mean(X) =
	map(i->ADMetricEvaluation.nan_mean(X[i,:]), 1:size(X,1))

function average_over_folds(df)
	if df[!,:model][1] == "IF"
		return aggregate(df, [:dataset, :model, :num_estimators], Statistics.mean)
	elseif df[!,:model][1] == "LOF"
		return aggregate(df, [:dataset, :model, :num_neighbors], Statistics.mean)
	elseif df[!,:model][1] == "OCSVM"
		return aggregate(df, [:dataset, :model, :gamma], Statistics.mean)
	elseif df[!,:model][1] == "kNN"
		return aggregate(df, [:dataset, :model, :metric, :k], Statistics.mean)
	end
end

function drop_cols!(df)
	if df[!,:model][1] == "IF"
		df = select!(df, Not(:num_estimators))
	elseif df[!,:model][1] == "LOF"
		df = select!(df, Not(:num_neighbors))
	elseif df[!,:model][1] == "OCSVM"
		df = select!(df, Not(:gamma))
	elseif df[!,:model][1] == "kNN"
		df = select!(df, Not(:metric))
		df = select!(df, Not(:k))
	end
end

function merge_param_cols!(df)
	if df[!,:model][1] == "IF"
		col = "num_estimators=".*string.(df[!,:num_estimators])
	elseif df[!,:model][1] == "LOF"
		col = "num_neighbors=".*string.(df[!,:num_neighbors])
	elseif df[!,:model][1] == "OCSVM"
		col = "gamma=".*string.(df[!,:gamma])
	elseif df[!,:model][1] == "kNN"
		col = "metric=".*string.(df[!,:metric])
		col = col.*" k=".*string.(df[!,:k])
	end
	insertcols!(df, 3, :params=>col)
end

function loaddata(dataset::String, path; allsubdatasets=true)
	subdatasets = readdir(joinpath(path, dataset))
	if !allsubdatasets
		subdatasets=subdatasets[1:4]
	end
	map(x->CSV.read(joinpath(path, dataset, x)), subdatasets)
end
subdatasets(dataframe_list::Array{DataFrame,1}) = unique([df[!,:dataset][1] for df in dataframe_list]) 
mergeds(df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	vcat(map(x->x[!,vcat([:dataset, :model], metrics)], df_list)...)
mergesubd(name::String, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[!,:dataset][1]==name,df_list), metrics)
mergesubd(names::Array{String,1}, df_list::Array{DataFrame,1}, metrics::Array{Symbol, 1}) = 
	mergeds(filter(x->x[!,:dataset][1] in names,df_list), metrics)
filter_string_by_beginning(x::String, master::String) = (length(x) < length(master)) ? false : (x[!,1:length(master)]==master)
filter_string_by_beginnings(x::String, masters::Array{String,1}) = any(map(y->filter_string_by_beginning(x,y),masters))
function load_all_by_model(data_path, model)
	datasets = readdir(data_path)
	dfs = []
	for dataset in datasets
		_dfs = loaddata(dataset, data_path)
		_dfs = filter(x->x[!,:model][1]==model, _dfs)
		push!(dfs, vcat(_dfs...))
	end
	dfs = filter(x->size(x,1)!=0, dfs)
	alldf = vcat(dfs...)
end

function collect_all_data(data_path; aggreg_f = nothing,
	metrics= [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, 
			:vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1])
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		map(merge_param_cols!, dfs)
		df = mergeds(dfs, vcat([:params, :iteration], metrics)) 
		if size(df,1) == 0
			continue
		end
		map(x->df[!,x]=Float64.(df[!,x]), metrics)
		if aggreg_f == nothing
			push!(res, df)
		else
			push!(res, aggregate(df, [:dataset, :model, :params], aggreg_f))
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

#function pareto_optimal_params(df, metrics)
#	X = convert(Array,df[metrics])'
#	weight_mask = fill(false, length(metrics))
#	weight_mask[findall(x->x == :auc_weighted_mean, metrics)[1]] == true
#	pareto_optimal_i = MultiObjective.pareto_best_index(X, weight_mask)
#	return df[pareto_optimal_i,:] 
#end

function rank_models(data_path::String; 
	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
	models = ["kNN", "IF", "LOF", "OCSVM"]
	)
	# first get the aggregated collection of all data - means over params
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		aggregdfs = []
		for (i,df) in enumerate(dfs)
			if size(df,1) == 0
				continue
			end
			_df = average_over_folds(df)
			merge_param_cols!(_df)
			drop_cols!(_df)
			push!(aggregdfs, _df)
		end
		push!(res, vcat(aggregdfs...))
	end
	# now join this into one df and find the maximum on a dataset and model - the best params
	alldf = vcat(filter(x->size(x,1)!=0,res)...) # in case some dir is empty
	alldf = aggregate(alldf, [:dataset, :model], maximum)

	datasets = unique(alldf[!,:dataset])
	# in this df the ranks are going to be computed
	rankdf = DataFrame(:dataset=>String[], :metric=>Any[], :kNN=>Float64[], :LOF=>Float64[],
		:IF=>Float64[], :OCSVM=>Float64[])
	for dataset in datasets
		aggmetrics = map(x->Symbol(string(x)*"_mean_maximum"), metrics)
		i=0
		for metric in aggmetrics  
			vals = alldf[alldf[!,:dataset].==dataset, [:model, metric]]
			vals[!,:rank]=rankvals(vals[!,metric])
			try
				push!(rankdf, [dataset, metric, 
					vals[!,:rank][vals[!,:model].=="kNN"][1],
					vals[!,:rank][vals[!,:model].=="LOF"][1],
					vals[!,:rank][vals[!,:model].=="IF"][1],
					vals[!,:rank][vals[!,:model].=="OCSVM"][1]
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


function model_ranks_stats(data_path, 
	metrics=[:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1,
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
			x = rankdf[!,model][rankdf[!,:metric].==metric]
			mu=Statistics.mean(x)
			std=Statistics.std(x)
			push!(mus, mu)
			push!(sds, std)
			plt.hist(x, 20, label=string(model), alpha=1, histtype="step")
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
	datasets = unique(alldf[!,:dataset])
	models = unique(alldf[!,:model])
	df = DataFrame(:dataset=>String[], :model=>String[], :params=>String[])
	for m in metrics
		df[m] = Float64[]
	end
	for dataset in datasets
		for model in models
			subdf = alldf[(alldf[!,:dataset].==dataset) .& (alldf[!,:model].==model), :]
			x = subdf[metric]
			inds = .!isnan.(x) 
			#inds = 1:length(x)
			x = x[inds]
			if size(x, 1) > 0
				imax = argmax(x)
				push!(df, subdf[inds,:][imax, vcat([:dataset, :model, :params], metrics)])
			else
				push!(df, vcat([dataset, "", ""], fill(NaN, length(metrics))))
			end
		end
	end
	return df
end

function collect_fold_averages(data_path, metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1];
		#pareto_optimal=false, 
		models = ["kNN", "LOF", "IF", "OCSVM"],
		allsubdatasets = true)
	# get the list of datasets in the master path
	datasets = readdir(data_path)
	# now collect the averages over folds 
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path; allsubdatasets = allsubdatasets)
		aggregdfs = []
		for df in dfs
			if size(df,1) == 0
				continue
			end
			_df = average_over_folds(df)
			#if pareto_optimal
			#	_df = pareto_optimal_params(_df, map(x->Symbol(string(x)*"_mean"), metrics))
			#end
			merge_param_cols!(_df)
			drop_cols!(_df)
			push!(aggregdfs, _df)
		end
		if length(aggregdfs) == 0
			continue
		end
		push!(res, vcat(aggregdfs...))
	end
	alldf = vcat(res...)
	# filter out some models
	filter!(x->x[:model] in models, alldf)
	# remove the _mean suffix from the dataset
	for name in names(alldf)
		new_name = Symbol(split(string(name), "_mean")[1])
		rename!(alldf, name => new_name	)
	end
	return alldf
end

function rel_max_loss(measure_dict, measures)
	# this contains the mean differences
	diff_df = DataFrame(:measure=>Symbol[])
	map(x->diff_df[x] = Float64[], measures)
	for meas_row in measures
		row = Array{Any,1}()
		push!(row, meas_row)
		for meas_column in measures
			# the diagonals should be 0/NaNs
			if meas_column == meas_row
				push!(row, NaN)
			else
				try 
					x1 = measure_dict[meas_row][meas_column][1] # available value = row
					x2 = measure_dict[meas_column][meas_column][1] # true maximum = column
					push!(row, abs.(x2.-x1)/x2)
				catch e 
					# this happens if the csv file is not present
					if isa(e, BoundsError)
						push!(row, NaN)
					else
						throw(e)
					end
				end
			end
		end
		push!(diff_df, row)
	end
	return diff_df
end

function compare_measures_by_dataset_and_model(data_path, measures = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]; allsubdatasets = true)
	alldf = collect_fold_averages(data_path, measures; allsubdatasets = allsubdatasets)
	measure_dict = Dict(zip(measures, map(x->collect_rows(alldf,x,measures),measures)))
	datasets = unique(alldf[:dataset])
	models = unique(alldf[:model])
	rel_loss_dfs = []
	for dataset in datasets
		for model in models
			filtered_dict=Dict(zip(measures, map(key->(@linq measure_dict[key] |> where(:dataset.==dataset, :model.==model)), measures)))
			rel_loss_df = rel_max_loss(filtered_dict, measures)
			insertcols!(rel_loss_df, 1, :dataset=>dataset)
			insertcols!(rel_loss_df, 2, :model=>model)
			push!(rel_loss_dfs, rel_loss_df)
		end
	end
	return vcat(rel_loss_dfs...)
end


function compare_measures(data_path, metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1];
		#pareto_optimal=false, 
		models = ["kNN", "LOF", "IF", "OCSVM"],
		allsubdatasets = true)
	# collect all fold averages
	alldf = collect_fold_averages(data_path, metrics;
		#pareto_optimal=pareto_optimal, 
		models = models,
		allsubdatasets = allsubdatasets)
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
	#nm = length(metrics)
	#i=0 # i se pridava v radku, postupne se pridavaji radky
	#for metric_row in metrics
	#	for metric_column in metrics
	#		x1 = measure_dict[metric_row][metric_column] # available value = row
	#		x2 = measure_dict[metric_column][metric_column] # true maximum = column
	#		i+=1
	#		ax = subplot(nm,nm,i)
	#		y = x2-x1
	#		plt[:hist](y[.!isnan.(y)],50)
	#		if i>nm*(nm-1)
	#			xlabel(string(metric_column))
	#		end
	#		if (i-1)%nm==0
	#			ylabel(string(metric_row))
	#		end
	#		
	#	end
	#end
	return mean_diff, sd_diff, rel_mean_diff, rel_sd_diff	
end

function collect_rows_model_is_parameter(alldf, metric, metrics)
	datasets = unique(alldf[!,:dataset])
	df = DataFrame(:dataset=>String[], :model=>String[], :params=>String[])
	for m in metrics
		df[!,m] = Float64[]
	end
	for dataset in datasets
		subdf = alldf[(alldf[!,:dataset].==dataset), :]
		x = subdf[!,metric]
		inds = .!isnan.(x) 
		#inds = 1:length(x)
		x = x[inds]
		if size(x, 1) > 0
			imax = argmax(x)
			push!(df, subdf[inds,:][imax, vcat([:dataset, :model, :params], metrics)])
		else
			push!(df, vcat([dataset, "", ""], fill(NaN, length(metrics))))
		end
	end
	return df
end

function compare_measures_model_is_parameter(data_path, metrics = 
		[:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1];
		#pareto_optimal=false, 
		models = ["kNN", "LOF", "IF", "OCSVM"],
		allsubdatasets = true)
	# collect all fold averages
	alldf = collect_fold_averages(data_path, metrics;
		#pareto_optimal=pareto_optimal, 
		models = models,
		allsubdatasets = allsubdatasets)
	measure_dict = Dict(zip(metrics, map(x->collect_rows_model_is_parameter(alldf,x,metrics),metrics)))
	
	# now create a set of tables that represent the mean loss and its variance in measure values 
	# when maximising by a another measure
	means = Dict(zip(metrics, map(x->Statistics.mean(measure_dict[x][!,x][.!isnan.(measure_dict[x][!,x])]), metrics)))
	mean_diff = DataFrame(:measure=>Symbol[])
	map(x->mean_diff[!,x] = Float64[], metrics)
	sd_diff = deepcopy(mean_diff)
	for metric_row in metrics
		mean_row = Array{Any,1}()
		sd_row = Array{Any,1}()
		push!(mean_row, metric_row)
		push!(sd_row, metric_row)
		for metric_column in metrics
			x1 = measure_dict[metric_row][!,metric_column] # available value = row
			x2 = measure_dict[metric_column][!,metric_column] # true maximum = column
			push!(mean_row, Statistics.mean(stripnans(x2.-x1)))
			push!(sd_row, Statistics.std(stripnans(x2.-x1)))
		end
		push!(mean_diff, mean_row)
		push!(sd_diff, sd_row)
	end
	# compute the relative losses
	rel_mean_diff = deepcopy(mean_diff)
	map(x->rel_mean_diff[!,x]=rel_mean_diff[!,x]/means[x],metrics)
	rel_sd_diff = deepcopy(sd_diff)
	map(x->rel_sd_diff[!,x]=rel_sd_diff[!,x]/means[x],metrics)
 
	# histogram plots
	#nm = length(metrics)
	#i=0 # i se pridava v radku, postupne se pridavaji radky
	#for metric_row in metrics
	#	for metric_column in metrics
	#		x1 = measure_dict[metric_row][metric_column] # available value = row
	#		x2 = measure_dict[metric_column][metric_column] # true maximum = column
	#		i+=1
	#		ax = subplot(nm,nm,i)
	#		y = x2-x1
	#		plt[:hist](y[.!isnan.(y)],50)
	#		if i>nm*(nm-1)
	#			xlabel(string(metric_column))
	#		end
	#		if (i-1)%nm==0
	#			ylabel(string(metric_row))
	#		end
	#		
	#	end
	#end
	return mean_diff, sd_diff, rel_mean_diff, rel_sd_diff	
end

function compare_measures_by_dataset(data_path, metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]
		#pareto_optimal=false
		)
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		aggregdfs = []
		for df in dfs
			_df = average_over_folds(df)
			#if pareto_optimal
			#	_df = pareto_optimal_params(_df, map(x->Symbol(string(x)*"_mean"), metrics))
			#end
			merge_param_cols!(_df)
			drop_cols!(_df)
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
end

select_hyperparams(df, subclass, measure, models) = map(x->argmax(df[measure][df[:model].==x]), models)

function sensitivity_df(data_path, dataset, measure)
	# dataset = "pendigits"
	dfs = loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = average_over_folds(df)
		merge_param_cols!(_df)
		drop_cols!(_df)
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

function dataset_measure_correlations(alldf, dataset, measures; correlation="kendall")
	subdf = @linq alldf |> where(:dataset .== dataset)
	cordf = DataFrame(:measure=>String[])
	map(x->cordf[!,x]=Float64[], measures)
	for rowm in measures
		row = Array{Any,1}()
		push!(row, String(rowm))
		x = subdf[!,rowm]
		for colm in measures
			r = NaN
			y = subdf[!,colm]
			is=.!(isnan.(x) .| isnan.(y))
			_x=x[is]
			_y=y[is]
			if length(_x)>0 && correlation == "kendall"
				r = StatsBase.corkendall(_x, _y)
			elseif length(_x)>0 && correlation == "pearson"
				r = Statistics.cor(x, y)
			end
			push!(row, r)
		end
		push!(cordf, row)
	end
	return cordf
end

function global_measure_correlation(data_path, measures = 
		[:auc, :auc_weighted, :auc_at_5, :auc_at_1, :prec_at_5, :prec_at_1, 
		:tpr_at_5, :tpr_at_1, :vol_at_5, :vol_at_1]; 
		correlation = "kendall", average_folds = false)
	af = (average_folds ? Statistics.mean : nothing)
	alldf = collect_all_data(data_path, aggreg_f = af, metrics = measures)
	# rename the columns back if needed
	if average_folds
		map(x->rename!(alldf, Symbol(string(x)*"_mean")=>x), measures)
	end
	
	datasets = unique(alldf[!,:dataset])
	# for every dataset, create the correlation table

	cordfs = map(x->dataset_measure_correlations(alldf, x, measures; correlation = correlation), 
		datasets)
	corarr = cat(map(x->convert(Matrix, x[!,measures]), cordfs)..., dims=3)
	cordfmean = deepcopy(cordfs[1])
	for (i,rowm) in enumerate(measures)
		for (j,colm) in enumerate(measures)
			cordfmean[i,j+1] = nan_mean(corarr[i,j,:])
		end
	end
	return cordfmean
end

function get_agregdf(data_path, dataset, subclass)
	dfs = ADMetricEvaluation.loaddata(dataset, data_path)
	filter!(x->x[:dataset][1] == dataset*"-"*subclass, dfs)
	aggregdfs = []
	for df in dfs
		_df = ADMetricEvaluation.average_over_folds(df)
		ADMetricEvaluation.merge_param_cols!(_df)
		ADMetricEvaluation.drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	aggregdf = vcat(aggregdfs...)
end


##############################
### MULTICLASS SENSITIVITY ###
##############################

function sensitivity_by_model(aggregdf, model, measure::Symbol; max_loss = true) 
	# create the basis for the return df
	subclasses = unique(aggregdf[:subclass])
	sensdf = DataFrame(Symbol("max/loss")=>String[])
	for subclass in subclasses
		sensdf[Symbol(subclass)] = Float64[]
	end

	# filter only the one model requested
	df = @linq aggregdf |> where(:model .== model) 
	
	# now fill the dataframe
	for rowclass in subclasses
		global rowvec = Array{Any,1}()
		push!(rowvec, string(rowclass))
		for columnclass in subclasses
			# we select hyperparams on one subclass
			rowsubdf = filter!(x->.!isnan(x[measure]), @linq df |> where(:subclass.==rowclass))
			columnsubdf = filter!(x->.!isnan(x[measure]), @linq df |> where(:subclass.==columnclass))
			hyperparams_ind = argmax(rowsubdf[measure])
			# and evaluate the difference between the measure on row and column df
			params = rowsubdf[:params][hyperparams_ind]
			if !max_loss
				rowval = rowsubdf[measure][hyperparams_ind]
				columnval = (@linq columnsubdf |> where(:params.==params))[measure][1]
			else
				rowval = (@linq columnsubdf |> where(:params.==params))[measure][1]
				columnval = maximum(columnsubdf[measure])
			end
			push!(rowvec, abs(rowval-columnval)/columnval*100.0)
		end
		push!(sensdf, rowvec)
	end
	return sensdf 
end

function sensitivity_by_model_no_diff(aggregdf, model, measure::Symbol; max_loss = true) 
	# create the basis for the return df
	subclasses = unique(aggregdf[:subclass])
	sensdf = DataFrame(Symbol("max/loss")=>String[])
	for subclass in subclasses
		sensdf[Symbol(subclass)] = Float64[]
	end

	# filter only the one model requested
	df = @linq aggregdf |> where(:model .== model) 
	
	# now fill the dataframe
	for rowclass in subclasses
		global rowvec = Array{Any,1}()
		push!(rowvec, string(rowclass))
		for columnclass in subclasses
			try
				# we select hyperparams on one subclass
				rowsubdf = filter!(x->.!isnan(x[measure]), @linq df |> where(:subclass.==rowclass))
				columnsubdf = filter!(x->.!isnan(x[measure]), @linq df |> where(:subclass.==columnclass))
				hyperparams_ind = argmax(rowsubdf[measure])
				# and evaluate the difference between the measure on row and column df
				params = rowsubdf[:params][hyperparams_ind]
				rowval = (@linq columnsubdf |> where(:params.==params))[measure][1]
				columnval = maximum(columnsubdf[measure])
				push!(rowvec, rowval/columnval)
			catch e 
				push!(rowvec, NaN)
			end
		end
		push!(sensdf, rowvec)
	end
	return sensdf 
end

function multiclass_sensitivities_no_diff(data_path, dataset, measure)
	# collect data from all the subclasses
	dfs = loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = average_over_folds(df)
		merge_param_cols!(_df)
		drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	aggregdf = vcat(aggregdfs...)
	# rename the mean columns
	map(x->rename!(aggregdf, x=>Symbol(split(string(x),"_mean")[1])), names(aggregdf)[4:end])
	# now create subclass column
	insertcols!(aggregdf, 2, :subclass=>map(x->split(x, dataset)[2][2:end], aggregdf[:dataset]))
	subclasses = unique(aggregdf[:subclass])
	models = unique(aggregdf[:model])

	return Dict(zip(Symbol.(models), map(x->sensitivity_by_model_no_diff(aggregdf, x, measure), models)))
end

function multiclass_sensitivities(data_path, dataset, measure)
	# collect data from all the subclasses
	dfs = loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = average_over_folds(df)
		merge_param_cols!(_df)
		drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	aggregdf = vcat(aggregdfs...)
	# rename the mean columns
	map(x->rename!(aggregdf, x=>Symbol(split(string(x),"_mean")[1])), names(aggregdf)[4:end])
	# now create subclass column
	insertcols!(aggregdf, 2, :subclass=>map(x->split(x, dataset)[2][2:end], aggregdf[:dataset]))
	subclasses = unique(aggregdf[:subclass])
	models = unique(aggregdf[:model])

	return Dict(zip(Symbol.(models), map(x->sensitivity_by_model(aggregdf, x, measure), models)))
end

nan_aggreg(x::Vector, af) = af(x[.!isnan.(x)])

function multiclass_sensitivities_stats(sensdf)
	df = DataFrame()
	df[:subclass] = sensdf[1]
	X = convert(Matrix, sensdf[2:end])
	N,M = size(X)
	# remove diagonal zeros
	I = .!LinearAlgebra.diagm(0 => fill(true,N))
	X = Array(reshape(X'[I], N-1, N)')
	df[:mean] = reshape(Statistics.mean(X,dims=2), N)
	df[:median] = reshape(Statistics.median(X,dims=2), N)
	df[:min] = reshape(minimum(X,dims=2), N)
	df[:max] = reshape(maximum(X,dims=2), N)
	return df
end

function multiclass_sensitivities_stats(data_path, dataset, measure) 
	sens = multiclass_sensitivities(data_path, dataset, measure)
	stats = map(x->multiclass_sensitivities_stats(x), values(sens))
	return Dict(zip(keys(sens), stats))
end

prepend_model_to_colnames!(df, model) =
	map(x->rename!(df, x => Symbol(string(model)*"-"*string(x))), names(df)[2:end])
	
function join_multiclass_sensitivities_stats(stats)
	# rename columns by prepending the model type
	map(x->prepend_model_to_colnames!(x[2], x[1]), zip(keys(stats), values(stats)))
	models = collect(keys(stats))
	res = stats[models[1]]
	for model in models[2:end]
		res = join(res, stats[model], on = :subclass)
	end
	return res
end

join_multiclass_sensitivities_stats(data_path, dataset, measure) = 
	join_multiclass_sensitivities_stats(multiclass_sensitivities_stats(data_path, dataset, measure))

# now continue with multiclass_sensitivities_stats(_no_diff) output
function diag_nans!(df)
	M,N = size(df)
	for i in 1:M
		df[i, i+1] = NaN
	end
	return df
end

function collect_one_dataset_one_measure_data(data_path, dataset, measure; no_diff=true, 
	average_subproblems = false)
	sens_dict = (no_diff) ? 
		multiclass_sensitivities_no_diff(data_path, dataset, measure) :
		multiclass_sensitivities(data_path, dataset, measure)	
	models = collect(keys(sens_dict))
	res_df = DataFrame(:dataset=>String[], :model=>String[],
		:subclass=>String[], measure=>Array{Float64,1}[])
	for model in models
		df = diag_nans!(sens_dict[model])
		subclasses = df[Symbol("max/loss")]
		for subclass in subclasses
			row = Array{Any,1}()
			push!(row, dataset, string(model), subclass)
			# this is the row from df
			x = convert(Matrix, df[df[Symbol("max/loss")].==subclass, 2:end])
			if average_subproblems
				push!(row, [nan_mean(x)]) # vectorize this
			else
				push!(row, x[.!isnan.(x)])
			end
			push!(res_df, row)
		end
	end
	return res_df
end

function collect_one_dataset_data(data_path, dataset, measures; no_diff=true,
	average_subproblems=false)
	res_df = DataFrame(:dataset=>String[], :model=>String[],
				:subclass=>String[])
	map(x->res_df[x]=Array{Float64,1}[], measures)
	try
		res_df = collect_one_dataset_one_measure_data(data_path, dataset, measures[1];
			no_diff=no_diff, average_subproblems=average_subproblems)
	catch e
		if isa(e, SystemError) # the dataset does not exist in the data_path
			return res_df
		else
			throw(e)
		end
	end
	if length(measures) == 1
		return res_df
	end
	for measure in measures[2:end]
		df = collect_one_dataset_one_measure_data(data_path, dataset, measure;
		no_diff=no_diff, average_subproblems=average_subproblems)
		res_df = join(res_df, df, on = [:dataset, :model, :subclass])
	end
	return res_df
end

collect_all_datasets_data(data_path, datasets, measures;
	no_diff=true, average_subproblems=false) = 
		vcat(map(x->collect_one_dataset_data(data_path, x, measures; 
		no_diff=no_diff, average_subproblems=average_subproblems), datasets)...)
