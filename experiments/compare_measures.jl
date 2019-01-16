using DataFrames
using ADMetricEvaluation
using Statistics

#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
pareto_optimal = false

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
metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1,
:tpr_at_1, :vol_at_1]
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
measure_dict = Dict(zip(metrics, map(x->collect_rows(alldf,x,metrics),metrics)))
Statistics.mean(measure_dict[:vol_at_5][:vol_at_5] - measure_dict[:auc][:vol_at_5])
Statistics.mean(measure_dict[:vol_at_5][:vol_at_5]) - Statistics.mean(measure_dict[:auc][:vol_at_5])

stripnans(x) = x[.!isnan.(x)]
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


# histogram plot
using PyPlot
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
