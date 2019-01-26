using ADMetricEvaluation
using DataFrames
using Statistics
using DataFramesMeta
using CSV

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]

datasets = readdir(data_path)
res = []
for dataset in datasets
	dfs = ADMetricEvaluation.loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = ADMetricEvaluation.average_over_folds(df)
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
measure_dict_all = Dict(zip(metrics, map(x->ADMetricEvaluation.collect_rows(alldf,x,metrics),metrics)))

datasets = unique(measure_dict_all[metrics[1]][:dataset])
per_dataset_dfs=[]
for dataset in datasets
	measure_dict = Dict(zip(metrics, map(x->(@linq measure_dict_all[x] |> where(:dataset.==dataset)), metrics)))
	means = Dict(zip(metrics, map(x->Statistics.mean(measure_dict[x][x][.!isnan.(measure_dict[x][x])]), metrics)))
	rel_mean_diff = DataFrame(:measure=>Symbol[])
	map(x->rel_mean_diff[x] = Float64[], metrics)
	for metric_row in metrics
		mean_row = Array{Any,1}()
		push!(mean_row, metric_row)
		for metric_column in metrics
			x1 = measure_dict[metric_row][metric_column] # available value = row
			x2 = measure_dict[metric_column][metric_column] # true maximum = column
			push!(mean_row, Statistics.mean(ADMetricEvaluation.stripnans(x2.-x1)))
		end
		push!(rel_mean_diff, mean_row)
	end
	# compute the relative losses
	map(x->rel_mean_diff[x]=rel_mean_diff[x]/means[x],metrics)
	push!(per_dataset_dfs, rel_mean_diff)
end
per_dataset_dfs = Dict(zip(datasets, per_dataset_dfs))
final_df = DataFrame(:dataset=>String[])
map(x->final_df[x]=Float64[] ,metrics)
for dataset in datasets
	row = Array{Any,1}()
	push!(row, dataset)
	subdf = per_dataset_dfs[dataset]
	X = convert(Matrix,subdf[:,metrics])
	meanvec = map(i->ADMetricEvaluation.nan_mean(X[i,:])*100, 1:size(X,1)) 
	row = vcat(row, meanvec)
	push!(final_df, row)
end

data_chars = CSV.read("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv")
data_chars = @linq data_chars |> where(:iteration.==1)
data_chars[:dataset] = String.(data_chars[:dataset])
final_df = join(final_df, data_chars, on=:dataset)
cols = names(final_df)
cols = filter(x->x!=:anomalous_p, cols)
cols = filter(x->x!=:dataset, cols)
cols = vcat([:dataset, :anomalous_p], cols)
final_df=final_df[cols]
sort!(final_df, :anomalous_p)
CSV.write("final_df.csv", final_df)