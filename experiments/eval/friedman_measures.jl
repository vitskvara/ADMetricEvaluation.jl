using ADMetricEvaluation
using Statistics
using PyPlot
using DataFrames
using PaperUtils

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"

rankdf, alldf = ADMetricEvaluation.rank_models(data_path)

metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5]

Nm = length(metrics)

algorithms = [:kNN, :LOF, :IF, :OCSVM]

function algdf(df, alg, metrics)
	_df = deepcopy(df)
	_df[:metric] = map(x->string(split(string(x),"_mean_maximum")[1]), _df[:metric])
	resdf = DataFrame(:dataset=>String[])
	map(x->resdf[x]=Float64[], metrics)
	datasets = unique(_df[:dataset])
	for dataset in datasets
		row = Array{Any,1}()
		push!(row, dataset)
		subdf = _df[_df[:dataset].==dataset, [:dataset, :metric, alg]]
		for metric in metrics
			x = subdf[alg][subdf[:metric].==string(metric)]
			push!(row, length(x) == 0 ? NaN : x[1])
		end
		push!(resdf, row)
	end
	return resdf
end

algdf(rankdf, :kNN, metrics)
# construct the table for individual algorithms
measure_ranks = Dict(zip(algorithms, map(x->algdf(rankdf, x, metrics), algorithms)))
ranked_ranks = Dict(zip(algorithms, map(x->PaperUtils.rankdf(measure_ranks[x], false), algorithms)))
friedman_statistics = Dict(zip(algorithms, map(x->PaperUtils.friedman_statistic(convert(Matrix,ranked_ranks[x][1:end-1,metrics])), algorithms)))