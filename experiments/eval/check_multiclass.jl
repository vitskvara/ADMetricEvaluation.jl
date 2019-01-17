using ADMetricEvaluation
using DataFrames
using Statistics
using PyPlot

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"

function get_agregdf(dataset, subclass)
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

bydf = get_agregdf("isolet", "1-23")
ondf = get_agregdf("isolet", "1-12")

