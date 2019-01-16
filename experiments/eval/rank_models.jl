using ADMetricEvaluation
using Statistics
using PyPlot
using DataFrames


#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_data_contaminated"

metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1,
:tpr_at_1, :vol_at_1]

rankdf, alldf = ADMetricEvaluation.rank_models(data_path, metrics=metrics)
ranks_mean, ranks_sd = ADMetricEvaluation.model_ranks_stats(data_path, metrics)
show()
println("RANK MEANS")
println(ranks_mean)
println("RANK STDS")
println(ranks_sd)

df1 = DataFrame(:dataset=>String[], :mse=>Float64[])
println("AUCw x VOL@5")
for dataset in unique(rankdf[:dataset])
	subdf=rankdf[rankdf[:dataset] .== dataset,:]
	x1 = permutedims(Vector(subdf[2,[:kNN, :LOF, :IF, :OCSVM]]))
	x2 = permutedims(Vector(subdf[6,[:kNN, :LOF, :IF, :OCSVM]]))
	mse = sum((x1.-x2).^2)/length(x1)
	push!(df1, [dataset, mse])
end
#println(df1)

df2 = DataFrame(:dataset=>String[], :mse=>Float64[])
println("AUC x AUCw")
for dataset in unique(rankdf[:dataset])
	subdf=rankdf[rankdf[:dataset] .== dataset,:]
	x1 = permutedims(Vector(subdf[2,[:kNN, :LOF, :IF, :OCSVM]]))
	x2 = permutedims(Vector(subdf[1,[:kNN, :LOF, :IF, :OCSVM]]))
	mse = sum((x1.-x2).^2)/length(x1)
	push!(df2, [dataset, mse])
end
#println(df2)