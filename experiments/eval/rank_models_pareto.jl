using ADMetricEvaluation
using Statistics
using PyPlot
using DataFrames

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"

rankdf, alldf = ADMetricEvaluation.rank_models(data_path, pareto_optimal=true)

metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5]

Nm = length(metrics)

df = DataFrame(:metric=>String[],:kNN=>Float64[], :LOF=>Float64[], :IF=>Float64[], :OCSVM=>Float64[])
figure(figsize=(10,5))
global ind = 1
for metric in (map(x->Symbol(string(x)*"_mean"), metrics))
	subplot(1,Nm,ind)
	mus = []
	for model in [:kNN, :LOF, :IF, :OCSVM]
		x = rankdf[model][rankdf[:metric].==metric]
		mu=Statistics.mean(x)
		push!(mus, mu)
		plt[:hist](x, 20, label=string(model), alpha=1, histtype="step")
	end
	push!(df, vcat([string(metric)], mus))
	legend()
	xlabel(metric)
	global ind+=1
end
show()

println(df)

df1 = DataFrame(:dataset=>String[], :mse=>Float64[])
println("AUCw x VOL@5")
for dataset in unique(rankdf[:dataset])
	subdf=rankdf[rankdf[:dataset] .== dataset,:]
	x1 = convert(Array, subdf[2,[:kNN, :LOF, :IF, :OCSVM]])
	x2 = convert(Array, subdf[6,[:kNN, :LOF, :IF, :OCSVM]])
	mse = sum((x1.-x2).^2)/length(x1)
	push!(df1, [dataset, mse])
end

df2 = DataFrame(:dataset=>String[], :mse=>Float64[])
println("AUC x AUCw")
for dataset in unique(rankdf[:dataset])
	subdf=rankdf[rankdf[:dataset] .== dataset,:]
	x1 = convert(Array, subdf[2,[:kNN, :LOF, :IF, :OCSVM]])
	x2 = convert(Array, subdf[1,[:kNN, :LOF, :IF, :OCSVM]])
	mse = sum((x1.-x2).^2)/length(x1)
	push!(df2, [dataset, mse])
end
