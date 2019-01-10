using ADMetricEvaluation
using Statistics
using PyPlot
using DataFrames

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"

rankdf, alldf = ADMetricEvaluation.rank_models(data_path)

metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5]

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