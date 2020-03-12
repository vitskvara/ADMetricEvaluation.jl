using CSV, DataFrames, Statistics
using ADMetricEvaluation
ADME = ADMetricEvaluation
inpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_pre"
outpath = "/home/vit/vyzkum/measure_evaluation/discriminability/data"
datasets = readdir(inpath)
max_fpr = 0.3
measures = ["auc_at", "tpr_at"]
measures_mnx = 	["auc", "auc_at_1", "auc_at_5", "tpr_at_1", "tpr_at_5"] 
crits = ["tukey_q", "tukey_mean", "welch_mean", "tukey_median", "welch_median"]
measures_mnx_s = vcat(map(m->[Symbol(m*"_max"), Symbol(m*"_min")], measures_mnx)...)
measures_disc_s = vcat(vcat(map(m->map(c->[Symbol(m*"_"*c), Symbol(m*"_"*c*"_fpr")], 
	crits), measures)...)...)

nanmax(x) = maximum(x[.!isnan.(x)])
nanmin(x) = minimum(x[.!isnan.(x)])

function get_one_df(dataset)
	path = joinpath(inpath, dataset)
	infiles = readdir(path)
	dfs = map(x->CSV.read(joinpath(path, x)), infiles)
	alldf = vcat(map(df->ADME.drop_cols!(ADME.merge_param_cols!(copy(df))), dfs)...);
	subsets = unique(alldf[!,:dataset])

	df = DataFrame()
	df[!,:dataset] = subsets
	data = Dict()
	for m in vcat(measures_mnx_s, measures_disc_s) 
		data[m] = []
	end

	for subset in subsets
		# get the df of interest
		subsetdf = filter(r->r[:dataset]==subset, alldf)
		# get the min max for measures
		for m in measures_mnx 
			push!(data[Symbol(m*"_max")], nanmax(subsetdf[!,Symbol(m)]))
			push!(data[Symbol(m*"_min")], nanmin(subsetdf[!,Symbol(m)]))
		end
			
		# compute the optimal fpr level according to different measure and criterions
		opt_fprs = map(x->ADME.optimal_fprs(subsetdf, x, max_fpr)[1], measures)
		for (i,m) in enumerate(measures)
			for c in crits
				push!(data[Symbol(m*"_"*c)], opt_fprs[i][Symbol(c)][:val])
				push!(data[Symbol(m*"_"*c*"_fpr")], opt_fprs[i][Symbol(c)][:fpr])
			end
		end
	end

	for m in vcat(measures_mnx_s, measures_disc_s) 
		df[!,m] = data[m]
	end
	return df
end

dfs = map(get_one_df, datasets)
alldf = vcat(dfs...)

CSV.write(joinpath(outpath, "fpr_per_dataset_full-0.00.csv"), alldf)

alldf = CSV.read(joinpath(outpath, "fpr_per_dataset_full-0.00.csv"))
using PyPlot

figure(figsize=(4,10))
for (i,m) in enumerate([:auc_at_tukey_q_fpr, :auc_at_tukey_mean_fpr, :auc_at_tukey_median_fpr, 
	:auc_at_welch_mean_fpr, :auc_at_welch_median_fpr])
	subplot(510+i)
	title(string(m))
	hist(alldf[!,m],30)
end
tight_layout()
savefig(joinpath(outpath, "auc_at_fpr_full-0.00.eps"))

figure(figsize=(4,10))
for (i,m) in enumerate([:tpr_at_tukey_q_fpr, :tpr_at_tukey_mean_fpr, :tpr_at_tukey_median_fpr, 
	:tpr_at_welch_mean_fpr, :tpr_at_welch_median_fpr])
	subplot(510+i)
	title(string(m))
	hist(alldf[!,m],30)
end
tight_layout()
savefig(joinpath(outpath, "tpr_at_fpr_full-0.00.eps"))

figure()
scatter(alldf[!,:auc_max]-alldf[!,:auc_min], alldf[!,:auc_at_tukey_mean_fpr])


# now create the measure comaprison dfs on a dataset basis
p1 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_joined"
p2 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_0.00_single_dirs"

for f in readdir(p1)
	csvs = readdir(joinpath(p1,f))
	for d in unique(map(x->x[1],split.(csvs,"_")))
		mkpath(joinpath(p2,d,d))
		fs = filter(x->occursin(d,x), csvs)		
		map(x->cp(joinpath(p1,f,x), joinpath(p2,d,d,x), force=true), fs)
	end
end

# now call the measure comparison functions on each of these subdirectories
base_measures = [:auc, :auc_weighted, :auc_at_1, :auc_at_5,
	 :tpr_at_1, :tpr_at_5, :prec_at_1, :prec_at_5, 
	 :f1_at_1, :f1_at_5, :vol_at_1, :vol_at_5
 	 ]
crits = ["tukey_q", "tukey_mean", "tukey_median", "welch_mean", "welch_median"]
for crit in crits
	measures = vcat(base_measures, Symbol.([crit*"_auc_at", crit*"_tpr_at"]))

	function nancolsrowmean(x)
		z = x[:,.!vec(mapslices(y->all(isnan.(y)), x, dims=1))]
		return (length(z) == 0) ? repeat([NaN], size(x,1)) : mean(z, dims=2)
	end

	datadirs = readdir(p2)
	dfs = map(x->ADME.compare_measures_model_is_parameter(joinpath(p2,x), measures)[1],
		datadirs)
	map(df->df[!,:mean]=vec(nancolsrowmean(Array(df[!,2:end]))), dfs)

	setmeans = map(m->map(df->df[findfirst(x->x.==m,df[!,:measure]),:mean],dfs), measures[end-1:end])
	measure_loss_df = DataFrame(
		:dataset=>datadirs
		)
	map(x->measure_loss_df[!,Symbol(string(x[2])*"_loss")]=setmeans[x[1]], 
		enumerate(measures[end-1:end]))
	global alldf = join(alldf, measure_loss_df, on=:dataset)
end

CSV.write(joinpath(outpath, "fpr_per_dataset_full-0.00.csv"), alldf)

figure(figsize=(8,8))
subplot(321)
title("AUC@Welch mean")
scatter(alldf[!,:auc_at_welch_mean_fpr],alldf[!,:welch_mean_auc_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

subplot(323)
title("AUC@Tukey q")
scatter(alldf[!,:auc_at_tukey_q_fpr],alldf[!,:tukey_q_auc_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

subplot(325)
title("AUC@Tukey mean")
scatter(alldf[!,:auc_at_tukey_mean_fpr],alldf[!,:tukey_mean_auc_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

subplot(322)
title("TPR@Welch mean")
scatter(alldf[!,:tpr_at_welch_mean_fpr],alldf[!,:welch_mean_tpr_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

subplot(324)
title("TPR@Tukey q")
scatter(alldf[!,:tpr_at_tukey_q_fpr],alldf[!,:tukey_q_tpr_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

subplot(326)
title("TPR@Tukey mean")
scatter(alldf[!,:tpr_at_tukey_mean_fpr],alldf[!,:tukey_mean_tpr_at_loss],s=3)
xlabel("selected FPR")
ylabel("relative loss")

tight_layout()
savefig(joinpath(outpath, "fpr_vs_measure_loss_full-0.00.eps"))