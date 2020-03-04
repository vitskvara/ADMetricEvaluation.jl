using Statistics, JLD2, FileIO, Distributions, Interpolations
using UCI, ADMetricEvaluation, EvalCurves
using PyPlot
include("../models.jl")
ADME = ADMetricEvaluation
base_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation"

# two sample t test
welch_test_statistic(μ1::Real, μ2::Real, σ1::Real, σ2::Real, n1::Int, n2::Int) = 
	(μ1 - μ2)/sqrt(σ1^2/n1 + σ2^2/n2)
function welch_df(σ1::Real, σ2::Real, n1::Int, n2::Int)
	df = (σ1^2/n1 + σ2^2/n2)^2/(σ1^4/(n1^2*(n1-1)) + σ2^4/(n2^2*(n2-1)))
	isnan(df) ? NaN : floor(Int, df)
end
critt(α::Real, df::Int) = quantile(TDist(df), 1-α)
critt(α::Real, df::Real) = isnan(df) ? NaN : quantile(TDist(df), 1-α)
welch_pval(t::Real, df::Int) = 1-cdf(TDist(df), t) # onesided version
welch_pval(t::Real, df::Real) = isnan(df) ? NaN : 1-cdf(TDist(df), t) # onesided version

score_fun(model, X) = -ScikitLearn.decision_function(model, Array(transpose(X))) 

function get_roc(model, X_tr, X_tst, y_tst)
	ScikitLearn.fit!(model, Array(transpose(X_tr)))
	scores = score_fun(model, X_tst)
	EvalCurves.roccurve(scores, y_tst)
end

function get_scores(roc, fprs, measure)
	map(fpr -> measure(roc...,fpr), fprs)
end

function compute_measures(models, model_names_diff, fprs, ks, fname, contamination)
	if isfile(fname)
		measure_vals = load(fname)["measure_vals"]
	else
		measure_vals = Dict{Symbol, Any}(
			:fpr => fprs,
			:k => ks
			)
		for (model, model_name) in zip(models, model_names_diff)
			auc_at_p = []
			tpr_at_p = []
			rocs = []
			for k in ks
				X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8, contamination, seed=k, 
					standardize=true)
				roc = get_roc(model, X_tr, X_tst, y_tst)
				push!(auc_at_p, get_scores(roc, fprs, 
					(x,y,z)->EvalCurves.auc_at_p(x,y,z,normalize=true)))
				push!(tpr_at_p, get_scores(roc, fprs, EvalCurves.tpr_at_fpr))
				push!(rocs, roc)
			end
			measure_vals[model_name] = Dict(
				:auc_at_p => hcat(auc_at_p...),
				:tpr_at_p => hcat(tpr_at_p...),
				:roc => rocs
				)
		end
	end
	return measure_vals
end

function interpolate_roc(roc, fpr_knots)
	interpolation = LinearInterpolation(roc...)
	tpr_interpolated = map(x->interpolation(x), fpr_knots)
	tpr_interpolated[1] = 0.0
	tpr_interpolated
end

function get_statistics(measure_vals, measure_name,n,α,model_names_diff)
	means = map(m->mean(measure_vals[m][measure_name],dims=2),model_names_diff)
	stds = map(m->std(measure_vals[m][measure_name],dims=2),model_names_diff)
	welch = map(x->welch_test_statistic.(x[1], x[2], x[3], x[4], n, n), 
		zip(means[1], means[2], stds[1], stds[2]))
	dfs = map(x->welch_df.(x[1], x[2], n, n), 
		zip(stds[1], stds[2]))
	critvals = critt.(α/2, dfs)
	pvals = welch_pval.(abs.(welch), dfs)*2
	return means, stds, welch, dfs, critvals, pvals
end

connect_minus(a,b) = string(a)*"-"*string(b)
connect_und(a,b) = string(a)*"_"*string(b)

function make_plot_save_data(contamination, outpath, base_dataset, sub, model_names, 
	model_params)
	if model_names[1] != model_names[2]
		model_names_diff = copy(model_names)
	else
		model_names_diff = map(x->Symbol(string(x[2])*"$(x[1])"), enumerate(model_names))
	end

	# get data
	dataset_fname = base_dataset*"-"*sub
	data, normal_labels, anomaly_labels = UCI.get_data(base_dataset);
	subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels);
	data = filter(x->x[2]==sub,subdatasets)[1][1];
	X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8, 0.01, seed=1, 
		standardize=true)
	np = sum(y_tst)
	nn = length(y_tst) - np

	# construct models
	models = map(x->eval(Symbol(String(x[1])*"_model"))(x[2]...), 
		zip(model_names, model_params))
	model_fname = reduce(connect_und, map(x->reduce(connect_minus,vcat(x[1],x[2])),
		zip(model_names, model_params)))

	# get scores and rocs
	fprs = collect(0.005:0.005:0.3);
	ks = collect(1:50);
	fname = joinpath(outpath, dataset_fname*"_"*model_fname*".jld2")
	measure_vals = compute_measures(models, model_names_diff, fprs, 
		ks, fname, contamination)
	measure_names = [:auc_at_p, :tpr_at_p]
	measure_titles = ["AUC@FPR", "TPR@FPR"]

	# interpolate rocs to be able to plot the mean and sigmas
	fpr_knots = collect(0.0:0.005:1.0)
	tprs_interpolated = map(m->hcat([interpolate_roc(roc, fpr_knots) for 
		roc in measure_vals[m][:roc]]...), model_names_diff)
	tprs_interpolated_mean = map(x->vec(mean(x, dims=2)), tprs_interpolated)
	tprs_interpolated_sd = map(x->vec(std(x, dims=2)), tprs_interpolated)
	aucs = round.(map(x->EvalCurves.auc(fpr_knots, x), tprs_interpolated_mean), digits=2)

	# get the welch test statistics over fprs
	n = ks[end]
	α = 0.05
	pauc_means, pauc_stds, pauc_welch, pauc_dfs, pauc_critvals, pauc_pvals = 
		get_statistics(measure_vals, :auc_at_p,n,α,model_names_diff)
	tpr_means, tpr_stds, tpr_welch, tpr_dfs, tpr_critvals, tpr_pvals = 
		get_statistics(measure_vals, :tpr_at_p,n,α,model_names_diff)
	save(fname, "measure_vals", measure_vals, 
		"tpr_stats", Dict(
			:means => tpr_means, 
			:stds => tpr_stds, 
			:welch => tpr_welch, 
			:dfs => tpr_dfs, 
			:critvals => tpr_critvals, 
			:vals => tpr_pvals
			),
		"pauc_stats", Dict(
			:means => pauc_means, 
			:stds => pauc_stds, 
			:welch => pauc_welch, 
			:dfs => pauc_dfs, 
			:critvals => pauc_critvals, 
			:vals => pauc_pvals
			))

	# finally, get the max of welch statistics
	imax = argmax(abs.(pauc_welch))[1], argmax(abs.(tpr_welch))[1]

	# do the plot
	figure(figsize=(10,10))
	suptitle(dataset_fname*" "*model_fname*", np=$np, nn=$nn")

	# plot rocs
	subplot(421)
	colors = ["b", "g"]
	for (i, (μ,σ)) in enumerate(zip(tprs_interpolated_mean, tprs_interpolated_sd))
		plot(fpr_knots, μ, label=string(model_names[i])*" AUC=$(aucs[i])", 
			c=colors[i])
		fill_between(fpr_knots, μ.-1.96*σ, μ.+1.96*σ, color=colors[i], alpha=0.2)
	end
	legend(frameon=false)
	xlabel("FPR")
	ylabel("TPR")
	xlim(.0,1.02) 
	ylim(.0,1.02) 
	title("ROCs from $n-fold crossvalidation")

	# plot welch statistic
	subplot(422)
	title("Model discriminability")
	plot(fprs,abs.(tpr_welch), label="TPR@FPR")
	plot(fprs,abs.(pauc_welch), label="AUC@FPR")
	plot(fprs, tpr_critvals, "--", label="TPR@FPR critval")
	plot(fprs, pauc_critvals, "--", label="AUC@FPR critval")
	xlabel("FPR")
	ylabel("Welch statistic")
	axvline(0.05, color="gray", alpha=0.5)
	xlim(0.0,0.3)
	legend(frameon=false)

	# now plot histograms 
	for (i,measure_name) in enumerate(measure_names)
		subplot(422+i)
		title(measure_titles[i])
		ylabel("FPR=$(fprs[imax[i]])")
		for (j,m) in enumerate(model_names_diff)
			hist(measure_vals[m][measure_name][imax[i],:],30, density=true, alpha=0.3, 
				color=colors[j])
		end
	end

	for (k,fpr) in enumerate([0.01, 0.05])
		itpr = argmax(fprs.==fpr)[1] 
		for (i,measure_name) in enumerate(measure_names)
			subplot(424+i+2*(k-1))
			ylabel("FPR=$(fprs[itpr])")
			for (j,m) in enumerate(model_names_diff)
				hist(measure_vals[m][measure_name][itpr,:],30, density=true, alpha=0.4,
					color=colors[j])
			end
		end
	end
	tight_layout(rect=[0, 0.03, 1, 0.95])
	fname = joinpath(outpath, dataset_fname*"_"*model_fname*".png")
	savefig(fname)
end