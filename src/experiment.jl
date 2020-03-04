"""
	precision_at_p(score_fun, X, y, p::Real; seed = nothing)

Compute precision at p% most anomalous samples. Subsample anomalies so 
that the ratio anomalous/all = p.
"""
function precision_at_p(score_fun, X, y, p::Real; seed = nothing, verb = false)
	N_a = sum(y)
	N_n = length(y) - N_a
	N = size(X,2)
	@assert N == length(y)
	(seed == nothing) ? nothing : Random.seed!(seed)
	# this is the number of sampled anomalies so that in the resulting subsampled dataset 
	# the ratio anomalous/(normal + anomalous) = p
	k = Int(floor(N*p/(1-p)))
	if k > N_a
		if verb
			@warn "Not enough anomalies to sample from"
		end
		return NaN
	end
	inds_sampled = StatsBase.sample(1:N_a, k, replace = false)
	Random.seed!() # restart the seed
	
	scores = score_fun(hcat(X[:,y.==0], X[:,y.==1][:,inds_sampled]))
	return EvalCurves.precision_at_k(scores, vcat(y[y.==0], y[y.==1][inds_sampled]), k)
end

mean_precision_at_p(score_fun, X, y, p::Real; n::Int=10, kwargs...) = 
	Statistics.mean(map(i->precision_at_p(score_fun, X, y, p; kwargs...), 1:n))

"""
	clusterdness(X,y)

Returns the ratio of variance between normal and anomalous samples.
"""
function clusterdness(X,y)
	if sum(y) == 0 # no anomalies
		return NaN
	elseif sum(y) == length(y) # no normal data points
		return NaN
	else
		return Statistics.mean(Statistics.var(X[:,y.==0], dims=2))/Statistics.mean(Statistics.var(X[:,y.==1], dims=2))
	end
end

"""
	experiment(model, parameters, X_train, y_train, X_test, y_test;
	mc_volume_iters::Int = 100000, mc_volume_repeats::Int = 10)

Basic one experiment function.
"""
function experiment(model, parameters, X_train, y_train, X_test, y_test;
	mc_volume_iters::Int = 100000, mc_volume_repeats::Int = 10,
	fpr_10 = false, robust_measures = false)
	# create and fit the model and produce anomaly scores
	m = model(parameters...)
	try
		ScikitLearn.fit!(m, Array(transpose(X_train)))
		score_fun(X) = -ScikitLearn.decision_function(m, Array(transpose(X))) 
		scores = score_fun(X_test)

		# now compute the needed metrics
		# auc-based
		resvec = Array{Any,1}()
		metric_vals = DataFrame(
						:auc=>Float64[],
						:auc_weighted=>Float64[],
						:auc_at_1=>Float64[],
						:auc_at_5=>Float64[],
						:prec_at_1=>Float64[],
						:prec_at_5=>Float64[],
						:tpr_at_1=>Float64[],
						:tpr_at_5=>Float64[],
						:vol_at_1=>Float64[],
						:vol_at_5=>Float64[],
						:f1_at_1=>Float64[],
						:f1_at_5=>Float64[]
						)
		fprvec, tprvec = EvalCurves.roccurve(scores, y_test)
		push!(resvec, EvalCurves.auc(fprvec, tprvec))
		push!(resvec, EvalCurves.auc(fprvec, tprvec, "1/x"))
		push!(resvec, EvalCurves.auc_at_p(fprvec,tprvec,0.01; normalize = true))
		push!(resvec, EvalCurves.auc_at_p(fprvec,tprvec,0.05; normalize = true))
		
		# instead of precision@k we will compute precision@p
		push!(resvec, mean_precision_at_p(score_fun, X_test, y_test, 0.01))
		push!(resvec, mean_precision_at_p(score_fun, X_test, y_test, 0.05))

		# tpr@fpr
		push!(resvec, EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.01))
		push!(resvec, EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.05))
		
		# volume of the anomalous samples
		X = hcat(X_train, X_test)
		bounds = EvalCurves.estimate_bounds(X)
		for (fpr, label) in [(0.01, :vol_at_1), (0.05, :vol_at_5)]
			threshold = EvalCurves.threshold_at_fpr(scores, y_test, fpr; warns = false)
			vf() = EvalCurves.volume_at_threshold(threshold, bounds, score_fun, mc_volume_iters)
			
			push!(resvec, 1-EvalCurves.mc_volume_estimate(vf, mc_volume_repeats))
		end

		# f1@alpha
		push!(resvec, EvalCurves.f1_at_fpr(scores, y_test, 0.01; warns=false))
		push!(resvec, EvalCurves.f1_at_fpr(scores, y_test, 0.05; warns=false))	

		if fpr_10
			df = DataFrame(
					:auc_at_10=>Float64[],
					:prec_at_10=>Float64[],
					:tpr_at_10=>Float64[],
					:vol_at_10=>Float64[],
					:f1_at_10=>Float64[]
					)
			metric_vals = hcat(metric_vals, df)
			push!(resvec, EvalCurves.auc_at_p(fprvec,tprvec,0.1; normalize = true))
			push!(resvec, mean_precision_at_p(score_fun, X_test, y_test, 0.1))
			push!(resvec, EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.1))
			threshold = EvalCurves.threshold_at_fpr(scores, y_test, 0.1; warns = false)
			vf10() = EvalCurves.volume_at_threshold(threshold, bounds, score_fun, mc_volume_iters)
			push!(resvec, 1-EvalCurves.mc_volume_estimate(vf10, mc_volume_repeats))
			push!(resvec, EvalCurves.f1_at_fpr(scores, y_test, 0.1; warns=false))
		end

		if robust_measures
			metric_vals = hcat(metric_vals,
				DataFrame(
					:auc_bs=>Float64[],
					:auc_at_1_bs=>Float64[],
					:tpr_at_1_bs=>Float64[],
					:auc_at_5_bs=>Float64[],
					:tpr_at_5_bs=>Float64[],
					:auc_gmm=>Float64[],
					:auc_at_1_gmm=>Float64[],
					:tpr_at_1_gmm=>Float64[],
					:auc_at_5_gmm=>Float64[],
					:tpr_at_5_gmm=>Float64[],
					:auc_gmm_5000=>Float64[],
					:auc_at_1_gmm_5000=>Float64[],
					:tpr_at_1_gmm_5000=>Float64[],
					:auc_at_5_gmm_5000=>Float64[],
					:tpr_at_5_gmm_5000=>Float64[]
					))

			N_samples = 1000
			# first compute the normal bootstrapping estimate
			push!(resvec, bootstrapped_measure(score_fun, EvalCurves.auc, 
				X_test, y_test, N_samples))
			push!(resvec, bootstrapped_measure(score_fun, 
				(x,y)->EvalCurves.auc_at_p(x,y,0.01; normalize = true), 
				X_test, y_test, N_samples))
			push!(resvec, bootstrapped_measure(score_fun, 
				(x,y)->EvalCurves.tpr_at_fpr(x,y,0.01), X_test, y_test, N_samples))
			push!(resvec, bootstrapped_measure(score_fun, 
				(x,y)->EvalCurves.auc_at_p(x,y,0.05; normalize = true), 
				X_test, y_test, N_samples))
			push!(resvec, bootstrapped_measure(score_fun, 
				(x,y)->EvalCurves.tpr_at_fpr(x,y,0.05), X_test, y_test, N_samples))

			# gmm bootstrapping
			# first do it by sampling the same number of positive and negative samples
			# as in the original test set
			rocs_gmm = fit_gmms_sample_rocs(score_fun(X_test), y_test, N_samples, 3)
			push!(resvec, measure_mean(rocs_gmm, EvalCurves.auc))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.auc_at_p(x,y, 0.01; normalize = true)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.tpr_at_fpr(x,y, 0.01)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.auc_at_p(x,y, 0.05; normalize = true)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.tpr_at_fpr(x,y, 0.05)))
			# now again, but this time sample a lot more 
			rocs_gmm = fit_gmms_sample_rocs(score_fun(X_test), y_test, N_samples, 3,
				5000)
			push!(resvec, measure_mean(rocs_gmm, EvalCurves.auc))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.auc_at_p(x,y, 0.01; normalize = true)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.tpr_at_fpr(x,y, 0.01)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.auc_at_p(x,y, 0.05; normalize = true)))
			push!(resvec, measure_mean(rocs_gmm, (x,y)->EvalCurves.tpr_at_fpr(x,y, 0.05)))
		end

		push!(metric_vals, resvec)
			
		return metric_vals
	catch e
		if isa(e, ArgumentError)
			println("Error in fit or predict:")
			println(e)
			println("")
		else
			rethrow(e)
		end
		df = DataFrame(
						:auc=>NaN,
						:auc_weighted=>NaN,
						:auc_at_1=>NaN,
						:auc_at_5=>NaN,
						:prec_at_1=>NaN,
						:prec_at_5=>NaN,
						:tpr_at_1=>NaN,
						:tpr_at_5=>NaN,
						:vol_at_1=>NaN,
						:vol_at_5=>NaN,
						:f1_at_1=>NaN,
						:f1_at_5=>NaN
						)
		if fpr_10
			df = hcat(df, DataFrame(
						:auc_at_10=>NaN,
						:prec_at_10=>NaN,
						:tpr_at_10=>NaN,
						:vol_at_10=>NaN,
						:f1_at_10=>NaN
						))
		end
		if robust_measures
			df = hcat(df,
				DataFrame(
					:auc_bs=>NaN,
					:auc_at_1_bs=>NaN,
					:tpr_at_1_bs=>NaN,
					:auc_at_5_bs=>NaN,
					:tpr_at_5_bs=>NaN,
					:auc_gmm=>NaN,
					:auc_at_1_gmm=>NaN,
					:tpr_at_1_gmm=>NaN,
					:auc_at_5_gmm=>NaN,
					:tpr_at_5_gmm=>NaN,
					:auc_gmm_5000=>NaN,
					:auc_at_1_gmm_5000=>NaN,
					:tpr_at_1_gmm_5000=>NaN,
					:auc_at_5_gmm_5000=>NaN,
					:tpr_at_5_gmm_5000=>NaN
					))
		end
		return df 
	end
end

"""
	discriminability_experiment(model, parameters, X_train, y_train, X_test, y_test,
	fprs)

Basic discriminability experiment function.
"""
function discriminability_experiment(model, parameters, X_train, y_train, X_test, y_test,
	fprs)
	# create and fit the model and produce anomaly scores
	m = model(parameters...)
	resvec = Array{Any,1}()
	metric_vals = DataFrame(
					:auc=>Float64[],
					:auc_weighted=>Float64[],
					:prec_at_1=>Float64[],
					:prec_at_5=>Float64[],
					:f1_at_1=>Float64[],
					:f1_at_5=>Float64[]
					)
	map(fpr->metric_vals[!,Symbol("auc_at_$(round(Int,100*fpr))")]=[],fprs)
	map(fpr->metric_vals[!,Symbol("tpr_at_$(round(Int,100*fpr))")]=[],fprs)

	try
		ScikitLearn.fit!(m, Array(transpose(X_train)))
		score_fun(X) = -ScikitLearn.decision_function(m, Array(transpose(X))) 
		scores = score_fun(X_test)

		# now compute the needed metrics
		# construct the output df
		
		# get the roc and the score
		fprvec, tprvec = EvalCurves.roccurve(scores, y_test)
		push!(resvec, EvalCurves.auc(fprvec, tprvec))
		push!(resvec, EvalCurves.auc(fprvec, tprvec, "1/x"))
		
		# instead of precision@k we will compute precision@p
		push!(resvec, mean_precision_at_p(score_fun, X_test, y_test, 0.01))
		push!(resvec, mean_precision_at_p(score_fun, X_test, y_test, 0.05))

		# f1@alpha
		push!(resvec, EvalCurves.f1_at_fpr(scores, y_test, 0.01; warns=false))
		push!(resvec, EvalCurves.f1_at_fpr(scores, y_test, 0.05; warns=false))	

		# tpr@fpr
		for fpr in fprs
			push!(resvec, EvalCurves.auc_at_p(fprvec,tprvec,fpr; normalize = true))
		end
		for fpr in fprs
			push!(resvec, EvalCurves.tpr_at_fpr(fprvec, tprvec, fpr))
		end	
		
	catch e
		if isa(e, ArgumentError)
			println("Error in fit or predict:")
			println(e)
			println("")
		else
			rethrow(e)
		end
		resvec = repeat([NaN], 6+2*length(fprs))
	end
	
	push!(metric_vals, resvec)
		
	return metric_vals

end

"""
	experiment_nfold(model, parameters, param_names, data::ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, exp_kwargs...)

Run the experiment n times with different resamplings of data.
"""
function experiment_nfold(model, parameters, param_names, data::UCI.ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, contamination::Real=0.05, 
	test_contamination = nothing, standardize=false, 
	discriminability_exp =false, fprs = collect(range(0.01,0.99,length=99)),
	exp_kwargs...)
	results = []
	for iexp in 1:n_experiments
		X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, p, contamination;
			test_contamination = test_contamination, seed = iexp, standardize=standardize)
		if discriminability_exp
			res = discriminability_experiment(model, parameters, X_tr, y_tr, X_tst, y_tst, 
				fprs)
		else
			res = experiment(model, parameters, X_tr, y_tr, X_tst, y_tst; exp_kwargs...)
		end	
		for (par_name, par_val) in zip(param_names, parameters)
			insertcols!(res, 1, par_name=>par_val) 
			# append the column to the beginning of the df
		end
		insertcols!(res, 1, :iteration=>iexp)
		# also, compute clusterdness
		#res[:clusterdness] = clusterdness(hcat(X_tr, X_tst), vcat(y_tr, y_tst))
		push!(results, res) # res is a dataframe 
	end
	return vcat(results...)
end


gridsearch(f, parameters...) = vcat(map(f, Base.product(parameters...))...)

"""
	run_experiment(model, model_name, param_vals, param_names, data::ADDataset, dataset_name;
	save_path = "", exp_nfold_kwargs...)

This iterates for a selected model over all params and dataset resampling iterations. 
If savepath is specified, saves the result in the given path.
"""
function run_experiment(model, model_name, param_vals, param_names, data::UCI.ADDataset, dataset_name;
	save_path = "", exp_nfold_kwargs...)
    res = gridsearch(x -> experiment_nfold(model, x, param_names, data; exp_nfold_kwargs...), param_vals...)
    insertcols!(res, 1, :model=>model_name)
    insertcols!(res, 1, :dataset=>dataset_name)
    if save_path != ""
    	CSV.write(joinpath(save_path, "$(dataset_name)_$(model_name).csv"), res)
    end
    return res
end

"""
	run_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	data_path = "", exp_kwargs...)

Runs the experiment for a given dataset.
"""
function run_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	data_path = "", exp_kwargs...)
	# load data
	raw_data = UCI.get_data(dataset_name, path=data_path)
	multiclass_data = UCI.create_multiclass(raw_data...)
	# setup path
	save_path = joinpath(master_save_path, dataset_name)
	mkpath(save_path)
	# now loop over all subclasses in a problem
	results = []
	p = Progress(length(model_list) * length(multiclass_data))
	for (i, (data, class_label)) in enumerate(multiclass_data)
		dataset_label = (class_label=="" ? dataset_name : dataset_name*"-"*class_label)
		# and over all models 
		for (model, model_name, params) in zip(model_list, model_names, param_struct)
			res = run_experiment(model, model_name, params[1], params[2], data, dataset_label; 
				save_path = save_path, exp_kwargs...)
			push!(results, res)
			ProgressMeter.next!(p; showvalues = [(:dataset,dataset_label), (:model,model_name)])
		end
	end
	return results
end

"""
	run_umap_experiment(dataset_name, model_list, model_names, param_struct, 
	master_save_path; exp_kwargs...)

Runs the experiment for UMAP data given a dataset name.
"""
run_umap_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	exp_kwargs...) = run_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	data_path = UCI.get_umap_datapath(), exp_kwargs...)

"""
	function run_synthetic_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	data_path = "", exp_kwargs...)

Runs the experiment for UMAP data given a dataset name.
"""
function run_synthetic_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	exp_kwargs...)
	# load data
	data = UCI.get_synthetic_data(dataset_name)
	# setup path
	save_path = joinpath(master_save_path, dataset_name)
	mkpath(save_path)
	# now loop over all subclasses in a problem
	results = []
	p = Progress(length(model_list))
	# and over all models 
	for (model, model_name, params) in zip(model_list, model_names, param_struct)
		res = run_experiment(model, model_name, params[1], params[2], data, dataset_name; 
			save_path = save_path, exp_kwargs...)
		push!(results, res)
		ProgressMeter.next!(p; showvalues = [(:dataset,dataset_name), (:model,model_name)])
	end
	return results
end

"""
	run_discriminability_experiment(dataset_name, model_list, model_names, param_struct, 
	master_save_path; exp_kwargs...)

Runs the experiment for UMAP data given a dataset name.
"""
run_discriminability_experiment(dataset_name, model_list, model_names, param_struct, 
	master_save_path;
	exp_kwargs...) = run_experiment(dataset_name, model_list, model_names, param_struct, 
	master_save_path; discriminability_exp=true, exp_kwargs...)

"""
	volume(bounds)

Compute volume enclosed by bounds.
"""
volume(bounds) = prod(map(x->x[2]-x[1], bounds))

"""
	n_clusters_hclust(X, h=4)

Number of clusters as computed by hclust.
"""
function n_clusters_hclust(X, h=4)
	maxn = min(5000,size(X,2))
	D = pairwise(Euclidean(),X[:,StatsBase.sample(1:maxn, maxn, replace=false)])
	hc = hclust(D)
	return length(unique(cutree(hc,h=h)))
end

"""
	n_clusters_affinity_prop(X)

Number of clusters as computed by affinity propagation.
"""
function n_clusters_affinity_prop(X)
	maxn = min(5000,size(X,2))
	S = -pairwise(Euclidean(),X[:,StatsBase.sample(1:maxn, maxn, replace=false)])
	med2 = 1/maxn
	#medS = Statistics.median(S)
	#medS = minimum(S)
	for i in 1:maxn
		S[i,i] = medS
	end
	ap = affinityprop(S)
	return length(unique(ap.assignments))
end

"""
	dataset_chars(X_tr, y_tr, X_tst, y_tst)

Compute different dataset characteristics.
"""
function dataset_chars(X_tr, y_tr, X_tst, y_tst)
	res = DataFrame()
	X = hcat(X_tr, X_tst)
	y = vcat(y_tr, y_tst)
	res[:anomalous_p] = sum(y)/length(y)
	res[:anomalous_p_test] = sum(y_tst)/length(y_tst)
	res[:clusterdness] = clusterdness(X,y)
	res[:M], res[:N] = size(X)
	# compute the volume ratios
	vx = volume(EvalCurves.estimate_bounds(X))
	res[:norm_vol] = volume(EvalCurves.estimate_bounds(X[:,y.==0]))/vx
	res[:anomal_vol] = volume(EvalCurves.estimate_bounds(X[:,y.==1]))/vx
	try
		res[:n_clusters] = n_clusters_hclust(X)
	catch
		res[:n_clusters] = NaN
	end
	return res
end

"""
	umap_dataset_chars(output_path; umap_data_path="", p=0.8, nexp=10, standardize=false)

Compute characteristics of all datasets.
"""
function umap_dataset_chars(output_path; umap_data_path="", p=0.8, contamination=0.0, nexp=10, standardize=false)
	datasets = readdir(UCI.get_raw_datapath())
	results = []
	mkpath(output_path)
	prog = Progress(length(datasets)) 
	for dataset in datasets
		multiclass_data = UCI.create_multiclass(UCI.get_umap_data(dataset, umap_data_path)...)	
		for (data, class_label) in multiclass_data
			dataset_label = (class_label=="" ? dataset : dataset*"-"*class_label)
			for iexp in 1:nexp
				X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, p, contamination; seed = iexp, 
					standardize=standardize)
				line = dataset_chars(X_tr, y_tr, X_tst, y_tst)
				insertcols!(line, 1, :iteration=>iexp)
				insertcols!(line, 1, :dataset=>dataset_label)
				push!(results, line)
			end
		end
		ProgressMeter.next!(prog; showvalues = [(:dataset,dataset)])
	end
	df = vcat(results...)
	CSV.write(joinpath(output_path, "dataset_overview_cont-$(contamination).csv"), df)
	return df 	
end
