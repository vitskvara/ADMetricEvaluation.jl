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
	mc_volume_iters::Int = 100000, mc_volume_repeats::Int = 10)
	# create and fit the model and produce anomaly scores
	m = model(parameters...)
	ScikitLearn.fit!(m, Array(transpose(X_train)))
	score_fun(X) = -ScikitLearn.decision_function(m, Array(transpose(X))) 
	scores = score_fun(X_test)

	# now compute the needed metrics
	# auc-based
	metric_vals = DataFrame() # auc, weighted auc, auc@5, auc@1, precision@k, tpr@fpr, vol@fpr
	fprvec, tprvec = EvalCurves.roccurve(scores, y_test)
	metric_vals[:auc] = EvalCurves.auc(fprvec, tprvec)
	metric_vals[:auc_weighted] = EvalCurves.auc(fprvec, tprvec, "1/x")
	metric_vals[:auc_at_5] = EvalCurves.auc_at_p(fprvec,tprvec,0.05; normalize = true)
	metric_vals[:auc_at_1] = EvalCurves.auc_at_p(fprvec,tprvec,0.01; normalize = true)
	
	# instead of precision@k we will compute precision@p
	metric_vals[:prec_at_5] = precision_at_p(score_fun, X_test, y_test, 0.05)
	metric_vals[:prec_at_1] = precision_at_p(score_fun, X_test, y_test, 0.01)

	# tpr@fpr
	metric_vals[:tpr_at_5] = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.05)
	metric_vals[:tpr_at_1] = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.01)
	
	# volume of the anomalous samples
	X = hcat(X_train, X_test)
	bounds = EvalCurves.estimate_bounds(X)
	for (fpr, label) in [(0.05, :vol_at_5), (0.01, :vol_at_1)]
		threshold = EvalCurves.threshold_at_fpr(scores, y_test, fpr; warn = false)
		vf() = EvalCurves.volume_at_fpr(threshold, bounds, score_fun, mc_volume_iters)
		
		metric_vals[label] = 1-EvalCurves.mc_volume_estimate(vf, mc_volume_repeats)
	end	
	return metric_vals
end

"""
	experiment_nfold(model, parameters, param_names, data::ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, exp_kwargs...)

Run the experiment n times with different resamplings of data.
"""
function experiment_nfold(model, parameters, param_names, data::UCI.ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, contamination::Real=0.05, standardize=false, exp_kwargs...)
	results = []
	for iexp in 1:n_experiments
		X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, p, contamination; seed = iexp, standardize=standardize)
		res = experiment(model, parameters, X_tr, y_tr, X_tst, y_tst; exp_kwargs...)
		for (par_name, par_val) in zip(param_names, parameters)
			insert!(res, 1, par_val, par_name) # append the column to the beginning of the df
		end
		insert!(res, 1, iexp, :iteration)
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
    insert!(res, 1, model_name, :model)
    insert!(res, 1, dataset_name, :dataset)
    if save_path != ""
    	CSV.write(joinpath(save_path, "$(dataset_name)_$(model_name).csv"), res)
    end
    return res
end

"""
	function run_umap_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	umap_data_path = "", exp_kwargs...)

Runs the experiment for UMAP data given a dataset name.
"""
function run_umap_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	umap_data_path = "", exp_kwargs...)
	# load data
	raw_data = UCI.get_umap_data(dataset_name, path=umap_data_path)
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
	function run_synthetic_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	umap_data_path = "", exp_kwargs...)

Runs the experiment for UMAP data given a dataset name.
"""
function run_synthetic_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	data_path = "", exp_kwargs...)
	# load data
	data = UCI.get_synthetic_data(dataset_name, path=data_path)
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
function umap_dataset_chars(output_path; umap_data_path="", p=0.8, nexp=10, standardize=false)
	datasets = readdir(UCI.get_raw_datapath())
	results = []
	mkpath(output_path)
	prog = Progress(length(datasets)) 
	for dataset in datasets
		multiclass_data = UCI.create_multiclass(UCI.get_umap_data(dataset, umap_data_path)...)	
		for (data, class_label) in multiclass_data
			dataset_label = (class_label=="" ? dataset : dataset*"-"*class_label)
			for iexp in 1:nexp
				X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, p; seed = iexp, 
					standardize=standardize)
				line = dataset_chars(X_tr, y_tr, X_tst, y_tst)
				insert!(line, 1, iexp, :iteration)
				insert!(line, 1, dataset_label, :dataset)
				push!(results, line)
			end
		end
		ProgressMeter.next!(prog; showvalues = [(:dataset,dataset)])
	end
	df = vcat(results...)
	CSV.write(joinpath(output_path, "dataset_overview.csv"), df)
	return df 	
end

"""
	umap_data(dataset, i; standardize=false)

Get X and y given a dataset name and subdataset index.
"""
function umap_data(dataset, i; standardize=false)
	data = UCI.get_umap_data(dataset)
	multidata = UCI.create_multiclass(data...)
	i = min(i, length(multidata))
	X_tr, y_tr, X_tst, y_tst = UCI.split_data(multidata[i][1]; standardize=standardize)
	return hcat(X_tr, X_tst), vcat(y_tr, y_tst)
end