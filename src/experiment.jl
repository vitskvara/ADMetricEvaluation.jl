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
	N = size(X_test,2)
	metric_vals[:prec_at_5] = EvalCurves.precision_at_k(scores, y_test, Int(floor(N*0.05)))
	metric_vals[:prec_at_1] = EvalCurves.precision_at_k(scores, y_test, Int(floor(N*0.01)))

	# tpr@fpr
	metric_vals[:tpr_at_5] = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.05)
	metric_vals[:tpr_at_1] = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.01)
	
	# enclosed volume
	X = hcat(X_train, X_test)
	bounds = EvalCurves.estimate_bounds(X)
	all_scores = score_fun(X)
	for (fpr, label) in [(0.05, :vol_at_5), (0.01, :vol_at_1)]
		threshold = EvalCurves.threshold_at_fpr(scores, y_test, fpr)
		vf() = EvalCurves.volume_at_fpr(threshold, bounds, score_fun, mc_volume_iters)
		
		metric_vals[label] = EvalCurves.mc_volume_estimate(vf, mc_volume_repeats)
	end	
	return metric_vals
end

"""
	experiment_nfold(model, parameters, param_names, data::ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, exp_kwargs...)

Run the experiment n times with different resamplings of data.
"""
function experiment_nfold(model, parameters, param_names, data::ADDataset; 
	n_experiments::Int = 10, p::Real = 0.8, exp_kwargs...)
	results = []
	for iexp in 1:n_experiments
		X_tr, y_tr, X_tst, y_tst = split_data(data, p; seed = iexp)
		res = experiment(model, parameters, X_tr, y_tr, X_tst, y_tst; exp_kwargs...)
		for (par_name, par_val) in zip(param_names, parameters)
			insert!(res, 1, par_val, par_name) # append the column to the beginning of the df
		end
		insert!(res, 1, iexp, :iteration)
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
function run_experiment(model, model_name, param_vals, param_names, data::ADDataset, dataset_name;
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
	umap_data_path = "/home/vit/vyzkum/anomaly_detection/data/UCI/umap", exp_kwargs...)


"""
function run_umap_experiment(dataset_name, model_list, model_names, param_struct, master_save_path;
	umap_data_path = "/home/vit/vyzkum/anomaly_detection/data/UCI/umap", exp_kwargs...)
	# load data
	raw_data = get_umap_data(dataset_name, umap_data_path)
	multiclass_data = create_multiclass(raw_data...)
	# setup path
	save_path = joinpath(master_save_path, dataset_name)
	mkpath(save_path)
	# now loop over all subclasses in a problem
	results = []
	for (i, (data, class_label)) in enumerate(multiclass_data)
		dataset_label = (class_label=="" ? dataset_name : dataset_name*"-"*class_label)
		# and over all models 
		for (model, model_name, params) in zip(model_list, model_names, param_struct)
			res = run_experiment(model, model_name, params[1], params[2], data, dataset_label; 
				save_path = save_path, exp_kwargs...)
			push!(results, res)
		end
	end
	return results
end

#function run_experiment(models, model_names, param_vals, param_names, data::ADDataset;
#	exp_nfold_kwargs...)
#    results = []
#    for (model, model_name, p_vals, p_names) in zip(models, model_names, param_vals, param_names)
#	    model_res = gridsearch(x -> experiment_nfold(model, x, p_names, data;
#	    	exp_nfold_kwargs...), p_vals...)
#	    insert!(model_res, 1, model_name, :model)
#	    push!(results, model_res)
#	end
#    return vcat(results...)
#end

# cycle over iteration/seed with fixed parameters, subdataset and model
# cycle over parameters with fixed subdataset and model
# save here - all rows of (iterations x params) with model and subdataset fixed
# cycle over model with fixed subdataset
# cycle over subdatasets in a dataset

