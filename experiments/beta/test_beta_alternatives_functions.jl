using ADMetricEvaluation
ADME = ADMetricEvaluation
using EvalCurves, DataFrames, Statistics, CSV, PyPlot, UCI
using EvalCurves: beta_auc, localized_auc, f1_at_fpr
using GaussianMixtures, Suppressor

include("../models.jl")
savepath = "/home/vit/vyzkum/measure_evaluation/beta_alternatives"

model_names = ["kNN", "LOF", "OCSVM", "IF"]
model_list = [kNN_model, LOF_model, OCSVM_model, IF_model]
param_struct = [
                ([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
                ([[10, 20, 50]], [:num_neighbors]),
                ([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
                ([[50 100 200]], [:num_estimators]),
             ]

function compute_results(model, X, y, fprs, measuref; nsamples=1000)	
	throw_errs = false
	score_fun(X) = -ScikitLearn.decision_function(model, Array(transpose(X)))

	measures = DataFrame()
	
	scores = try
		score_fun(X)
	catch e
#		println(e)
		for fpr in fprs
			sfpr = "$(round(Int,100*fpr))"
			measures[!,Symbol("measure_at_$sfpr")] = [NaN]
		end
		return measures
	end	
	fprvec, tprvec = roccurve(scores, y)
	
	# compute the measure
	for fpr in fprs
		sfpr = "$(round(Int,100*fpr))"
		measures[!,Symbol("measure_at_$sfpr")] = 
			ADME._try_measure_computation(measuref, scores, y, fpr, nsamples;
			 throw_errs = throw_errs, warns=false)
	end
	return measures
end

function one_model_results(model, parameters, param_names, data, measuref; kwargs...)
	p = 0.6
	contamination = 0.0
	standardize = true
	n_experiments = 10
	test_contamination = nothing

	results_val = []
	results_tst = []
	for iexp in 1:n_experiments
		X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, p, contamination;
			test_contamination = test_contamination, seed = iexp, standardize=standardize)
		X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);
	
		m = model(parameters...)
		ScikitLearn.fit!(m, Array(transpose(X_tr)))

		# compute eval and test scores
		measures_val = compute_results(m, X_val, y_val, fprs, measuref; kwargs...)
		measures_tst = compute_results(m, X_tst, y_tst, fprs, measuref; kwargs...)

		for (par_name, par_val) in zip(param_names, parameters)
			insertcols!(measures_val, 1, par_name=>par_val) 
			insertcols!(measures_tst, 1, par_name=>par_val) 
			# append the column to the beginning of the df
		end
		insertcols!(measures_val, 1, :iteration=>iexp)
		insertcols!(measures_tst, 1, :iteration=>iexp)
		push!(results_val, measures_val)
		push!(results_tst, measures_tst)
	end
	return vcat(results_val...), vcat(results_tst...)
end

function test_measure(dataset, subdataset, measuref, fprs, model_list, model_names, param_struct; kwargs...)
	data_path = "" # UCI.get_processed_datapath()
	
	raw_data = UCI.get_data(dataset, path=data_path)
	multiclass_data = UCI.create_multiclass(raw_data...)
	data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]

	results = []
	for (model, model_name, params) in zip(model_list, model_names, param_struct)	
		param_vals = params[1]
		param_names = params[2]
	    res_val, res_tst = 
	    	ADME.gridsearch(x -> one_model_results(model, x, param_names, data, measuref; kwargs...), param_vals...)
	    insertcols!(res_val, 1, :model=>model_name)
	    insertcols!(res_val, 1, :dataset=>subdataset)
	    insertcols!(res_tst, 1, :model=>model_name)
	    insertcols!(res_tst, 1, :dataset=>subdataset)
	    push!(results, (res_val, res_tst))
	end
	return results
end

function save_results(results, path, model_names, subdataset)
	mkpath(path)
	for (result, model_name) in zip(results, model_names)
		CSV.write(joinpath(path, "$(subdataset)_$(model_name)_validation.csv"), result[1])
		CSV.write(joinpath(path, "$(subdataset)_$(model_name)_test.csv"), result[2])
	end
end

function load_results(path, model_names; subdataset="")
	ispath(path) ? nothing : (return [])
	fs = readdir(path)
	if subdataset != ""
		filter!(x->occursin(subdataset*"_",x), fs)
	end
	results = []
	for model_name in model_names
		_fs = filter(x->occursin(model_name, x), fs)
		if length(_fs) > 0
			res_val = CSV.read(joinpath(path, filter(x->occursin("validation", x), _fs)[1]))
			res_test = CSV.read(joinpath(path, filter(x->occursin("test", x), _fs)[1]))
			push!(results, (res_val, res_test))
		end
	end
	return results
end

# now join them
function join_dfs(df1, df2)
	if df1[!,:model][1] == "kNN"
		df1[!,:metric] = Symbol.(df1[!,:metric])
		df2[!,:metric] = Symbol.(df2[!,:metric])
		return join(df1, df2, on = [:dataset, :model, :metric, :k, :iteration])
	elseif df1[!,:model][1] == "LOF"
		return join(df1, df2, on = [:dataset, :model, :num_neighbors, :iteration]) 
	elseif df1[!,:model][1] == "OCSVM"
		return join(df1, df2, on = [:dataset, :model, :gamma, :iteration]) 
	elseif df1[!,:model][1] == "IF"
		return join(df1, df2, on = [:dataset, :model, :num_estimators, :iteration]) 
	end
end

# now compute the measure losses
function collect_fold_averages(dfs)
	# get the list of datasets in the master path
	aggregdfs = []
	for df in dfs
		if size(df,1) == 0
			continue
		end
		_df = ADME.average_over_folds(df)
		ADME.merge_param_cols!(_df)
		ADME.drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	alldf = vcat(aggregdfs...)
	# filter out some models
	join_und(x,y) = x*"_"*y
	# remove the _mean suffix from the dataset
	for name in names(alldf)
		ss = split(string(name), "_")
		new_name = (length(ss)==1) ? Symbol(ss[1]) : Symbol(reduce(join_und,ss[1:end-1]))
		rename!(alldf, name => new_name	)
	end
	return alldf
end

function exact_match(needle, haystack)
	n = length(needle)
	haystack[end-n+1:end] == needle
end
function compute_means(df, fprs, column_measures)
	nr = size(df,1)
	df[!,:mean] = zeros(nr)
	for row in eachrow(df)
		cm = row[:measure]
		ms = filter(x->x!=cm, column_measures)
		row[:mean] = mean(row[ms])
	end
	for fpr in fprs
		col_head = Symbol("mean_at_$fpr")
		df[!,col_head] = zeros(nr)
		for row in eachrow(df) 
			cm = row[:measure]
			ms = filter(y->exact_match("_$fpr", string(y)), filter(x->x!=cm, column_measures))
			row[col_head] = mean(row[ms])
		end
	end
	return df
end

function rel_measure_loss(alldf_val, alldf_tst, row_measures, column_measures, fprs, target_measure, filter_ratio = 0.0)
	# now collect the best results
	measure_dict_val = Dict(zip(row_measures, 
		map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_tst,x,column_measures),row_measures)))
	# these are as if selected by the column measures
	measure_dict_test = Dict(zip(column_measures, 
		map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_tst,x,column_measures),column_measures)))

	# we have to throw away some datasets where the results of bauc are either all NANs or they are
	# mostly nans
	datasets = unique(alldf_val[!,:dataset])
	filtered_datasets = 
		filter(x->sum(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,target_measure])) > length(filter(r->r[:dataset]==x, alldf_val)[!,target_measure])*(1-filter_ratio), datasets)

	if length(filtered_datasets) == 0
		nothing
	else
		alldf_val = filter(r->!(r[:dataset] in filtered_datasets), alldf_val)
		alldf_tst = filter(r->!(r[:dataset] in filtered_datasets), alldf_tst)
	end

	results = ADME.compute_measure_loss(alldf_val, alldf_tst, row_measures, column_measures)

	return compute_means(results[3], fprs, column_measures)
end

function measure_test_results(dataset, subdataset, measuref, savepath, fprs, orig_path; kwargs...)
	results = test_measure(dataset, subdataset, measuref, fprs, model_list, model_names, param_struct; kwargs...)

	original_results = load_results(joinpath(orig_path, dataset), model_names; subdataset = subdataset)

	dfs_val = map(x -> join_dfs(x[1][1], x[2][1]), zip(original_results, results))
	dfs_tst = map(x -> join_dfs(x[1][2], x[2][2]), zip(original_results, results))

	# also get the data anyway
	raw_data = UCI.get_data(dataset)
	multiclass_data = UCI.create_multiclass(raw_data...)
	data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]
	seed = 1
	X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, 0.6, 0.00;
		test_contamination = nothing, seed = seed, standardize=true)
	X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);

	alldf_val = collect_fold_averages(dfs_val)
	alldf_tst = collect_fold_averages(dfs_tst)

	fprs100 = map(x->round(Int, x*100), fprs)
	measure_loss_df = rel_measure_loss(alldf_val, alldf_tst, row_measures, column_measures, fprs100, 
		:measure_at_5, 0.0)

	return measure_loss_df, alldf_val, alldf_tst, (X_tr, y_tr, X_val, y_val, X_tst, y_tst)
end

function save_measure_test_results(dataset, subdataset, measuref, savepath, fprs, orig_path; kwargs...)
	path = joinpath(savepath, subdataset)
	results = test_measure(dataset, subdataset, measuref, fprs, model_list, model_names, param_struct; kwargs...)
	save_results(results, path, model_names, subdataset)	
	return results
end

function load_and_join(dataset, subdataset, orig_path, new_path)
	original_results = load_results(joinpath(orig_path, dataset), model_names; subdataset = subdataset)
	results = load_results(joinpath(new_path, subdataset), model_names)

	n = min(length(results), length(original_results))
	if n == 0
		return nothing, nothing
	end
	dfs_val = map(x -> join_dfs(x[1][1], x[2][1]), zip(original_results[1:n], results[1:n]))
	dfs_tst = map(x -> join_dfs(x[1][2], x[2][2]), zip(original_results[1:n], results[1:n]))

	alldf_val = collect_fold_averages(dfs_val)
	alldf_tst = collect_fold_averages(dfs_tst)

	return alldf_val, alldf_tst
end
get_subsets(dataset) = unique(map(x->split(x, "_")[1], readdir(joinpath(orig_path, dataset))))

function get_data(dataset, subdataset, seed)
	raw_data = UCI.get_data(dataset)
	multiclass_data = UCI.create_multiclass(raw_data...)
	data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]
	X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, 0.6, 0.00;
		test_contamination = nothing, seed = seed, standardize=true)
	X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);
	return X_tr, y_tr, X_val, y_val, X_tst, y_tst
end

gaussian_pdf(nd::Real,p::Real,n::Int) = 1/(sqrt(2*pi)*sqrt(n*p*(1-p)))*exp(-(nd-n*p)^2/(2*n*p*(1-p)))*n
function gauss_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    n = length(y_true) - sum(y_true) # number of negative samples

    # compute roc
    roc = roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = EvalCurves.linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = gaussian_pdf.(xw.*n, fpr, n)

    wauroc = auc(roci..., w)
end

function empirical_histogram_weights(x::Vector, samples::Vector, rounding=8)
    # do some rounding to ensure equalities
    x = round.(x, digits=rounding)
    dxs = round.(x[2:end].-x[1:end-1], digits=rounding)
    samples = round.(samples, digits = rounding)

    N = length(dxs)
    w = zeros(N)
    for i in 1:N
        if dxs[i] != 0
            w[i] = sum(x[i] .<= samples .< x[i+1])/length(samples)/dxs[i]
        end
    end
    return w
end
function hist_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    # first sample fprs and get parameters of the beta distribution
    fprs = EvalCurves.fpr_distribution(scores, y_true, fpr, nsamples, d, warns=warns)
    # filter out NaNs
    fprs = fprs[.!isnan.(fprs)]
    (length(fprs) == 0) ? (return NaN) : nothing

    # check for consistency
    if !EvalCurves._check_sampled_fpr_consistency(fpr, fprs)
        warns ? (@warn "the requested fpr is out of the sampled fpr distribution, returning NaN") : nothing
        return NaN
    end

    # compute roc
    roc = roccurve(scores, y_true)
    
    # compute the histogram weights
    w = empirical_histogram_weights(roc[1], fprs)

    # compute the integral
    wauroc = auc(roc..., w)
end


sample(gmm::GMM) = (n=StatsBase.sample(1:gmm.n, Weights(gmm.w)); randn()*sqrt(gmm.Σ[n])+gmm.μ[n])
sample(gmm::GMM, N::Int) = [sample(gmm) for _ in 1:N]
function gmm_fit(scores::Vector, y_true::Vector, ncomponents::Int)
    gmm0 = try
        @suppress begin
            GMM(ncomponents, scores[y_true.==0])
        end
    catch e    
        @warn(e)
        nothing
    end
    gmm1 = try
        @suppress begin
            GMM(ncomponents, scores[y_true.==1])
        end
    catch e    
        @warn(e)
        nothing
    end
    return gmm0, gmm1
end
function tpr_at_fpr_gmm(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; 
        min_samples::Int=1000, nc::Int=3, warns = true)
    # fit the gmms
    gmm0, gmm1 = gmm_fit(scores, y_true, nc);
    if (gmm0 == nothing) || (gmm1 == nothing)
        return NaN
    end
    
    # sample scores
    N = max(min_samples, length(scores))
    
    # get rocs and the respective tpr values
    ts = map(_->EvalCurves.tpr_at_fpr(roccurve(vcat(sample(gmm0, N), sample(gmm1, N)), 
                vcat(zeros(N), ones(N)))..., fpr), 1:nrepeats)
    mean(ts)
end