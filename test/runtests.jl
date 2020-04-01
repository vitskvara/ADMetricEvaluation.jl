import ADMetricEvaluation
ADME = ADMetricEvaluation
using UCI
using EvalCurves
using Test
using Random

Random.seed!(123)

include(joinpath(dirname(@__FILE__), "../experiments/models.jl"))
	
dataset = "yeast"
xy = UCI.get_umap_data(dataset)
yy = UCI.create_multiclass(xy...)
zyem = UCI.split_data(yy[1][1], 0.8, difficulty = [:easy, :medium])

@testset "EXPERIMENT" begin
	params = (5, :gamma)
	parnames = (:k, :metric)
	model = kNN_model
	res = ADME.experiment(model, params, zyem[1], zyem[2], zyem[3], zyem[4];
		mc_volume_iters = 1000, mc_volume_repeats = 5)
	@test size(res) == (1,12)
	resn = ADME.experiment_nfold(model, params, parnames, yy[1][1]; n_experiments = 2, p=0.5,
		mc_volume_iters = 1000, mc_volume_repeats = 10)
	@test size(resn) == (2,15)
	parameters = [[1, 3, 5], [:gamma, :kappa]]
	resgs = ADME.gridsearch(x -> ADME.experiment_nfold(model, x, parnames, yy[1][1]; 
		n_experiments = 2, p = 0.5, mc_volume_iters = 1000, mc_volume_repeats = 10), parameters...)
	@test size(resgs) == (12,15)
	model_name = "kNN"
	resexp = ADME.run_experiment(model, model_name, parameters, parnames, yy[1][1], dataset; 
		save_path = ".", n_experiments = 2, p = 0.5, mc_volume_iters = 1000, mc_volume_repeats = 10)
	@test size(resexp) == (12,17)
	file = "$(dataset)_$(model_name).csv"
	@test isfile(file)
	rm(file) 

	# test of precision@p
	x =  hcat(fill(0,1,10), fill(1,1,4))
	y = reshape(copy(x),length(x))
	sf(X) = reshape(X, size(X,2))
	@test ADME.precision_at_p(sf, x, y, 0.1) == 1.0
	@test isnan(ADME.precision_at_p(sf, x, y, 0.4))
	function sf_wrong(X)
		res = fill(0,13)
		res[end] = 1
		res[end-2] = 1
		return res
	end
	@test ADME.precision_at_p(sf_wrong, x, y, 0.2) == 2/3

	# test of clusterdness
	@test isnan(ADME.clusterdness(randn(5,200), fill(0,200)))
	@test isnan(ADME.clusterdness(randn(5,200), fill(1,200)))
	@test abs(ADME.clusterdness(randn(5,200), vcat(fill(0,100), fill(1,100))) - 1.0) < 1.0
end

@testset "Bootstrapping" begin
	# classic bootstrapping
	X_tr, y_tr, X_tst, y_tst = zyem
	model = kNN_model(5, :gamma)
	ScikitLearn.fit!(model, Array(transpose(X_tr)))
	sf(X) = -ScikitLearn.decision_function(model, Array(transpose(X)))
	roc = EvalCurves.roccurve(sf(X_tst), y_tst)
	auc = EvalCurves.auc(roc...)
	pauc = EvalCurves.auc_at_p(roc..., 0.1)
	bs_auc = ADME.bootstrapped_measure(sf, EvalCurves.auc, X_tst, y_tst, 100)
	@test abs(auc-bs_auc) < 0.03
	bs_pauc = ADME.bootstrapped_measure(sf, (x,y)->EvalCurves.auc_at_p(x, y, 0.1), X_tst, y_tst, 100)
	@test abs(pauc-bs_pauc) < 0.01

	rocs_gmm = ADME.fit_gmms_sample_rocs(sf(X_tst), y_tst, 100, 3)
	# also test other cases where the last argument is present
	gmm_auc = ADME.measure_mean(rocs_gmm, EvalCurves.auc)
	@test abs(auc-gmm_auc) < 0.03
	gmm_pauc = ADME.measure_mean(rocs_gmm, (x,y)->EvalCurves.auc_at_p(x, y, 0.1))
	@test abs(pauc-gmm_pauc) < 0.01
end

@testset "beta auc" begin
	X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(yy[1][1], 0.6, difficulty = [:easy, :medium])
	X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);
	model = kNN_model(5, :gamma)
	ScikitLearn.fit!(model, Array(transpose(X_tr)))
	fprs = [0.01, 0.05, 0.1]
	measures_val = ADME.evaluate_val_test_experiment(model, X_val, y_val, fprs; 
		nsamples=1000, throw_errs = true)
	@test size(measures_val) == (1, 2 + length(fprs)*6)
	@test all(.!isnan.(Array(measures_val[1,:])))

	model = kNN_model
	parameters = [5, :gamma]
	measures_val, measures_tst = ADME.val_test_experiment(model, parameters, X_tr, y_tr, X_val, y_val,
		X_tst, y_tst, fprs)
	@test size(measures_val) == (1, 2 + length(fprs)*6)
	@test all(.!isnan.(Array(measures_val[1,:])))
	@test size(measures_tst) == (1, 2 + length(fprs)*6)
	@test all(.!isnan.(Array(measures_tst[1,:])))

	model = kNN_model
	parameters = [5, :gamma]
	param_names = [:k, :metric]
	data = yy[1][1]
	results_val, results_tst = ADME.val_test_experiment_nfold(model, parameters, param_names, data; 
		n_experiments = 5, p = 0.6, contamination=0.0,
		standardize=true, 
		fprs = fprs)
	@test size(results_val) == (5, 3 + 2 + length(fprs)*6)
	@test all(.!isnan.(Array(results_val[:,4:end])))
	@test size(results_tst) == (5, 3 + 2 + length(fprs)*6)
	@test all(.!isnan.(Array(results_tst[:,4:end])))

	model_name = "kNN"
	param_vals = [[5, 11], [:gamma, :kappa]]
	res_val, res_tst = ADME.run_val_test_experiment(model, model_name, param_vals, param_names, 
		data, dataset;
		save_path = ".", 
		n_experiments = 5, p = 0.6, contamination=0.0,
		standardize=true, 
		fprs = fprs)
	@test size(res_val) == size(res_tst) == (4*5, 2 + 3 + 2 + length(fprs)*6)
	file_val = "$(dataset)_$(model_name)_validation.csv"
	@test isfile(file_val)
	rm(file_val)
	file_tst = "$(dataset)_$(model_name)_test.csv"
	@test isfile(file_tst)
	rm(file_tst)

	model_names = ["kNN", "LOF"]
	models = [kNN_model, LOF_model]
	param_struct = [
	                ([[1, 3], [:gamma, :kappa]], [:k,:metric]),
	                ([[10]], [:num_neighbors])
	             ]
	results = ADME.run_experiment(dataset, models, model_names, param_struct, ".";
		val_test = true, n_experiments = 5, p = 0.6, contamination=0.0, standardize=true, fprs = fprs)
	@test length(results) == 2*length(yy)
	@test length(results[1]) == 2
	@test size(results[1][1]) == size(results[1][2]) == (4*5, 2 + 3 + 2 + length(fprs)*6)
	@test size(results[2][1]) == size(results[2][2]) == (5, 1 + 3 + 2 + length(fprs)*6)
	fs = filter(x->occursin(".csv", x), readdir("./$dataset"))
	@test length(fs) == 2*2*length(yy)
	rm(dataset, recursive=true)
end

Random.seed!()