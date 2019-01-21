# run experiment on the full dataset
using UCI

@everywhere begin
	using ADMetricEvaluation
	using ScikitLearn
	using StatsBase

	ScikitLearn.@sk_import svm : OneClassSVM
	ScikitLearn.@sk_import ensemble : IsolationForest
	ScikitLearn.@sk_import neighbors : LocalOutlierFactor

	OCSVM_model(γ="auto") = ScikitLearn.OneClassSVM(gamma = γ)
	IF_model(n_estimators = 100) = ScikitLearn.IsolationForest(n_estimators = n_estimators, contamination = "auto", behaviour = "new")
	LOF_model(n_neighbors = 20) =  ScikitLearn.LocalOutlierFactor(n_neighbors = n_neighbors, novelty = true, contamination = "auto")

	mutable struct kNN_model
		k::Int
		m::Symbol
		t::Symbol
		knn::ADMetricEvaluation.KNNAnomaly
	end

	kNN_model(k::Int, metric::Symbol) = 
		kNN_model(k, metric, :KDTree, ADMetricEvaluation.KNNAnomaly(Array{Float32,2}(undef,1,0), metric, :KDTree))
	# create a sklearn-like fit function
	ScikitLearn.fit!(knnm::kNN_model, X) = (knnm.knn = ADMetricEvaluation.KNNAnomaly(Array(transpose(X)), knnm.m, knnm.t)) 
	ScikitLearn.decision_function(knnm::kNN_model, X) = -StatsBase.predict(knnm.knn, Array(transpose(X)), knnm.k)

	host = gethostname()
	#master path where data will be stored
	if host == "vit"
		outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_data"
	elseif host == "axolotl.utia.cas.cz"
		outpath = "/home/skvara/work/anomaly_detection/data/metric_evaluation/full_data"
	elseif host == "soroban-node-03"
		outpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_data"
	end

	mkpath(outpath)

	datasets = ""

	# settings
	n_experiments = 10
	p = 0.8
	mc_volume_iters = 10000
	mc_volume_repeats = 10

	# models
	models = [kNN_model, LOF_model, OCSVM_model, IF_model]
	model_names = ["kNN", "LOF", "OCSVM", "IF"]

	param_struct = [
					([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
				 	([[10, 20, 50]], [:num_neighbors]),
				 	([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
				 	([[50 100 200]], [:num_estimators]),
				 ]

	function run_experiment(dataset)
		@time res = ADMetricEvaluation.run_experiment(dataset, models, model_names, param_struct, outpath;
			n_experiments = n_experiments, p = p, contamination=0.00, mc_volume_iters = mc_volume_iters, 
			mc_volume_repeats = mc_volume_repeats, standardize=true)
		println("$dataset finished!")
	end
end

datasets = filter!(x->!(x in ["gisette", "isolet",
	"magic-telescope", "miniboone", "multiple-features",
	"statlog-shuttle"]), readdir(UCI.get_raw_datapath()))

pmap(run_experiment, datasets)