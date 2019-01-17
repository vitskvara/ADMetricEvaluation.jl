# run experiment on the full dataset
using UCI

@everywhere begin
	using ADMetricEvaluation
	using ScikitLearn
	using StatsBase

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
		outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/data_contaminated"
	elseif host == "axolotl.utia.cas.cz"
		outpath = "/home/skvara/work/anomaly_detection/data/metric_evaluation/full_data_contaminated"
	elseif host == "soroban-node-03"
		outpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_data_contaminated"
	end

	mkpath(outpath)

	datasets = ""

	# settings
	n_experiments = 10
	p = 0.8
	mc_volume_iters = 10000
	mc_volume_repeats = 10

	# models
	models = [kNN_model]
	model_names = ["kNN"]

	param_struct = [
					([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
				 ]

	function run_experiment(dataset)
		@time res = ADMetricEvaluation.run_umap_experiment(dataset, models, model_names, param_struct, outpath;
			n_experiments = n_experiments, p = p, contamination=0.05, mc_volume_iters = mc_volume_iters, 
			mc_volume_repeats = mc_volume_repeats, standardize=true)
		println("$dataset finished!")
	end
end

datasets = filter!(x->!(x in ["gisette"]), readdir(UCI.get_processed_datapath()))

pmap(run_experiment, datasets)