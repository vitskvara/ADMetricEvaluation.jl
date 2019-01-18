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
	outpath = "parallel_test"
	
	mkpath(outpath)

	datasets = ""

	# settings
	n_experiments = 10
	p = 0.8
	mc_volume_iters = 10
	mc_volume_repeats = 1

	# models
	models = [kNN_model]
	model_names = ["kNN"]

	param_struct = [
					([[1, 3], [:gamma, :kappa]], [:k,:metric]),
				 ]

	function run_experiment(dataset)
		@time res = ADMetricEvaluation.run_umap_experiment(dataset, models, model_names, param_struct, outpath;
			n_experiments = n_experiments, p = p, contamination=0.05, mc_volume_iters = mc_volume_iters, 
			mc_volume_repeats = mc_volume_repeats, standardize=true)
		println("$dataset finished!")
	end
end

datasets = filter!(x->!(x in []), readdir(UCI.get_raw_datapath()))
#datasets = ["abalone", "iris", "statlog-vehicle"]

pmap(run_experiment, datasets)