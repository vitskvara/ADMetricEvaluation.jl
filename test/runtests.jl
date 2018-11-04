import ADMetricEvaluation
ADME = ADMetricEvaluation
using UCI
using Test

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
	@test size(res) == (1,10)
	resn = ADME.experiment_nfold(model, params, parnames, yy[1][1]; n_experiments = 2, p=0.5,
		mc_volume_iters = 1000, mc_volume_repeats = 10)
	@test size(resn) == (2,13)
	parameters = [[1, 3, 5], [:gamma, :kappa]]
	resgs = ADME.gridsearch(x -> ADME.experiment_nfold(model, x, parnames, yy[1][1]; 
		n_experiments = 2, p = 0.5, mc_volume_iters = 1000, mc_volume_repeats = 10), parameters...)
	@test size(resgs) == (12,13)
	model_name = "kNN"
	resexp = ADME.run_experiment(model, model_name, parameters, parnames, yy[1][1], dataset; 
		save_path = ".", n_experiments = 2, p = 0.5, mc_volume_iters = 1000, mc_volume_repeats = 10)
	@test size(resexp) == (12,15)
	file = "$(dataset)_$(model_name).csv"
	@test isfile(file)
	rm(file) 
end