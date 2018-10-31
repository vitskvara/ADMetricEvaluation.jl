import ADMetricEvaluation
ADME = ADMetricEvaluation
using Test 
include(joinpath(dirname(@__FILE__), "../experiments/models.jl"))

data_path = "/home/vit/vyzkum/anomaly_detection/data/UCI/umap"

dataset = "abalone"
xa = ADME.get_umap_data(dataset, data_path)
ya = ADME.create_multiclass(xa...)

dataset = "cardiotocography"
xc = ADME.get_umap_data(dataset, data_path)
yc = ADME.create_multiclass(xc...)
	
dataset = "yeast"
xy = ADME.get_umap_data(dataset, data_path)
yy = ADME.create_multiclass(xy...)

zaa = ADME.split_data(ya[1][1], 0.8)
zae = ADME.split_data(ya[1][1], 0.8, difficulty = :easy)
zaem = ADME.split_data(ya[1][1], 0.8, difficulty = [:easy, :medium])

zya = ADME.split_data(yy[1][1], 0.8)
#zye = ADME.split_data(yy[1][1], 0.8, difficulty = :easy)
zyem = ADME.split_data(yy[1][1], 0.8, difficulty = [:easy, :medium])

@testset "DATA" begin
	@test typeof(xa[1]) == ADME.ADDataset
	@test length(ya) == 1

	@test typeof(xc[1]) == ADME.ADDataset
	@test length(yc) > 1
	@test yc[1][2] == "2-9"

	@test yy[1][2] == "CYT-MIT"

	@test size(zaa[1],2) == size(zaem[1],2) == size(zae[1],2)
	@test sum(zaa[2]) == sum(zae[2]) == sum(zaem[2]) == 0 
	@test size(zaa[3],2) > size(zaem[3],2) > size(zae[3],2)
	@test sum(zaa[4]) > sum(zaem[4]) > sum(zae[4]) > 0

	try
		zye = ADME.split_data(yy[1][1], 0.8, difficulty = :easy)
	catch e
		@test isa(e, ErrorException)
	end
	@test size(zya[1],2) == size(zyem[1],2)
	@test sum(zya[2])  == sum(zyem[2]) == 0 
	@test size(zya[3],2) == size(zyem[3],2)
	@test sum(zya[4]) == sum(zyem[4]) > 0
end

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