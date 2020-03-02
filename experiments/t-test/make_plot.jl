include("plot_functions.jl")
contamination = 0.05
outpath = joinpath("/home/vit/vyzkum/measure_evaluation/t-test", string(contamination))
mkpath(outpath)

# setup dataset and models
base_dataset = "ecoli"
sub = "cp-im"

base_dataset = "statlog-satimage"
sub = "1-3"

base_dataset = "yeast"
sub = "CYT-VAC"


model_names = [:OCSVM, :kNN]
model_params = [[0.5], [21, :kappa]]

model_names = [:kNN, :kNN]
model_params = [[1, :gamma], [21, :kappa]]

model_names = [:kNN, :kNN]
model_params = [[19, :kappa], [21, :kappa]]

model_names = [:OCSVM, :kNN]
model_params = [[1.0], [21, :kappa]]

model_names = [:LOF, :kNN]
model_params = [[20], [21, :kappa]]

model_names = [:LOF, :IF]
model_params = [[20], [200]]


base_dataset = "breast-cancer-wisconsin"
sub = ""

model_names = [:IF, :OCSVM]
model_params = ([100], [0.1])


#### functions ########
make_plot_save_data(contamination, outpath, base_dataset, sub, model_names, 
	model_params)