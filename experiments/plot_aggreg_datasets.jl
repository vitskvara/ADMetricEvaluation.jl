using PyPlot
import ADMetricEvaluation
ADME = ADMetricEvaluation
host = gethostname()
#master path where data will be stored
data_type = "umap_data_contaminated"
if host == "vit"
	data_path = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation", data_type)
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
	output_path = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation", data_type*"_aggregated_plots")
elseif host == "axolotl.utia.cas.cz"
	data_path = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation", data_type)
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
	output_path = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation", data_type*"_aggregated_plots")
end
mkpath(output_path)
#datasets = filter(x->!(x in ["ecoli", "iris", "isolet", "multiple-features", "pendigits",
#	"statlog-satimage", "statlog-shuttle", "synthetic-control-chart"]), 
	#readdir(data_path))
datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "yeast",
	"cardiotocography", "gisette", "glass", "haberman", "ionosphere", "libras", "madelon", 
	"magic-telescope", "miniboone", "parkinsons", "pima-indians", "sonar",
	"spect-heart", "statlog-segment",
	"statlog-vehicle", "vertebral-column", "wall-following-robot", "waveform-1",
	"waveform-2", "wine", "yeast"]
# page-blocks to rozbiji
models = ["kNN", "LOF", "OCSVM", "IF"]
filters = []
#filters=[(:n_clusters, ".<=2"), (:norm_vol, ".>0.8")]
f = ADME.correlation_grid_datasets(data_path, dataset_info; 
	datasets=datasets, models=models,filters=filters)
#construct the filename
f = "agreggated_datasets"
if filters != []
	for _filter in filters
		fstring = repr(alldf[_filter[1]]) * _filter[2]
		f *= "_"*string(_filter[1])*-filter[2] 
	end
end
f *= ".png"
# save the figure
savefig(joinpath(output_path, f))
show()