import ADMetricEvaluation
ADME = ADMetricEvaluation
host = gethostname()
#master path where data will be stored
if host == "vit"
	data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
elseif host == "axolotl.utia.cas.cz"
	data_path = "/home/skvara/work/anomaly_detection/data/metric_evaluation/umap_data"
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
end

#datasets = filter(x->!(x in ["ecoli", "iris", "isolet", "multiple-features", "pendigits",
#	"statlog-satimage", "statlog-shuttle", "synthetic-control-chart"]), 
	#readdir(data_path))
datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "yeast"]
models = ["kNN", "LOF", "OCSVM", "IF"]
f = ADME.correlation_grid_datasets(data_path, dataset_info; datasets=datasets, models=models)
show()
