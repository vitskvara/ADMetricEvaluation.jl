import ADMetricEvaluation
ADME = ADMetricEvaluation

dataset = ARGS[1]
#dataset = "iris"
isubd = (length(ARGS) > 1) ? isubd = Int(Meta.parse(ARGS[2])) : 1
host = gethostname()
#master path where data will be stored
if host == "vit"
	data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
elseif host == "axolotl.utia.cas.cz"
	data_path = "/home/skvara/work/anomaly_detection/data/metric_evaluation/umap_data"
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
end

ADME.correlation_grid_plot(dataset, isubd, data_path; dataset_info=dataset_info)
show()
