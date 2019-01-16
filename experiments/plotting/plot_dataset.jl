import ADMetricEvaluation
ADME = ADMetricEvaluation

dataset = ARGS[1]

isubd = (length(ARGS) > 1) ? isubd = Int(Meta.parse(ARGS[2])) : 1
host = gethostname()
#master path where data is stored
data_type = "umap_data_contaminated"
if host == "vit"
	data_path = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation", data_type)
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
elseif host == "axolotl.utia.cas.cz"
	data_path = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation", data_type)
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
end

ADME.single_dataset_corr_grid(dataset, isubd, data_path; dataset_info=dataset_info)
show()
