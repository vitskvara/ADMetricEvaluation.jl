# julia process_discriminability_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_pre
using ADMetricEvaluation

master_path = ARGS[1]
max_fpr = 1.0

datasets = readdir(master_path)
for dataset in datasets
	ADMetricEvaluation.process_discriminability_data(master_path, dataset, max_fpr)
end
