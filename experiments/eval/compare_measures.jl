using DataFrames
using ADMetricEvaluation
using Statistics
using PyPlot

#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
measures = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, 
		:tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]
mean_diff, sd_diff, rel_mean_diff, rel_sd_diff = ADMetricEvaluation.compare_measures(data_path, measures)