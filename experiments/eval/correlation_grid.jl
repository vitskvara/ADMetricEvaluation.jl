using ADMetricEvaluation
using DataFrames, DataFramesMeta
using StatsBase
using Statistics

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
cordfmean = ADMetricEvaluation.global_measure_correlation(data_path)

cordfmean2 = ADMetricEvaluation.global_measure_correlation(data_path; average_folds=true)