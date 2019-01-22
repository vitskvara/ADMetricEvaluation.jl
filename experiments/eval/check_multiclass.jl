using ADMetricEvaluation
using DataFrames
using Statistics
using PyPlot
using DataFramesMeta
using LinearAlgebra

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"

bydf = ADMetricEvaluation.get_agregdf(data_path, "isolet", "1-23")
ondf = ADMetricEvaluation.get_agregdf(data_path, "isolet", "1-12")

#############################
dataset = "pendigits"
measure = :auc

sensitivity_dfs = ADMetricEvaluation.multiclass_sensitivities(data_path, dataset, measure)
stats = ADMetricEvaluation.multiclass_sensitivities_stats(data_path, dataset, measure) 
joined_stats = ADMetricEvaluation.join_multiclass_sensitivities_stats(stats)

