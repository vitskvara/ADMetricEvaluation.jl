using ADMetricEvaluation

### UMAP
output_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_characteristics"
mkpath(output_path)

ADMetricEvaluation.umap_dataset_chars(output_path; contamination=0.0)
ADMetricEvaluation.umap_dataset_chars(output_path; contamination=0.01)
ADMetricEvaluation.umap_dataset_chars(output_path; contamination=0.05)