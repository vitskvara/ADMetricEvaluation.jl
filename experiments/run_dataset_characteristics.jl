using ADMetricEvaluation

### UMAP
#output_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation"
#ADMetricEvaluation.umap_dataset_chars(output_path)

### synthetic
output_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/synthetic"
inpath="/home/vit/.julia/dev/UCI/synthetic"
ADMetricEvaluation.umap_dataset_chars(output_path; umap_data_path=inpath)