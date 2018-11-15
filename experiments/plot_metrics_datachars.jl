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
datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "yeast",
	"cardiotocography", "gisette", "glass", "haberman", "ionosphere", "libras", "madelon", 
	"magic-telescope", "miniboone", "parkinsons", "pima-indians", "sonar",
	"spect-heart", "statlog-segment",
	"statlog-vehicle", "vertebral-column", "wall-following-robot", "waveform-1",
	"waveform-2", "wine", "yeast"]
# page-blocks to rozbiji
models = ["kNN", "LOF", "OCSVM", "IF"]
filters = []
#filters = [(:clusterdness, ".<100")]
#filters=[(:n_clusters, ".<=2"), (:norm_vol, ".>0.8")]
data_chars = [:anomalous_p, :log_clusterdness, :norm_vol, :anomal_vol, :n_clusters]
f = ADME.scatter_grid_metrics_datachars(data_path, dataset_info; 
	datasets=datasets, models=models,filters=filters,
	data_chars = data_chars)
show()
