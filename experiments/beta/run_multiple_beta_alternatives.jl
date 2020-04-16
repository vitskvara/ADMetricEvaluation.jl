include("test_beta_alternatives_functions.jl")

svpath = "/home/vit/vyzkum/measure_evaluation/beta_alternatives/hist_auc"
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_beta_contaminated-0.00"

datasets = readdir(orig_path)
fprs = [0.01, 0.05]

measuref = gauss_auc
for dataset in datasets
	subsets = get_subsets(dataset)
	for subdataset in subsets
		results = save_measure_test_results(dataset, subdataset, measuref, svpath, fprs, orig_path)
	end
end
