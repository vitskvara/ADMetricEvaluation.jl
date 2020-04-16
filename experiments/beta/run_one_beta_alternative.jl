include("test_beta_alternatives_functions.jl")

svpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/beta_alternatives/hist_auc"
orig_path = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
fprs = [0.01, 0.05]

measuref = hist_auc
dataset = ARGS[1]
subsets = get_subsets(dataset)
for subdataset in subsets
	results = save_measure_test_results(dataset, subdataset, measuref, svpath, fprs, orig_path)
end
