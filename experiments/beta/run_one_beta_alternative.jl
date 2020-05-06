include("test_beta_alternatives_functions.jl")

svpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/beta_alternatives/hist_auc_v2"
orig_path = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
fprs = collect(range(0.01,0.1, length=10))

measuref = hist_auc
dataset = ARGS[1]
subsets = get_subsets(dataset)
for subdataset in subsets
	results = save_measure_test_results(dataset, subdataset, measuref, svpath, fprs, orig_path; 
        nsamples=1000, throw_errs=true)
end

# cat full_list | xargs -n 1 -P 48 ./run_one_beta_alternative.sh
