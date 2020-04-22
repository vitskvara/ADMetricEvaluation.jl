include("test_beta_alternatives_functions.jl")

svpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/beta_alternatives/bs_auc"
orig_path = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
fprs = [0.01, 0.02, 0.05, 0.1]

measuref = auc_at_fpr_bootstrap
dataset = ARGS[1]
subsets = get_subsets(dataset)
for subdataset in subsets
	results = save_measure_test_results(dataset, subdataset, measuref, svpath, fprs, orig_path; 
        nsamples=1000, throw_errs=true)
end

# cat ../bootstrapping/full_list | xargs -n 1 -P 48 ./run_one_beta_alternative.sh