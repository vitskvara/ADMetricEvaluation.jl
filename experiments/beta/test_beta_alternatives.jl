include("test_beta_alternatives_functions.jl")
#### sandbox
model_names = ["kNN", "LOF", "OCSVM", "IF"]
model_list = [kNN_model, LOF_model, OCSVM_model, IF_model]
param_struct = [
                ([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
                ([[10, 20, 50]], [:num_neighbors]),
                ([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
                ([[50 100 200]], [:num_estimators]),
             ]

fprs = [0.01, 0.05]
measuref = beta_auc
dataset = "pendigits"
subdataset = "pendigits-4-0"
path = joinpath(savepath, subdataset)
if isdir(path) && length(readdir(path)) > 0
	results = load_results(path, model_names)
else
	results = test_measure(dataset, subdataset, measuref, fprs, model_list, model_names, param_struct)
	save_results(results, path, model_names, subdataset)	
end

orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
original_results = load_results(joinpath(orig_path, dataset), model_names; subdataset = subdataset)

# now join them
dfs_val = map(x -> join_dfs(x[1][1], x[2][1]), zip(original_results, results))
dfs_tst = map(x -> join_dfs(x[1][2], x[2][2]), zip(original_results, results))

# also get the data anyway
raw_data = UCI.get_data(dataset)
multiclass_data = UCI.create_multiclass(raw_data...)
data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]
seed = 1
X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, 0.6, 0.00;
	test_contamination = nothing, seed = seed, standardize=true)
X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);

# yes, the numbers seem to be the same
#df1 = results[1][1]
#df2 = original_results[1][1]
#df1[!,:metric] = string.(df1[!,:metric])

#df = join(df2, df1, on = [:dataset, :model, :metric, :k, :iteration])
#df[!,[:dataset, :model, :iteration, :metric, :k, :auc, :bauc_at_1, :measure_at_1]]

# now compute the measure losses

alldf_val = collect_fold_averages(dfs_val)
alldf_tst = collect_fold_averages(dfs_tst)

full_res_df = rel_measure_loss(alldf_val, alldf_tst, row_measures, column_measures, [1, 5])

df = ADME.compute_measure_loss(alldf_val, alldf_test, row_measures, column_measures)[3]

compute_means(df[3], fprs, column_measures)


