using PaperUtils
using ADMetricEvaluation
using DataFrames
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--bootstrapping"
		action = :store_true
        help = "compute only the bootstrapping tables"
end
parsed_args = parse_args(ARGS, s)
bootstrapping = parsed_args["bootstrapping"]

function construct_tex_table(data_path, texfile, metrics, metric_labels, texlabel, texcaption)
	if data_path == ""
		return nothing, nothing
	end
	ranks_mean, ranks_sd = ADMetricEvaluation.model_ranks_stats(data_path, metrics)

	ndecimal = 2
	ranks_mean_rounded = PaperUtils.rpaddf(PaperUtils.cols2string(PaperUtils.rounddf(ranks_mean, ndecimal, 2:5)), ndecimal)
	ranks_sd_rounded = PaperUtils.rpaddf(PaperUtils.cols2string(PaperUtils.rounddf(ranks_sd, ndecimal, 2:5)), ndecimal)
	ranks_df = deepcopy(ranks_mean_rounded)
	for model in [:kNN, :LOF, :IF, :OCSVM]
		ranks_df[!,model] = ranks_df[!,model].*"\$\\pm\$".*ranks_sd_rounded[!,model]
	end
	# rename first col and measures
	rename!(ranks_df, :metric => :measure)
	ranks_df[!,:measure] = metric_labels
	# finally, swap rows to correspond to the paper order
	#ranks_df = ranks_df[[1,2,3,5,4,6],:]
	# and save the table to a texfile
	global fname = joinpath(savepath, texfile)
	ranks_s = PaperUtils.df2tex(ranks_df, texcaption; label = texlabel, pos = "h", align = "c",
	    fitcolumn = false, lasthline = false, firstvline = false)
	PaperUtils.string2file(fname, ranks_s)
	return ranks_df, ranks_s
end

if !bootstrapping
	savepath = "."
	#savepath = "/home/vit/Dropbox/Cisco/metric_evaluation_paper/dmkd_journal"
	umap_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_f1_contaminated-0.00"
	full_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_f1_contaminated-0.00"

	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :f1_at_5, :vol_at_5,
	 :auc_at_1, :prec_at_1, :tpr_at_1, :f1_at_1, :vol_at_1]
	metric_labels = ["AUC", "AUC\$_w\$", "AUC@0.05", "precision@0.05", "TPR@0.05", "F1@0.05", "CVOL@0.05",
			 "AUC@0.01", "precision@0.01", "TPR@0.01", "F1@0.01", "CVOL@0.01"]

	#metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5,
	# :auc_at_1, :prec_at_1, :tpr_at_1, :vol_at_1]
	#metric_labels = ["AUC", "AUC\$_w\$", "AUC@0.05", "precision@0.05", "TPR@0.05", "CVOL@0.05",
	#		"AUC@0.01", "precision@0.01", "TPR@0.01", "CVOL@0.01"]

	ranks_umap_df, ranks_umap_s = 
		construct_tex_table(
			umap_path, 
			"table_model_ranks_umap_0.tex", metrics, metric_labels, "tab:model_ranks_umap_0", 
			"Means and standard deviations of algorithm ranks using different measures, UMAP datasets, 0\\% training contamination.")
	println("succesfuly exported the UMAP model ranks table")

	ranks_df, ranks_s = 
		construct_tex_table(
			full_path, 
			"table_model_ranks_0.tex", metrics, metric_labels, "tab:model_ranks_0", 
			"Means and standard deviations of algorithm ranks using different measures, 0\\% training contamination.")
	println("succesfuly exported the full model ranks table")

else
	savepath = "/home/vit/vyzkum/measure_evaluation/bootstrapping"
	umap_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_bootstrapping_contaminated-0.00"
	full_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_bootstrapping_contaminated-0.00"
#	full_path = ""

	metrics = [:auc, :auc_bs, :auc_gmm, :auc_gmm_5000, :auc_weighted, 
		 :auc_at_5, :auc_at_5_bs, :auc_at_5_gmm, :auc_at_5_gmm_5000,
		 :prec_at_5, 
		 :tpr_at_5, :tpr_at_5_bs, :tpr_at_5_gmm, :tpr_at_5_gmm_5000,
		 :f1_at_5, :vol_at_5,
	 	 :auc_at_1, :auc_at_1_bs, :auc_at_1_gmm, :auc_at_1_gmm_5000,
	 	 :prec_at_1, 
	 	 :tpr_at_1, :tpr_at_1_bs, :tpr_at_1_gmm, :tpr_at_1_gmm_5000,
	 	 :f1_at_1, :vol_at_1
	 	 ]

	metric_labels = ["AUC", "AUC-BS", "AUC-GMM", "AUC-GMM-5k", "AUC\$_w\$", 
		"AUC@0.05", "AUC@0.05-BS", "AUC@0.05-GMM", "AUC@0.05-GMM-5k",
		"precision@0.05", 
		"TPR@0.05", "TPR@0.05-BS", "TPR@0.05-GMM", "TPR@0.05-GMM-5k",
		"F1@0.05", "CVOL@0.05",
		"AUC@0.01", "AUC@0.01-BS", "AUC@0.01-GMM", "AUC@0.01-GMM-5k",
		"precision@0.01", 
		"TPR@0.01", "TPR@0.01-BS", "TPR@0.01-GMM", "TPR@0.01-GMM-5k",
		"F1@0.01", "CVOL@0.01"]

	ranks_umap_df, ranks_umap_s = 
		construct_tex_table(
			umap_path, 
			"table_model_ranks_umap_0.tex", metrics, metric_labels, "tab:model_ranks_umap_0", 
			"Means and standard deviations of algorithm ranks using different measures, UMAP datasets, 0\\% training contamination.")
	println("succesfuly exported the UMAP model ranks table")

	ranks_df, ranks_s = 
		construct_tex_table(
			full_path, 
			"table_model_ranks_0.tex", metrics, metric_labels, "tab:model_ranks_0", 
			"Means and standard deviations of algorithm ranks using different measures, 0\\% training contamination.")
	println("succesfuly exported the full model ranks table")
end