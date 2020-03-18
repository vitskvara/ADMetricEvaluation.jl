using ADMetricEvaluation
using PaperUtils
using DataFrames
using Statistics

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--bootstrapping"
		action = :store_true
        help = "compute only the bootstrapping tables"
    "--discriminability"
		action = :store_true
        help = "compute only the discriminability tables"
end
parsed_args = parse_args(ARGS, s)
bootstrapping = parsed_args["bootstrapping"]
discriminability = parsed_args["discriminability"]

function create_tex_table(data_path, measures, measure_dict, filename, label, caption; 
	shading=false, column_mean = false, sep_last=false, df2texkwargs...)
	ndecimal = 2
	cordf = ADMetricEvaluation.global_measure_correlation(data_path, measures, average_folds = false)
	df = deepcopy(cordf)
	# filter the appropriate columns and rows and change their order
	df = df[!,vcat([:measure], measures)]
	irows = map(x->x in string.(measures), df[!,:measure])
	df = df[irows,:]

	# rename the rows and columns
	for (msym, mstr) in measure_dict
		df[!,:measure][df[!,:measure].== string(msym)] .= mstr
		rename!(df, msym=>Symbol(mstr))
	end

	# compute the column mean
	if column_mean
		df[!,:mean] = reshape(Statistics.mean(convert(Matrix, df[!,2:end]), dims=2), size(df,1))
	end

	cols = 2:length(measures)+ (column_mean ? 2 : 1)
	df_str = PaperUtils.round_string_rpad(df, ndecimal, cols)

	# add shading to the best and second best row
	if shading
		for (i, measure) in enumerate(filter!(x->x!=:measure, names(df)))
			sortis = sortperm(df[!,measure], rev=true)
			if column_mean && i==size(df_str,2)-1
				df_str[!,measure][sortis[1]] = "\\cellcolor{gray!45}"*df_str[!,measure][sortis[1]]
				df_str[!,measure][sortis[2]] = "\\cellcolor{gray!30}"*df_str[!,measure][sortis[2]]				
				df_str[!,measure][sortis[3]] = "\\cellcolor{gray!15}"*df_str[!,measure][sortis[3]]				
			else
				df_str[!,measure][sortis[2]] = "\\cellcolor{gray!45}"*df_str[!,measure][sortis[2]]
				df_str[!,measure][sortis[3]] = "\\cellcolor{gray!30}"*df_str[!,measure][sortis[3]]
				df_str[!,measure][sortis[4]] = "\\cellcolor{gray!15}"*df_str[!,measure][sortis[4]]
			end
		end
	end

	# also put dashes on the diagonal
	for i in 1:length(measures)
		df_str[i,i+1] = "--"
	end

	# separate last column with mean
	if column_mean && sep_last
		insertcols!(df_str, size(df,2), :emptycol=>"")
	end

	fname = joinpath(savepath, filename)
	table_str = PaperUtils.df2tex(df_str, caption; label = label, asterisk=true, 
		vertcolnames=true, df2texkwargs...)
	# remove the header of the empty col
	if sep_last
		table_str = replace(table_str, "emptycol"=>"")
	end
	PaperUtils.string2file(fname, table_str)
	return df, df_str, table_str
end

if !bootstrapping && !discriminability
	savepath = "."
	datapath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_f1_contaminated-0.00"

	measure_dict = Dict(zip(
			[:auc, :auc_weighted, :auc_at_5, :auc_at_1, 
			:prec_at_5, :prec_at_1, :tpr_at_5, :tpr_at_1, 
			:f1_at_5, :f1_at_1, :vol_at_5, :vol_at_1],
			["AUC", "AUC\$_w\$", "AUC@0.05", "AUC@0.01", 
			"precision@0.05", "precision@0.01", 
			"TPR@0.05", "TPR@0.01", "F1@0.05", "F1@0.01", "CVOL@0.05", "CVOL@0.01"]
			))
	measures = 		
			[:auc, :auc_weighted, :auc_at_5, :auc_at_1, 
			:prec_at_5, :prec_at_1, :tpr_at_5, :tpr_at_1, 
			:f1_at_5, :f1_at_1,:vol_at_5, :vol_at_1]


	#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
	#cordf = ADMetricEvaluation.global_measure_correlation(data_path)

	
	cordf_full, cordf_str_full_0, cordf_tex_str_full_0 = create_tex_table(
		datapath,
		measures, measure_dict,
		"table_measure_correlation_full_0.tex",
		"tab:measure_correlation_full_0",
		"Average of Kendall correlation between measures over datasets, 0\\% contamination. Level of shading highlights three highest correlations in a column.";
		column_mean = true, shading = true, sep_last=true
	)

	println("measure correlations for FULL data exported to TEX.")
elseif discriminability
	savepath = "/home/vit/vyzkum/measure_evaluation/discriminability"
	datapath_umap = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_joined"
	datapath_full = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_joined"

	base_measures = [:auc, :auc_weighted, :auc_at_1, :auc_at_5,
		 :tpr_at_1, :tpr_at_5, :prec_at_1, :prec_at_5,
		 :f1_at_1, :f1_at_5, :vol_at_1, :vol_at_5
	 	 ]
	base_measure_labels = ["AUC", "AUC\$_w\$", "AUC@0.01", "AUC@0.05",
		"TPR@0.01", "TPR@0.05", "precision@0.01", "precision@0.05",
		"F1@0.01", "F1@0.05", "CVOL@0.01", "CVOL@0.05"
		]

	edit_crit(x) = replace(x, "_"=>"-")
	measures = vcat(base_measures, vcat(map(x->Symbol.([x*"_auc_at", x*"_tpr_at"]),
		["tukey_q", "tukey_mean", "tukey_median", "welch_mean", "welch_median"])...))
	measure_labels = vcat(base_measure_labels, 
		vcat(map(x->["AUC@$(edit_crit(x))", "TPR@$(edit_crit(x))"],
		["tukey_q", "tukey_mean", "tukey_median", "welch_mean", "welch_median"])...))

	measure_dict = Dict(zip(measures, measure_labels))

	cordf_umap, cordf_str_umap_0, cordf_tex_str_umap_0 = create_tex_table(
		datapath_umap,
		measures, measure_dict,
		"table_measure_correlation_umap_0.tex",
		"tab:measure_correlation_umap_0",
		"Average of Kendall correlation between measures over UMAP datasets, 0\\% contamination. Level of shading highlights three highest correlations in a column.";
		column_mean = true, shading = true, sep_last=true, fittext = true
	)
	println("measure correlations for UMAP data exported to TEX.")
	
	cordf_full, cordf_str_full_0, cordf_tex_str_full_0 = create_tex_table(
		datapath_full,
		measures, measure_dict,
		"table_measure_correlation_full_0.tex",
		"tab:measure_correlation_full_0",
		"Average of Kendall correlation between measures over datasets, 0\\% contamination. Level of shading highlights three highest correlations in a column.";
		column_mean = true, shading = true, sep_last=true, fittext = true
	)
	println("measure correlations for FULL data exported to TEX.")
else
	savepath = "/home/vit/vyzkum/measure_evaluation/bootstrapping"
	datapath_umap = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_bootstrapping_contaminated-0.00"
	datapath_full = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_bootstrapping_contaminated-0.00"

	measures = 	[:auc, :auc_bs, :auc_gmm, :auc_gmm_5000, :auc_weighted, 
		 :auc_at_5, :auc_at_5_bs, :auc_at_5_gmm, :auc_at_5_gmm_5000,
		 :prec_at_5, 
		 :tpr_at_5, :tpr_at_5_bs, :tpr_at_5_gmm, :tpr_at_5_gmm_5000,
		 :f1_at_5, :vol_at_5,
	 	 :auc_at_1, :auc_at_1_bs, :auc_at_1_gmm, :auc_at_1_gmm_5000,
	 	 :prec_at_1, 
	 	 :tpr_at_1, :tpr_at_1_bs, :tpr_at_1_gmm, :tpr_at_1_gmm_5000,
	 	 :f1_at_1, :vol_at_1
	 	 ]
	measure_dict = Dict(zip(measures,
			["AUC", "AUC-BS", "AUC-GMM", "AUC-GMM-5k", "AUC\$_w\$", 
		"AUC@0.05", "AUC@0.05-BS", "AUC@0.05-GMM", "AUC@0.05-GMM-5k",
		"precision@0.05", 
		"TPR@0.05", "TPR@0.05-BS", "TPR@0.05-GMM", "TPR@0.05-GMM-5k",
		"F1@0.05", "CVOL@0.05",
		"AUC@0.01", "AUC@0.01-BS", "AUC@0.01-GMM", "AUC@0.01-GMM-5k",
		"precision@0.01", 
		"TPR@0.01", "TPR@0.01-BS", "TPR@0.01-GMM", "TPR@0.01-GMM-5k",
		"F1@0.01", "CVOL@0.01"]
			))

	#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"
	#cordf = ADMetricEvaluation.global_measure_correlation(data_path)

	cordf_umap, cordf_str_umap_0, cordf_tex_str_umap_0 = create_tex_table(
		datapath_umap,
		measures, measure_dict,
		"table_measure_correlation_umap_0.tex",
		"tab:measure_correlation_umap_0",
		"Average of Kendall correlation between measures over UMAP datasets, 0\\% contamination. Level of shading highlights three highest correlations in a column.";
		column_mean = true, shading = true, sep_last=true, fittext = true
	)
	println("measure correlations for UMAP data exported to TEX.")
	
	cordf_full, cordf_str_full_0, cordf_tex_str_full_0 = create_tex_table(
		datapath_full,
		measures, measure_dict,
		"table_measure_correlation_full_0.tex",
		"tab:measure_correlation_full_0",
		"Average of Kendall correlation between measures over datasets, 0\\% contamination. Level of shading highlights three highest correlations in a column.";
		column_mean = true, shading = true, sep_last=true, fittext = true
	)
	println("measure correlations for FULL data exported to TEX.")
end