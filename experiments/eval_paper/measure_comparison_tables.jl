using PaperUtils
using ADMetricEvaluation
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

function create_df(mean_df, measures, measure_names, sd_df=nothing; colmeans=false,
	percents=false, shadingdf=nothing, sep_last=false, submeans=false)
	df = deepcopy(mean_df)
	# filter the appropriate columns and rows
	df = df[!,vcat([:measure], measures)]
	irows = map(x->x in string.(measures), df[!,:measure])
	df = df[irows,:]
	measure_inds = 1:length(measures)
	if colmeans && submeans
		measure_inds = measure_inds[1:end-3]
	elseif colmeans
		measure_inds = measure_inds[1:end-1]
	end
	for (i,measure) in enumerate(measures)
		if sd_df!=nothing
			df[!,measure] = df[!,measure].*"\$\\pm\$".*sd_df[!,measure][irows]
		end
		if percents
			df[!,measure].*="\\%"
		end
		# add shading to the best and second best row
		if shadingdf!=nothing
			sortis = sortperm(shadingdf[!,measure][irows])
			if colmeans && submeans && i in length(measures)-2:length(measures)
				df[!,measure][sortis[1]] = "\\cellcolor{gray!45}"*df[!,measure][sortis[1]]
				df[!,measure][sortis[2]] = "\\cellcolor{gray!30}"*df[!,measure][sortis[2]]				
				df[!,measure][sortis[3]] = "\\cellcolor{gray!15}"*df[!,measure][sortis[3]]
			elseif colmeans && i==length(measures)
				df[!,measure][sortis[1]] = "\\cellcolor{gray!45}"*df[!,measure][sortis[1]]
				df[!,measure][sortis[2]] = "\\cellcolor{gray!30}"*df[!,measure][sortis[2]]				
				df[!,measure][sortis[3]] = "\\cellcolor{gray!15}"*df[!,measure][sortis[3]]
			else
				df[!,measure][sortis[2]] = "\\cellcolor{gray!45}"*df[!,measure][sortis[2]]
				df[!,measure][sortis[3]] = "\\cellcolor{gray!30}"*df[!,measure][sortis[3]]
				df[!,measure][sortis[4]] = "\\cellcolor{gray!15}"*df[!,measure][sortis[4]]				
			end
		end
		if submeans && i <= length(measure_inds)-2
			df[!,measure][i] = "--"
		elseif i <= length(measure_inds)
			df[!,measure][i] = "--"
		end
	end

	df[!,:measure] = measure_names[measure_inds]
	rename!(df, :measure => Symbol("max/loss"))
	# rename the colums as well
	for (old,new) in zip(measures, measure_names)
		rename!(df, old => Symbol(new))
	end
	# also, if sep_last, put an empty column before the one with means
	if sep_last && colmeans && submeans
		insertcols!(df, size(df,2)-2, :emptycol=>"")
	elseif sep_last && colmeans
		insertcols!(df, size(df,2), :emptycol=>"")
	end
	return df
end

function construct_tex_tables(data_path, measures, measure_names, filename,
		caption, label; show_sd=false, colmeans = false, group_measures=false, 
		sep_last=false, models = ["kNN", "LOF", "IF", "OCSVM"], allsubdatasets=true,
		submeans = false,
		 df2tex_kwargs...)
	if data_path == ""
		return nothing, nothing, nothing
	end
	mean_diff, sd_diff, rel_mean_diff, rel_sd_diff = 
		ADMetricEvaluation.compare_measures_model_is_parameter(data_path, measures; 
			models = models, 
			allsubdatasets = allsubdatasets)
	ndecimal = 3
	if colmeans
		map(X->X[!,:mean]=ADMetricEvaluation.matrix_col_nan_mean(convert(Matrix, X[!,2:end])), [mean_diff, sd_diff, 
			rel_mean_diff, rel_sd_diff])
		measures = push!(copy(measures), :mean)
		measure_names = push!(copy(measure_names), "mean")
	end
	if submeans
		map(X->X[!,:mean_at_1]=ADMetricEvaluation.matrix_col_nan_mean(convert(Matrix, 
			X[!,[:auc_at_1,:tpr_at_1, :prec_at_1, :f1_at_1]])), [mean_diff, sd_diff, 
			rel_mean_diff, rel_sd_diff])
		map(X->X[!,:mean_at_5]=ADMetricEvaluation.matrix_col_nan_mean(convert(Matrix, 
			X[!,[:auc_at_5,:tpr_at_5, :prec_at_5, :f1_at_5]])), [mean_diff, sd_diff, 
			rel_mean_diff, rel_sd_diff])
		measures = push!(copy(measures), :mean_at_1)
		measure_names = push!(copy(measure_names), "mean@0.01")
		measures = push!(copy(measures), :mean_at_5)
		measure_names = push!(copy(measure_names), "mean@0.05")
	end
	cols = 2:(1+length(measures))
	ndecimal = 1
	mean_diff_rounded = PaperUtils.round_string_rpad(mean_diff, ndecimal, cols)
	sd_diff_rounded = PaperUtils.round_string_rpad(sd_diff, ndecimal, cols)
	map(x->rel_mean_diff[!,x]=rel_mean_diff[!,x]*100, measures)
	map(x->rel_sd_diff[!,x]=rel_sd_diff[!,x]*100, measures)
	rel_mean_diff_rounded = PaperUtils.round_string_rpad(rel_mean_diff, ndecimal, cols)
	rel_sd_diff_rounded = PaperUtils.round_string_rpad(rel_sd_diff, ndecimal, cols)
	if show_sd 
		abs_df = create_df(mean_diff_rounded, measures, measure_names, sd_diff_rounded; 
			colmeans=colmeans, sep_last=sep_last, submeans=submeans)
		rel_df = create_df(rel_mean_diff_rounded, measures, measure_names, rel_sd_diff_rounded;
	 		colmeans=colmeans, percents = true, shadingdf=rel_mean_diff, sep_last=sep_last,
	 		submeans=submeans)
	else
		abs_df = create_df(mean_diff_rounded, measures, measure_names; 
			colmeans=colmeans, sep_last=sep_last, submeans=submeans)
		rel_df = create_df(rel_mean_diff_rounded, measures, measure_names;
	 		colmeans=colmeans, percents = true, shadingdf=rel_mean_diff, sep_last=sep_last,
	 		submeans=submeans)
	end
	#global fname = joinpath(savepath, filename1)
	#abs_s = PaperUtils.df2tex(abs_df, caption1; label = label1, df2tex_kwargs...)
	#PaperUtils.string2file(fname, abs_s)

	fname = joinpath(savepath, filename)
	rel_s = PaperUtils.df2tex(rel_df, caption; label = label, df2tex_kwargs...)
	# remove the header of the empty col
	if sep_last
		rel_s = replace(rel_s, "emptycol"=>"")
	end
	PaperUtils.string2file(fname, rel_s)
	
	return abs_df, rel_df, rel_s
end


if !bootstrapping && !discriminability
	savepath = "."
	path5 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_f1_contaminated-0.05"
	path0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_f1_contaminated-0.00"
	path1 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_f1_contaminated-0.01"

	measures = [:auc, :auc_weighted, :auc_at_5, :auc_at_1, :prec_at_5, :prec_at_1, 
			:tpr_at_5, :tpr_at_1, :f1_at_5, :f1_at_1, :vol_at_5, :vol_at_1]
	measure_names = ["AUC", "AUC\$_w\$", "AUC@0.05", "AUC@0.01", "precision@0.05", "precision@0.01", 
			"TPR@0.05", "TPR@0.01", "F1@0.05", "F1@0.01", "CVOL@0.05", "CVOL@0.01"]

	abs_df_full_5, rel_df_full_5, rel_s_full_5 = construct_tex_tables(
			path5,
			measures,
			measure_names,
			"table_measure_comparison_full_5_by_models.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. 5\\% training contamination.",
			"tab:measure_comparison_full_5_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_full_0, rel_df_full_0, rel_s_full_0 = construct_tex_tables(
			path0,
			measures,
			measure_names,
			"table_measure_comparison_full_0_by_models.tex", 
			"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 0\\%  training contamination. Level of shading highlights three best results in a column.",
			"tab:measure_comparison_full_0_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_full_1, rel_df_full_1, rel_s_full_1 = construct_tex_tables(
			path1,
			measures,
			measure_names,
			"table_measure_comparison_full_1_by_models.tex", 
			"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 1\\% contamination.",
			"tab:measure_comparison_full_1_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)
elseif discriminability
	savepath = "/home/vit/vyzkum/measure_evaluation/discriminability"
	path_umap = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_joined"
	path5 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.05_joined"
	path0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_joined"
	path1 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.01_joined"
	
	base_measures = [:auc, :auc_weighted, :auc_at_1, :auc_at_5,
		 :tpr_at_1, :tpr_at_5, :prec_at_1, :prec_at_5, 
		 :f1_at_1, :f1_at_5#, :vol_at_1, :vol_at_5
	 	 ]
	base_measure_names = ["AUC", "AUC\$_w\$", "AUC@0.01", "AUC@0.05",
		"TPR@0.01", "TPR@0.05", "precision@0.01", "precision@0.05",
		"F1@0.01", "F1@0.05"#, "CVOL@0.01", "CVOL@0.05"		
		]

	edit_crit(x) = replace(x, "_"=>"-")
	for crit in ["tukey_q", "tukey_mean", "tukey_median", "welch_mean", "welch_median"]
		measures = vcat(base_measures, Symbol.([crit*"_auc_at", crit*"_tpr_at"]))
		ecrit = edit_crit(crit)
		measure_names = vcat(base_measure_names, ["AUC@$ecrit", "TPR@$ecrit"])

		abs_df_umap_0, rel_df_umap_0, rel_s_umap_0 = construct_tex_tables(
				path_umap,
				measures,
				measure_names,
				"table_measure_comparison_umap_0_by_models_$(crit).tex", 
				"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination.",
				"tab:measure_comparison_umap_0_by_models_$(crit)"; 
				colmeans=true, sep_last=true, submeans = true,
				asterisk = true, fittext=true, vertcolnames=true 
				)
				
		abs_df_full_5, rel_df_full_5, rel_s_full_5 = construct_tex_tables(
				path5,
				measures,
				measure_names,
				"table_measure_comparison_full_5_by_models_$(crit).tex", 
				"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. 5\\% training contamination.",
				"tab:measure_comparison_full_5_by_models_$(crit)"; 
				colmeans=true, sep_last=true, submeans = true,
				asterisk = true, fittext=true, vertcolnames=true)

		abs_df_full_0, rel_df_full_0, rel_s_full_0 = construct_tex_tables(
				path0,
				measures,
				measure_names,
				"table_measure_comparison_full_0_by_models_$(crit).tex", 
				"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 0\\%  training contamination. Level of shading highlights three best results in a column.",
				"tab:measure_comparison_full_0_by_models_$(crit)"; 
				colmeans=true, sep_last=true, submeans = true,
				asterisk = true, fittext=true, vertcolnames=true)

		abs_df_full_1, rel_df_full_1, rel_s_full_1 = construct_tex_tables(
				path1,
				measures,
				measure_names,
				"table_measure_comparison_full_1_by_models_$(crit).tex", 
				"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 1\\% contamination.",
				"tab:measure_comparison_full_1_by_models_$(crit)"; 
				colmeans=true, sep_last=true, submeans = true,
				asterisk = true, fittext=true, vertcolnames=true)
	end
else
	savepath = "/home/vit/vyzkum/measure_evaluation/bootstrapping"
	path_umap = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_bootstrapping_contaminated-0.00"
	path5 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_bootstrapping_contaminated-0.05"
	path0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_bootstrapping_contaminated-0.00"
	path1 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_bootstrapping_contaminated-0.01"

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
	measure_names = ["AUC", "AUC-BS", "AUC-GMM", "AUC-GMM-5k", "AUC\$_w\$", 
		"AUC@0.05", "AUC@0.05-BS", "AUC@0.05-GMM", "AUC@0.05-GMM-5k",
		"precision@0.05", 
		"TPR@0.05", "TPR@0.05-BS", "TPR@0.05-GMM", "TPR@0.05-GMM-5k",
		"F1@0.05", "CVOL@0.05",
		"AUC@0.01", "AUC@0.01-BS", "AUC@0.01-GMM", "AUC@0.01-GMM-5k",
		"precision@0.01", 
		"TPR@0.01", "TPR@0.01-BS", "TPR@0.01-GMM", "TPR@0.01-GMM-5k",
		"F1@0.01", "CVOL@0.01"]

	abs_df_umap_0, rel_df_umap_0, rel_s_umap_0 = construct_tex_tables(
			path_umap,
			measures,
			measure_names,
			"table_measure_comparison_umap_0_by_models.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination.",
			"tab:measure_comparison_umap_0_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)
			
	abs_df_full_5, rel_df_full_5, rel_s_full_5 = construct_tex_tables(
			path5,
			measures,
			measure_names,
			"table_measure_comparison_full_5_by_models.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. 5\\% training contamination.",
			"tab:measure_comparison_full_5_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_full_0, rel_df_full_0, rel_s_full_0 = construct_tex_tables(
			path0,
			measures,
			measure_names,
			"table_measure_comparison_full_0_by_models.tex", 
			"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 0\\%  training contamination. Level of shading highlights three best results in a column.",
			"tab:measure_comparison_full_0_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_full_1, rel_df_full_1, rel_s_full_1 = construct_tex_tables(
			path1,
			measures,
			measure_names,
			"table_measure_comparison_full_1_by_models.tex", 
			"Means of relative loss in a column measure when optimal model and hyperparameters are selected using the row measure. 1\\% contamination.",
			"tab:measure_comparison_full_1_by_models"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_umap_0_nobs, rel_df_umap_0_nobs, rel_s_umap_0_nobs = construct_tex_tables(
			path_umap,
			[:auc, :auc_weighted, 
				 :auc_at_5, 
				 :prec_at_5, 
				 :tpr_at_5, 
				 :f1_at_5, :vol_at_5,
			 	 :auc_at_1, 
			 	 :prec_at_1, 
			 	 :tpr_at_1,
			 	 :f1_at_1, :vol_at_1
			 	 ],
			["AUC", "AUC\$_w\$", 
				"AUC@0.05", 
				"precision@0.05", 
				"TPR@0.05", 
				"F1@0.05", "CVOL@0.05",
				"AUC@0.01", 
				"precision@0.01", 
				"TPR@0.01", 
				"F1@0.01", "CVOL@0.01"],
			"table_measure_comparison_umap_0_by_models_nobs.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination. No bootstrapping.",
			"tab:measure_comparison_umap_0_by_models_nobs"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_umap_0_5k, rel_df_umap_0_5k, rel_s_umap_0_5k = construct_tex_tables(
			path_umap,
			[:auc_gmm_5000, :auc_weighted, 
			 :auc_at_5_gmm_5000,
			 :prec_at_5, 
			 :tpr_at_5_gmm_5000,
			 :f1_at_5, :vol_at_5,
		 	 :auc_at_1_gmm_5000,
		 	 :prec_at_1, 
		 	 :tpr_at_1_gmm_5000,
		 	 :f1_at_1, :vol_at_1
		 	 ],
			["AUC-GMM-5k", "AUC\$_w\$", 
				"AUC@0.05-GMM-5k",
				"precision@0.05", 
				"TPR@0.05-GMM-5k",
				"F1@0.05", "CVOL@0.05",
				"AUC@0.01-GMM-5k",
				"precision@0.01", 
				"TPR@0.01-GMM-5k",
				"F1@0.01", "CVOL@0.01"],
			"table_measure_comparison_umap_0_by_models_5k.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination. GMM bootstrapping with 5k samples.",
			"tab:measure_comparison_umap_0_by_models_5k"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_umap_0_gmm, rel_df_umap_0_gmm, rel_s_umap_0_gmm = construct_tex_tables(
			path_umap,
			[:auc_gmm, :auc_weighted, 
			 :auc_at_5_gmm,
			 :prec_at_5, 
			 :tpr_at_5_gmm,
			 :f1_at_5, :vol_at_5,
		 	 :auc_at_1_gmm,
		 	 :prec_at_1, 
		 	 :tpr_at_1_gmm,
		 	 :f1_at_1, :vol_at_1
		 	 ],
			["AUC-GMM", "AUC\$_w\$", 
				"AUC@0.05-GMM",
				"precision@0.05", 
				"TPR@0.05-GMM",
				"F1@0.05", "CVOL@0.05",
				"AUC@0.01-GMM",
				"precision@0.01", 
				"TPR@0.01-GMM",
				"F1@0.01", "CVOL@0.01"],
			"table_measure_comparison_umap_0_by_models_gmm.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination. GMM bootstrapping.",
			"tab:measure_comparison_umap_0_by_models_gmm"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

	abs_df_umap_0_bs, rel_df_umap_0_bs, rel_s_umap_0_bs = construct_tex_tables(
			path_umap,
			[:auc_bs, :auc_weighted,
			 :auc_at_5_bs,
			 :prec_at_5, 
			 :tpr_at_5_bs,
			 :f1_at_5, :vol_at_5,
		 	 :auc_at_1_bs,
		 	 :prec_at_1, 
		 	 :tpr_at_1_bs,
		 	 :f1_at_1, :vol_at_1
		 	 ],
			["AUC-bs", "AUC\$_w\$", 
				"AUC@0.05-bs",
				"precision@0.05", 
				"TPR@0.05-bs",
				"F1@0.05", "CVOL@0.05",
				"AUC@0.01-bs",
				"precision@0.01", 
				"TPR@0.01-bs",
				"F1@0.01", "CVOL@0.01"],
			"table_measure_comparison_umap_0_by_models_bs.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination. Normal bootstrapping.",
			"tab:measure_comparison_umap_0_by_models_bs"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)

measures = 	[:auc, :auc_bs, :auc_gmm, :auc_weighted, 
		 :auc_at_5, :auc_at_5_bs, :auc_at_5_gmm, 
		 :prec_at_5, 
		 :tpr_at_5, :tpr_at_5_bs, :tpr_at_5_gmm, 
		 :f1_at_5, :vol_at_5,
	 	 :auc_at_1, :auc_at_1_bs, :auc_at_1_gmm, 
	 	 :prec_at_1, 
	 	 :tpr_at_1, :tpr_at_1_bs, :tpr_at_1_gmm, 
	 	 :f1_at_1, :vol_at_1
	 	 ]
	measure_names = ["AUC", "AUC-BS", "AUC-GMM", "AUC\$_w\$", 
		"AUC@0.05", "AUC@0.05-BS", "AUC@0.05-GMM", 
		"precision@0.05", 
		"TPR@0.05", "TPR@0.05-BS", "TPR@0.05-GMM", 
		"F1@0.05", "CVOL@0.05",
		"AUC@0.01", "AUC@0.01-BS", "AUC@0.01-GMM", 
		"precision@0.01", 
		"TPR@0.01", "TPR@0.01-BS", "TPR@0.01-GMM", 
		"F1@0.01", "CVOL@0.01"]

	abs_df_umap_0, rel_df_umap_0, rel_s_umap_0 = construct_tex_tables(
			path_umap,
			measures,
			measure_names,
			"table_measure_comparison_umap_0_by_models_bs_vs_gmm.tex", 
			"Means of relative performance loss in a column measure when optimal model and hyperparameters are selected using the row measure. UMAP dataset, 0\\% training contamination.",
			"tab:measure_comparison_umap_0_by_models_bs_vs_gmm"; 
	#		models = ["kNN", "IF", "LOF"],
			colmeans=true, sep_last=true,
			asterisk = true, fittext=true, vertcolnames=true)
end
