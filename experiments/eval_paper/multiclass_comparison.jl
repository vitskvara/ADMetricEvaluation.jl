using ADMetricEvaluation
using PaperUtils
using Statistics
using CSV
using DataFrames
using DataFramesMeta
using PyPlot
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

basepath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation"
datasets = readdir(joinpath(basepath, "umap_f1_contaminated-0.00"))
multiclass_datasets = readdir(joinpath(basepath, "full_f1_contaminated-0.00"))

function extract_class(fulldataset, dataset)
	sp = split(fulldataset, dataset)
	if length(sp) == 1
		return ""
	else
		return sp[2][2:end]
	end 
end

function prepare_all_df(datapath, measures, datasets)
	alldf = ADMetricEvaluation.collect_fold_averages(datapath, measures)
	insertcols!(alldf, 2, :class=>fill("",size(alldf,1)))

	
	# extract classes adn rename 
	for dataset in datasets
		alldf[!,:class][occursin.(dataset,alldf[!,:dataset])] = map(x->extract_class(x, dataset), alldf[!,:dataset][occursin.(dataset,alldf[!,:dataset])])
		alldf[!,:dataset][occursin.(dataset,alldf[!,:dataset])] .= dataset
	end

	# extract only multiclass rows
	alldf = alldf[map(i->alldf[!,:dataset][i] in multiclass_datasets, collect(1:size(alldf,1))),:]

	datasets = unique(alldf[!,:dataset])
	return alldf
end

function maximize_by_class(alldf, dataset, measures)
	df = @linq alldf |> where(:dataset.==dataset)
	classes = unique(df[:class])
	res_df = DataFrame(:measure=>String[], :class=>String[])
	map(x->res_df[Symbol(x)] = Float64[], classes)
	for measure in measures
		for byclass in classes
			bydf = @linq df |> where(:class.==byclass)
			imax = argmax(bydf[measure])
			maxmodel = bydf[:model][imax]
			maxparams = bydf[:params][imax]
			row = Array{Any,1}()
			push!(row, measure)
			push!(row, byclass)	
			for onclass in classes
				ondf = @linq df |> where(:class.==onclass)
				select_val = ondf[measure][(ondf[:model].==maxmodel).&(ondf[:params].==maxparams)][1]
				max_val = maximum(ondf[measure])
				push!(row, abs(max_val-select_val)/max_val)
			end		
			push!(res_df, row)
		end
	end
	return res_df
end

function maximize_by_class_measure(alldf, dataset, measures)
	df = @linq alldf |> where(:dataset.==dataset)
	classes = unique(df[!,:class])
	res_df = DataFrame(:measure_by=>String[], :measure_on=>String[], 
		:class=>String[])
	map(x->res_df[!,Symbol(x)] = Float64[], classes)
	for bymeasure in measures
		for onmeasure in measures
			for byclass in classes
				bydf = @linq df |> where(:class.==byclass)
				valid_inds = .!isnan.(bydf[!,bymeasure])
				if sum(valid_inds) > 0
					imax = argmax(bydf[!,bymeasure][valid_inds])
					maxmodel = bydf[!,:model][valid_inds][imax]
					maxparams = bydf[!,:params][valid_inds][imax]
					row = Array{Any,1}()
					push!(row, string(bymeasure))
					push!(row, string(onmeasure))
					push!(row, string(byclass))
					for onclass in classes
						ondf = @linq df |> where(:class.==onclass)
						select_val = ondf[!,onmeasure][(ondf[!,:model].==maxmodel).&(ondf[!,:params].==maxparams)] 
						if length(select_val) == 0
							select_val = NaN
						else
							select_val = select_val[1]
						end
						onvals = ondf[!,onmeasure][.!isnan.(ondf[!,onmeasure])]
						if length(onvals) == 0
							max_val = NaN
						else
							max_val = maximum(onvals)
						end
						push!(row, abs(max_val-select_val)/max_val)
					end
					push!(res_df, row)
				end
			end
		end
	end
	return res_df
end

function get_x(df, measure_by, measure_on)
	subdf = @linq df |> where(:measure_by .== string(measure_by), :measure_on .== string(measure_on))
	x = vec(convert(Matrix, subdf[!,4:end]))
	return x[.!isnan.(x)]
end

function create_boxplots(plot_df, fig_meas)
	Nmeas = length(fig_meas)

	f = figure(figsize=(8,10))
	subi = 0
	for rowmeas in fig_meas
		for colmeas in fig_meas
			subi += 1
			subplot(Nmeas, Nmeas, subi)
			X = get_x(plot_df, rowmeas, colmeas)
			boxplot(X)
			ylim([0,1])
			if subi <= Nmeas
				title(colmeas)
			end
			if (subi-1)%Nmeas==0
				ylabel(rowmeas)
			else
				yticks([])
			end			
		end
	end
	tight_layout()
	plt[:subplots_adjust](hspace=0, wspace=0)
	return f
end

function create_boxplots(alldf, dataset, measures, fig_meas)
	plot_df = maximize_by_class_measure(alldf, dataset, measures)
	create_boxplots(plot_df, fig_meas)
	return plot_df
end

function collect_max_dfs_dict(alldf, measures)
	datasets = unique(alldf[!,:dataset])
	return Dict(zip(datasets, map(x->maximize_by_class_measure(alldf, x, measures), datasets)))
end

function create_boxplots_across_datasets(max_dfs_dict, measures, fig_meas)
	datasets = collect(keys(max_dfs_dict))

	Nmeas = length(fig_meas)
	f = figure(figsize=(8,10))
	subi = 0
	for rowmeas in fig_meas
		for colmeas in fig_meas
			subi += 1
			subplot(Nmeas, Nmeas, subi)
			X = vcat(map(dataset-> get_x(max_dfs_dict[dataset], rowmeas, colmeas), datasets)...)
			boxplot(X)
			ylim([0,1])
			if subi <= Nmeas
				title(colmeas)
			end
			if (subi-1)%Nmeas==0
				ylabel(rowmeas)
			else
				yticks([])
			end			
		end
	end
	tight_layout()
	plt[:subplots_adjust](hspace=0, wspace=0)
	return f
end

function compute_means(max_df, measures)
	res_df = DataFrame(:measure=>String[])
	map(x->res_df[x]=Float64[], measures)
	for rowmeas in measures
		row = Array{Any,1}()
		push!(row, string(rowmeas))
		for colmeas in measures
			push!(row, Statistics.mean(get_x(max_df, rowmeas, colmeas)))
		end
		push!(res_df, row)
	end  
	return res_df
end

function compute_means_across_datasets(max_dfs_dict, measures)
	datasets = collect(keys(max_dfs_dict))
	
	res_df = DataFrame(:measure=>String[])
	map(x->res_df[!,x]=Float64[], measures)
	for rowmeas in measures
		row = Array{Any,1}()
		push!(row, string(rowmeas))
		for colmeas in measures
			X = vcat(map(dataset-> get_x(max_dfs_dict[dataset], rowmeas, colmeas), datasets)...)
			push!(row, Statistics.mean(X))
		end
		push!(res_df, row)
	end  
	return res_df
end

function create_str_df(mean_df, measure_names; shade=true)
	df = deepcopy(mean_df)
	ndecimals = 1
	# get to percents
	M,N = size(mean_df)
	for i in 1:M
		for j in 2:N
			df[i,j] *= 100
		end
	end
	str_df = PaperUtils.round_string_rpad(df, ndecimals, 2:size(df,2))
	for i in 1:M
		for j in 2:N
			if str_df[i,j] == "0."*repeat("0",ndecimals)
				str_df[i,j] = "\$<\$0."*repeat("0", ndecimals-1)*"1\\%"
			else
				str_df[i,j] *= "\\%"
			end
		end
	end
	if shade
		for j in 2:N
			x = df[:,j]
			sortis = sortperm(x)
			str_df[sortis[1], j] = "\\cellcolor{gray!45}"*str_df[sortis[1], j]
			str_df[sortis[2], j] = "\\cellcolor{gray!30}"*str_df[sortis[2], j]
			str_df[sortis[3], j] = "\\cellcolor{gray!15}"*str_df[sortis[3], j]
		end
	end
	# finally rename cols and rows
	for (colname, str) in zip(names(str_df)[2:end], measure_names)
		rename!(str_df, colname => Symbol(str))
#		str_df[:measure][str_df[:measure].==string(colname)] .== str
	end
	str_df[!,:measure] = measure_names
	return str_df
end

function create_tex_table(mean_df, outfile, measure_names, label, caption; shade = true, df2tex_kwargs...)
	str_df = create_str_df(mean_df, measure_names; shade=shade)
	tex_str = PaperUtils.df2tex(str_df, caption; label = label, df2tex_kwargs...)
	PaperUtils.string2file(outfile, tex_str)
	return str_df, tex_str
end


if !bootstrapping && !discriminability
	savepath = "."
	#savepath = "/home/vit/Dropbox/Cisco/metric_evaluation_paper/dmkd_journal"
	datasets = readdir(joinpath(basepath, "umap_f1_contaminated-0.00"))
	measures = [:auc, :auc_weighted, :auc_at_5, :auc_at_1, :prec_at_5, :prec_at_1, 
		:tpr_at_5, :tpr_at_1, :f1_at_5, :f1_at_1, :vol_at_5, :vol_at_1]

	# get the datafraes containing all the experiments
	fig_meas = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :f1_at_5, :vol_at_5]
	meas_names = ["AUC", "AUC\$_w\$", "AUC@0.05", "precision@0.05", "TPR@0.05", "F1@0.05", "CVOL@0.05"] 

	#### all datasets umap and full ####### 
	alldf0 = prepare_all_df(joinpath(basepath, "full_f1_contaminated-0.00"), measures, datasets)
	alldf_umap0 = prepare_all_df(joinpath(basepath, "umap_f1_contaminated-0.00"), measures, datasets)
	max_dfs_full0 = collect_max_dfs_dict(alldf0, measures)
	max_dfs_umap0 = collect_max_dfs_dict(alldf_umap0, measures)

	all_means_df_umap = compute_means_across_datasets(max_dfs_umap0, fig_meas)
	all_str_df_umap, all_tex_str_umap = create_tex_table(
		all_means_df_umap, 
		joinpath(savepath, "table_multiclass_all_means_umap_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_umap_0", 
		"All UMAP datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true)

	all_means_df_full = compute_means_across_datasets(max_dfs_full0, fig_meas)
	all_str_df_full, all_tex_str_full = create_tex_table(
		all_means_df_full, 
		joinpath(savepath, "table_multiclass_all_means_full_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_full_0", 
		"All full datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true)
elseif discriminability
	savepath = "/home/vit/vyzkum/measure_evaluation/discriminability"
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

	fig_meas = measures
	meas_names = measure_labels
	
	#### all datasets umap and full ####### 
	alldf0 = prepare_all_df(joinpath(basepath, "full_discriminability_contaminated-0.00_joined"), measures, datasets)
	alldf_umap0 = prepare_all_df(joinpath(basepath, "umap_discriminability_contaminated-0.00_joined"), measures, datasets)
	max_dfs_full0 = collect_max_dfs_dict(alldf0, measures)
	max_dfs_umap0 = collect_max_dfs_dict(alldf_umap0, measures)

	all_means_df_umap = compute_means_across_datasets(max_dfs_umap0, fig_meas)
	all_str_df_umap, all_tex_str_umap = create_tex_table(
		all_means_df_umap, 
		joinpath(savepath, "table_multiclass_all_means_umap_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_umap_0", 
		"All UMAP datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true, fittext=true)

	all_means_df_full = compute_means_across_datasets(max_dfs_full0, fig_meas)
	all_str_df_full, all_tex_str_full = create_tex_table(
		all_means_df_full, 
		joinpath(savepath, "table_multiclass_all_means_full_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_full_0", 
		"All full datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true, fittext=true)
else
	savepath = "/home/vit/vyzkum/measure_evaluation/bootstrapping"
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

	fig_meas = 	[:auc, :auc_bs, :auc_gmm, :auc_gmm_5000, :auc_weighted, 
			 :auc_at_5, :auc_at_5_bs, :auc_at_5_gmm, :auc_at_5_gmm_5000,
			 :prec_at_5, 
			 :tpr_at_5, :tpr_at_5_bs, :tpr_at_5_gmm, :tpr_at_5_gmm_5000,
			 :f1_at_5, :vol_at_5
		 	 ]
	meas_names = ["AUC", "AUC-BS", "AUC-GMM", "AUC-GMM-5k", "AUC\$_w\$", 
		"AUC@0.05", "AUC@0.05-BS", "AUC@0.05-GMM", "AUC@0.05-GMM-5k",
		"precision@0.05", 
		"TPR@0.05", "TPR@0.05-BS", "TPR@0.05-GMM", "TPR@0.05-GMM-5k",
		"F1@0.05", "CVOL@0.05"]

	#### all datasets umap and full ####### 
	alldf0 = prepare_all_df(joinpath(basepath, "full_bootstrapping_contaminated-0.00"), measures, datasets)
	alldf_umap0 = prepare_all_df(joinpath(basepath, "umap_bootstrapping_contaminated-0.00"), measures, datasets)
	max_dfs_full0 = collect_max_dfs_dict(alldf0, measures)
	max_dfs_umap0 = collect_max_dfs_dict(alldf_umap0, measures)

	all_means_df_umap = compute_means_across_datasets(max_dfs_umap0, fig_meas)
	all_str_df_umap, all_tex_str_umap = create_tex_table(
		all_means_df_umap, 
		joinpath(savepath, "table_multiclass_all_means_umap_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_umap_0", 
		"All UMAP datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true)

	all_means_df_full = compute_means_across_datasets(max_dfs_full0, fig_meas)
	all_str_df_full, all_tex_str_full = create_tex_table(
		all_means_df_full, 
		joinpath(savepath, "table_multiclass_all_means_full_0.tex"), 
		meas_names,
		"tab:multiclass_all_means_full_0", 
		"All full datasets, mean of multiclass sensitivities, 0\\% training contamination."; shade = true, 
		vertcolnames=true)
end