using DataFrames, CSV, Statistics, PyPlot, Distributions
using ADMetricEvaluation
ADME = ADMetricEvaluation

function mean_var_dfs(df, measure)
	# determine some constants
	fprs = ADME.fpr_levels(df, measure)
	nexp = maximum(df[!,:iteration]) # number of experiments

	# get the dfs with means and variances
	colnames = names(df)
	meas_cols = filter(x->occursin(measure, string(x)), colnames)
	subdf = df[!,vcat([:model, :params, :iteration], meas_cols)]

	# now get the actual values
	mean_df = aggregate(subdf, [:model, :params], ADME.nanmean);
	ADME.remove_appendix!(mean_df, meas_cols, "nanmean");
	var_df = aggregate(subdf, [:model, :params], ADME.nanvar);
	ADME.remove_appendix!(var_df, meas_cols, "nanvar");
	mean_vals_df = mean_df[!,meas_cols] # these two only contain the value columns
	var_vals_df = var_df[!,meas_cols]

	return mean_vals_df, var_vals_df, meas_cols, nexp
end

function discriminability_statistics(mean_df, var_df, meas_cols, nexp)
	# get the statistics
	tq, wt_mean, wt_med, tt_mean, tt_med = 
		ADME.stat_lines(mean_df, var_df, meas_cols, nexp)
end

function crit_vals(α, var_vals_df, meas_cols, nexp)
	# get sizes
	nr = size(var_vals_df,1)
	nc = size(var_vals_df,2) # first three columns are not interesting
	
	# tukey q and statistic critval - easy, its the same for all columns
	tqc = ADME.tukey_critval(α, nr, (nr-1)*nexp)

	# parwise tukey and welch statistic
	wcm = zeros(Float32, binomial(nr,2), nc) # welch crit_val matrix
	l = 0
	for i in 1:nr-1
		for j in i+1:nr
			l += 1
			for k in 1:nc
				# welch statistic
				wcm[l,k] = ADME.welch_critval(α, var_vals_df[i,k], var_vals_df[j,k], nexp, nexp)
			end
		end
	end
	
	return tqc, wcm
end

function nanargmax(x) 
	_x = x[.!isnan.(x)]
	length(_x) > 0 ? findfirst(x .== maximum(_x)) : nothing
end

# savepath
savepath = "/home/vit/vyzkum/measure_evaluation/discriminability/plots"

# get the df
master_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_pre"
datasets = readdir(master_path)
dataset = "wine"
inpath = joinpath(master_path, dataset)
alldf,_,_ = ADME.collect_dfs(inpath);

# now filter it down
subsets = unique(alldf[!,:dataset])
subset = subsets[1]

# now compute the statistics for each measure and criteria
measures = ["auc_at", "tpr_at"]
subsetdf = filter(r->r[:dataset]==subset, alldf)

α = 0.05
crit_labels = ["tukey_q", "welch_mean", "welch_median", "tukey_mean", "tukey_median"]
means_vars = map(x->mean_var_dfs(subsetdf,x), measures)
stats = map(x->discriminability_statistics(x[1], x[2], x[3], x[4]), means_vars)
fprs = ADME.fpr_levels(subsetdf, measures[1])
cvals = map(x->crit_vals(α, x[2], x[3], x[4]), means_vars)

figure(figsize=(8,8))
for (j,m) in enumerate(measures)
	for (i,st) in enumerate(stats[j])
		subplot(5,2,2*i - 1 + (j-1))
		title(crit_labels[i]*" "*m)
		plot(fprs, st, label="statistic")
		imax = nanargmax(st)
		if i in [2, 3]
			# welch
			crit_val_vec = vec(ADME.nanrowmean(Array(cvals[j][2])))
			iover = findfirst(crit_val_vec .< st)
			plot(fprs, crit_val_vec, label="crit. val.")
		else
			# tukey
			crit_val = cvals[j][1]
			iover = findfirst(crit_val .< st)
			plot(fprs, repeat([crit_val], length(fprs)), label="crit. val.")
		end
		if iover != nothing
			axvline(fprs[iover], c="brown", linewidth=1)
			yl = ylim()
			text(fprs[iover]+0.02, (yl[2]-yl[1])/2, "$(fprs[iover])")#, c="brown")
		end
		if imax != nothing
			axvline(fprs[imax], c= "k", linewidth=1)
			yl = ylim()
			text(fprs[imax]+0.02, (yl[2]-yl[1])/2, "$(fprs[imax])")#, c="k")
		end
	end
end
tight_layout()
savefig(joinpath(savepath, "$(dataset)-$(subset).png"))

























α = 0.01	
keynames = vcat(map(y->map(x->y*"_"*x, crit_labels), measures)...)
data = Dict()
map(x->data[x]=[], keynames)
for dataset in datasets
	inpath = joinpath(master_path, dataset)
	alldf,_,_ = ADME.collect_dfs(inpath)

	# now filter it down
	subsets = unique(alldf[!,:dataset])
	for subset in subsets
		# now compute the statistics for each measure and criteria
		measures = ["auc_at", "tpr_at"]
		subsetdf = filter(r->r[:dataset]==subset, alldf)

		crit_labels = ["tukey_q", "welch_mean", "welch_median", "tukey_mean", "tukey_median"]
		means_vars = map(x->mean_var_dfs(subsetdf,x), measures)
		stats = map(x->discriminability_statistics(x[1], x[2], x[3], x[4]), means_vars)
		fprs = ADME.fpr_levels(subsetdf, measures[1])
		cvals = map(x->crit_vals(α, x[2], x[3], x[4]), means_vars)

		l = 0
		for (j,m) in enumerate(measures)
			for (i,st) in enumerate(stats[j])
				l += 1		
				if i in [2, 3]
					# welch
					crit_val_vec = vec(ADME.nanrowmean(Array(cvals[j][2])))
					iover = findfirst(crit_val_vec .< st)
				else
					# tukey
					crit_val = cvals[j][1]
					iover = findfirst(crit_val .< st)
				end
				fpr_val = iover == nothing ? NaN : fprs[iover]
				push!(data[keynames[l]], fpr_val)
			end
		end
	end
end

figure(figsize=(8,8))
l=-1
for (i,k) in enumerate(keynames)
	global l+=2
	subplot(5,2,l)
	hist(data[k],99)
	title(k)
	i == 5 ? l = 0 : nothing 
end
tight_layout()
savefig(joinpath(savepath, "fpr_hist_alpha-$α.png"))

# also test if these histograms are seen in the experimental data
α = 0.05
master_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_post"
datasets = readdir(master_path)
measures = ["auc_at", "tpr_at"]
crit_labels = ["tukey_q", "welch_mean", "welch_median", "tukey_mean", "tukey_median"]

keynames = vcat(map(y->map(x->y*"_"*x, crit_labels), measures)...)
keynames_df = vcat(map(y->map(x->x*"_"*y, crit_labels), measures)...)
data = Dict()
map(x->data[x]=[], keynames)

for dataset in datasets
	inpath = joinpath(master_path, dataset)
	alldf,_,_ = ADME.collect_dfs(inpath);
	subsets = unique(alldf[!,:dataset])
	for subset in subsets
		subsetdf = filter(r->r[:dataset]==subset, alldf)
		for (k, kdf) in zip(keynames, keynames_df)
			colname = Symbol("$(kdf)_fpr")
			push!(data[k], subsetdf[1,colname])
		end
	end
end

figure(figsize=(8,8))
l=-1
for (i,k) in enumerate(keynames)
	global l+=2
	subplot(5,2,l)
	hist(data[k],99)
	title(k)
	i == 5 ? l = 0 : nothing 
end
tight_layout()
savefig(joinpath(savepath, "fpr_hist_alpha-$(α)_experiment.png"))








# do the same with friedman test
using DataFrames, CSV, Statistics, PyPlot, Distributions
using ADMetricEvaluation
ADME = ADMetricEvaluation
using PaperUtils
savepath = "/home/vit/vyzkum/measure_evaluation/discriminability/plots"

# get the df
master_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_pre"
measures = ["auc_at", "tpr_at"]
α = 0.05

# this is all gonna be a loop
datasets = readdir(master_path)
measure = measures[1]
dataset = "glass"
inpath = joinpath(master_path, dataset)
alldf,_,_ = ADME.collect_dfs(inpath);

# now filter it down
subsets = unique(alldf[!,:dataset])
subset = subsets[1]
subsetdf = filter(r->r[:dataset]==subset, alldf)
fprs = ADME.fpr_levels(subsetdf, measure)

fpr = fprs[1]
df = deepcopy(subsetdf)

function create_df_for_ranking(df, fpr, measure)
	# now create the df which we will rank
	fpr100 = round(Int, 100*fpr)
	colname = Symbol("$(measure)_$(fpr100)")

	# prepare the empty df
	param_vec = unique(df[!,:params])
	ks = unique(df[:,:iteration])
	_df = DataFrame(:k => string.(ks))
	for p in param_vec
		# filter the df and sort it by iterations
		subdf = sort(filter(r->r[:params]==p, df), :iteration)
		_df[!,Symbol(p)] = subdf[!,colname]
	end

	return _df
end

function get_friedman_stats(df, fprs, measure, adjusted=false)
	stats = []
	for fpr in fprs
		rdf = PaperUtils.rankdf(create_df_for_ranking(df, fpr, measure))
		n, k = size(rdf) .- (1,1)

		# now compute the friedman statistic
		R = Vector(rdf[end,2:end])
		fstat = adjusted ? 
			ADME.adjusted_friedman_test_statistic(R,n,k) : 
			ADME.friedman_test_statistic(R,n,k)	
		push!(stats, fstat)
	end
	stats
end

res = get_friedman_stats(subsetdf, fprs, measure)
ftest_critval = ADME.friedman_critval(α, k)

ares = get_friedman_stats(subsetdf, fprs, measure, true)
aftest_critval = ADME.adjusted_friedman_critval(α, n, k)

# ok, so this is not gonna work - the models are too different, and fpr = 0.01 will be selected always
# a different approach - first preselect the optimal models, then only compare the 4 best ones
fpr100 = round(Int, 100*fpr)
colname = Symbol("$(measure)_$(fpr100)")
measure_col = colname

df = deepcopy(subsetdf)

function opt_models(df, measure_col)
	_mcol = Symbol("$(measure_col)_mean")
	models = unique(df[!,:model])
	opt_params = []
	for model in models
		mdf = filter(r->r[:model]==model, df) # df of selected model
		adf = aggregate(mdf, [:dataset, :model, :params], mean) # aggregated df
		push!(opt_params, adf[ADME.nanargmax(adf[!,_mcol]), :params])
	end
	return opt_params, models
end

# now do the same but limit subdf to the 4 best params
opt_params, models = opt_models(subsetdf, measure_col)
optdf = filter(r->r[:params] in opt_params, subsetdf) 
kopt = length(models)
nopt = maximum(optdf[!,:iteration])

fres = get_friedman_stats(optdf, fprs, measure)
ftest_critval = ADME.friedman_critval(α, kopt)
afres = get_friedman_stats(optdf, fprs, measure, true)
ftest_critval = ADME.adjusted_friedman_critval(α, nopt, kopt)

# ok, lets try some histograms
measures = ["auc_at", "tpr_at"]
α = 0.01
datasets = readdir(master_path)

data = Dict()
map(m->data["$(m)_friedman"]=Float32[], measures)
map(m->data["$(m)_friedman_adjusted"]=Float32[], measures)
for dataset in datasets
	inpath = joinpath(master_path, dataset)
	alldf,_,_ = ADME.collect_dfs(inpath);

	# now filter it down
	subsets = unique(alldf[!,:dataset])
	for subset in subsets
		subsetdf = filter(r->r[:dataset]==subset, alldf)
		measure_col = :auc

		# select the optimal params based on what?!?!?!
		opt_params, models = opt_models(subsetdf, measure_col)
		optdf = filter(r->r[:params] in opt_params, subsetdf) 
		kopt = length(models)
		nopt = maximum(optdf[!,:iteration])
		
		# get the critical values
		ftest_critval = ADME.friedman_critval(α, kopt)
		aftest_critval = ADME.adjusted_friedman_critval(α, nopt, kopt)

		# now finally obtain the index of fpr that is the most discriminable
		for measure in measures
			fprs = ADME.fpr_levels(subsetdf, measure)

			fstats = get_friedman_stats(optdf, fprs, measure)
			afstats = get_friedman_stats(optdf, fprs, measure, true)

			iover = findfirst(ftest_critval .< fstats)
			aiover = findfirst(aftest_critval .< afstats)

			iover = iover == nothing ? 1 : iover
			aiover = aiover == nothing ? 1 : aiover

			push!(data["$(measure)_friedman"], fprs[iover])
			push!(data["$(measure)_friedman_adjusted"], fprs[aiover])
		end
	end
end

keynames = keys(data)

figure(figsize=(8,4))
l=-1
for (i,k) in enumerate(keynames)
	global l+=2
	subplot(2,2,l)
	hist(data[k],99)
	title(k)
	i == 2 ? l = 0 : nothing 
end
tight_layout()
savefig(joinpath(savepath, "friedman_fpr_hist_alpha-$α.png"))
