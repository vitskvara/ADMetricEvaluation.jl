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
alldf = ADME.collect_dfs(inpath)

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
	alldf = ADME.collect_dfs(inpath)

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
	hist(data[k],49)
	title(k)
	i == 5 ? l = 0 : nothing 
end
tight_layout()
savefig(joinpath(savepath, "fpr_hist_alpha-$α.png"))












figure(figsize=(4,8))
for (i,st) in enumerate(zip(stats[1], cvals[1]))
	subplot(510+i)
	title(crit_labels[i])
	plot(fprs, st[1])
	plot(fprs, st[2])
end
tight_layout()

α = 0.05
m1 = 20.8
m2 = 23.0
s1 = sqrt(7.9)
s2 = sqrt(3.8)
n1 = 15
n2 = 15
t = ADME.welch_test_statistic(m1, m2, s1, s2, n1, n2)
df = ADME.welch_df(s1, s2, n1, n2)
ADME.critt(α/2, df)


i = 4
j = 10
k = 1
nexp = 50
# which of these we use?
msw = sqrt((means_vars[1][2][i,k] + means_vars[1][2][j,k])/2)
msw = (nexp-1)*sum(means_vars[1][2][:,k])
msw = mean(means_vars[1][2][:,k])
ADME.tukey_test_statistic(means_vars[1][1][i,k], means_vars[1][1][j,k], msw, nexp)
ngrps = size(means_vars[1][1])[1]
ADME.tukey_critval(α, ngrps, (ngrps-1)*nexp)
