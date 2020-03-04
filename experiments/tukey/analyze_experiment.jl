include("../t-test/plot_functions.jl")
include("functions.jl")

using DataFrames, CSV, ADMetricEvaluation
ADME = ADMetricEvaluation

inpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_pre"
dataset = "wine"
dataset = "statlog-satimage"
dfs = ADME.loaddata(dataset, inpath)
alldf = vcat(map(df->ADME.drop_cols!(ADME.merge_param_cols!(df)), dfs)...);

# do it for only one subdataset first
subset = "2-1"
subset = "1-3"
subdf = filter(r->occursin(subset,r[:dataset]),alldf)
colnames = names(subdf)
measure = "auc_at"
meas_cols = filter(x->occursin(measure, string(x)), colnames)
subcols = vcat([:model, :params, :iteration], meas_cols)
subdf = subdf[!,subcols]
mean_df = aggregate(subdf, [:model, :params], nanmean);
remove_appendix!(mean_df, meas_cols, "nanmean");
var_df = aggregate(subdf, [:model, :params], nanvar);
remove_appendix!(var_df, meas_cols, "nanvar");

mean_df=mean_df[setdiff(1:end, 4),:]
var_df=var_df[setdiff(1:end, 4),:]
# now we need to compute pairwise welch, pairwise tukey and tukey q

fprs = fpr_levels(alldf, "auc_at")

figure()
for row in eachrow(mean_df)
	plot(fprs, Array(row[4:end]))
end

n = 50
tq = map(c->tukey_q(mean_df[c], var_df[c], repeat([n], length(mean_df[c]))), meas_cols)

# the pairwise stuff is tricky
nr = size(mean_df,1)
nc = size(mean_df,2) - 3
wtm = zeros(Float32, binomial(nr,2), nc)
ttm = zeros(Float32, binomial(nr,2), nc)
l = 0
for i in 1:nr
	for j in i+1:nr
		global l += 1
		for k in 1:nc
			# welch statistic
			wtm[l,k] = abs(welch_test_statistic(mean_df[i,k+3], mean_df[j,k+3], var_df[i,k+3], 
				var_df[j,k+3], n, n))
			# tukey statistic
			msw = (n-1)*sum(var_df[:,k+3])
			ttm[l,k] = tukey_stat(mean_df[i,k+3], mean_df[j,k+3], msw, n)
		end
	end
end
wt = vec(mean(wtm, dims=1))
tt = vec(mean(ttm, dims=1))

figure()
subplot(311)
plot(fprs, tq)
subplot(312)
plot(fprs,tt)
subplot(313)
plot(fprs,wt)

figure()
plot(fprs[1:30],wt[1:30])


