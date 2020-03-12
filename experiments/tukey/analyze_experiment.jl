include("../t-test/plot_functions.jl")
include("functions.jl")

using DataFrames, CSV, ADMetricEvaluation
ADME = ADMetricEvaluation

master_inpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_pre"
dataset = "wine"
dataset = "statlog-satimage"
subset = "2-1"
subset = "1-3"
dataset = "ecoli"
subset = "cp-imL"

outpath = create_outpath(master_inpath, dataset)
inpath = joinpath(master_inpath, dataset)
# get individual datasets
dfs = map(x->CSV.read(joinpath(inpath, x)), readdir(inpath))
alldf = vcat(map(df->ADME.drop_cols!(ADME.merge_param_cols!(df)), dfs)...);
subsetdf = filter(r->r[:dataset]=="$(dataset)-$(subset)", alldf)

#
df = copy(subsetdf)
measure = "auc_at"
max_fpr = 1.0

# determine some constants
fprs = fpr_levels(df, measure)
fprs = fprs[fprs.<=max_fpr]
nfprs = length(fprs) # number of fpr levels of interes
nexp = maximum(df[!,:iteration]) # number of experiments

# get the dfs with means and variances
colnames = names(df)
meas_cols = filter(x->occursin(measure, string(x)), colnames)[1:nfprs]
subdf = df[!,vcat([:model, :params, :iteration], meas_cols)]

# now get the actual values
mean_df = aggregate(subdf, [:model, :params], nanmean);
remove_appendix!(mean_df, meas_cols, "nanmean");
var_df = aggregate(subdf, [:model, :params], nanvar);
remove_appendix!(var_df, meas_cols, "nanvar");
mean_vals_df = mean_df[!,meas_cols] # these two only contain the value columns
var_vals_df = var_df[!,meas_cols]

# get the lines
tq, wt_mean, wt_med, tt_mean, tt_med = 
	stat_lines(mean_vals_df, var_vals_df, meas_cols, nexp)

# plot it
figure()
subplot(311)
plot(fprs, tq)
subplot(312)
plot(fprs,tt_mean)
subplot(313)
plot(fprs,wt_mean)

df =copy(dfs[1])
opt_fpr = opt_fprs[1]
measure = measures[1]

