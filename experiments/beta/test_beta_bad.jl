include("test_beta_alternatives_functions.jl")

# now lets put it all together
fprs = [0.01, 0.05]
measuref = beta_auc
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
row_measures = [:auc, :auc_weighted, :auc_at_1, :tpr_at_1, :bauc_at_1, :lauc_at_1, :measure_at_1,
		:auc_at_5, :tpr_at_5, :bauc_at_5, :lauc_at_5, :measure_at_5]
column_measures = [:auc_at_1, :auc_at_5, :prec_at_1, :prec_at_5,
		:tpr_at_1, :tpr_at_5]

dataset = "cardiotocography"
subdataset = "cardiotocography-2-7"

measure_loss_df, alldf_val, alldf_tst, data = 
	measure_test_results(dataset, subdataset, measuref, savepath, fprs, orig_path)
X_tr, y_tr, X_val, y_val, X_tst, y_tst = data

measuref = beta_auc_alt
measure_loss_df_alt, alldf_val_alt, alldf_tst_alt, data = 
	measure_test_results(dataset, subdataset, measuref, savepath, fprs, orig_path)

# now inspect the worst result
alldf_val[!, [:model, :params, :auc, :auc_at_5, :bauc_at_5]]
alldf_val_alt[!, [:model, :params, :auc,:tpr_at_5, :auc_at_5, :bauc_at_5, :measure_at_5]]
# │ 39  │ OCSVM  │ gamma=100.0        │ 0.531466 │ 0.0266904 │ 0.758705  │
# wine-2-3                                            │ OCSVM  │ gamma=5.0

measuref = beta_auc_alt
measure_loss_df_alt, alldf_val_alt, alldf_tst_alt, data = 
	measure_test_results("wine", "wine-2-3", measuref, savepath, fprs, orig_path)

# now inspect the worst result
alldf_val[!, [:model, :params, :auc, :auc_at_5, :bauc_at_5]]
alldf_val_alt[!, [:model, :params, :auc,:tpr_at_5, :auc_at_5, :bauc_at_5, :measure_at_5]]
# │ 39  │ OCSVM  │ gamma=100.0        │ 0.531466 │ 0.0266904 │ 0.758705  │
# wine-2-3                                            │ OCSVM  │ gamma=5.0

model = OCSVM_model(100.0)
ScikitLearn.fit!(model, Array(transpose(X_tr)))
score_fun(X) = -ScikitLearn.decision_function(model, Array(transpose(X)))
scores = score_fun(X_val)
fprvec, tprvec = roccurve(scores, y_val)

figure()
plot(fprvec, tprvec)

fpr = 0.05
beta_auc(scores, y_val, fpr, 1000)

beta_auc_alt(scores, y_val, fpr, 1000)


function beta_auc_alt(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    # first sample fprs and get parameters of the beta distribution
    fprs = EvalCurves.fpr_distribution(scores, y_true, fpr, nsamples, d, warns=warns)
    # filter out NaNs
    fprs = fprs[.!isnan.(fprs)]
    (length(fprs) == 0) ? (return NaN) : nothing

    # test whether the fprs actually make any sense
    m, s2 = mean_and_var(fprs)
    if !(m-3*sqrt(s2) < fpr < m+3*sqrt(s2))
    	warns ? (@warn "the requested fpr is out of the sampled fpr distribution, returning NaN") : nothing
    	return NaN
    end

    # now continue
    α, β = EvalCurves.estimate_beta_params(fprs)

    # compute roc
    roc = EvalCurves.roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = EvalCurves.linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = exp.(EvalCurves.beta_logpdf.(xw, α, β))
    wauroc = EvalCurves.auc(roci..., w)
end


fprs = EvalCurves.fpr_distribution(scores, y_val, 0.05, 1000, 0.5)
α, β = EvalCurves.estimate_beta_params(fprs)
roc = EvalCurves.roccurve(scores, y_val)
interp_len = max(1001, length(roc[1]))
roci = EvalCurves.linear_interpolation(roc..., n=interp_len)

# weights are given by the beta pdf and are centered on the trapezoids
dx = (roci[1][2] - roci[1][1])/2
xw = roci[1][1:end-1] .+ dx
w = exp.(EvalCurves.beta_logpdf.(xw, α, β))

figure()
subplot(121)
title("ROC")
plot(roc...)
subplot(122)
hist(fprs, density = true)
plot(xw, w)
title("fpr distribution and beta fit at FPR=0.05")
savefig(joinpath(savepath, "bad_fpr_fit.png"))

# we have to come up with a defense against this
