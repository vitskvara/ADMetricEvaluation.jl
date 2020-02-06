
# now - take a dataset with auc~0.8, scatter the roc, compute a and b
# show the ci for pAUC for varying FPR
# this is based on https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.5777?casa_token=k-CRRr4v3KAAAAAA:gSEX7Ul3DhKFWRGCgcor8qkPnritKWZYXZZEhavNDPlTQZJ1egFEOlfeXJITdg9mLV_qnMNJCwj91z0
# also, show the CI for FPR and TPR based on the models in 
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.7019&rep=rep1&type=pdf
using EvalCurves
using UCI
using Distances
using PyPlot

include("functions.jl")

# get data
dataset = "wine"
data, normal_labels, anomaly_labels = UCI.get_data(dataset)
X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8; seed = 123)

# now compute it
k = 5
scores = knn_scores(X_tr, X_tst, k)
roc = EvalCurves.roccurve(scores, y_tst)
auc = EvalCurves.auc(roc...)

plot(roc...)
title(auc)
ylim([0,1.02])
xlim([-.02,1])

scatter(roc..., s= 3)
title(auc)
ylim([0,1.02])
xlim([-.02,1])

xvec=0:0.01:1
a = 1.2
b = 0.8
yvec = roc_curve_parametrized.(xvec,a,b)
plot(xvec, yvec)

# now compute the pauc
fpr = 0.1
pauc = EvalCurves.auc_at_p(roc..., fpr)
npauc = pauc/fpr
tpr = EvalCurves.tpr_at_fpr(roc..., fpr)

m = sum(y_tst)
n = length(y_tst) - m

vauc = var_pauc(fpr, a, b, n, m)
stauc = sqrt(vauc)

#tpr_ci = bin(n, tpr, 0.95)
tpr_ci = bin(n, tpr, 0.05)

# this is std of binomial distribution
sqrt(n*tpr*(1-tpr))

fprs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.9]
staucs = map(x->sqrt(var_pauc(x, a, b, n, m)), fprs)
paucs = map(x->EvalCurves.auc_at_p(roc..., x), fprs)
tprs = map(x->EvalCurves.tpr_at_fpr(roc..., x), fprs)
tpr_cis = map(x->bin(n, x, 0.05), tprs)

using DataFrames
df = DataFrame(
	:fpr => fprs,
	:pauc => paucs,
	:std_pauc => staucs,
	:rel_std_pauc => staucs./paucs,
	:tpr => tprs,
	:tpr_ci => tpr_cis,
	:rel_ci_length => map(x->abs(x[2][2]-x[2][1])/2/x[1] ,zip(tprs, tpr_cis))
	)

# try to compute the a,b from means and sds of the data
scores = knn_scores(X_tr, X_tst, k)
roc = EvalCurves.roccurve(scores, y_tst)
m0 = mean(scores[y_tst.==0])
sd0 = std(scores[y_tst.==0])
m1 = mean(scores[y_tst.==1])
sd1 = std(scores[y_tst.==1])

a = abs(m1-m0)/sd1
b = sd0/sd1

xvec = 0:0.01:1.0
yvec = roc_curve_parametrized.(xvec, a, b)
figure()
scatter(roc...,s=2)
plot(xvec, yvec, c="r")
