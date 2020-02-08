include("functions.jl")

bin(10, 0.5, 0.05)
bin(10, 0.8, 0.05)
bin(100, 0.5, 0.05)

poisson(1e4, 0, 0.05)
poisson(1e4, 1e-4, 0.05)
poisson(1e5, 1e-4, 0.05)

gauss(1e3, 0.5, 0.05)
gauss(1e3, 0.8, 0.05)

bin(1e6, 1e-5, 0.05)

# in this section, make preparations for computation of mean and std of AUC based on 
# http://papers.nips.cc/paper/2645-confidence-intervals-for-the-area-under-the-roc-curve.pdf

sigma_hanley(0.7, 232, 136)
sigma_hanley(0.63, 469, 231)
sigma_hanley(0.87, 164, 139)
sigma_hanley(0.84, 247, 2190)

sigma_max(0.7, 232, 136)
sigma_max(0.63, 469, 231)
sigma_max(0.87, 164, 139)
sigma_max(0.84, 247, 2190)

expected_auc(88, 232, 136)
expected_auc(182, 469, 231)
expected_auc(39, 164, 139)
expected_auc(74, 247, 2190)

sigma_cortes(88, 232, 136)
sigma_cortes(182, 469, 231)
sigma_cortes(39, 164, 139)
sigma_cortes(74, 247, 2190)

error_interval_bin(10, 30, 0.9)
error_interval_bin(10, 100, 0.9)
error_interval_gauss(10, 100, 0.9)

cl = 0.95
ϵ = 1-cl^2
error_interval_gauss(100, 1000, ϵ)
error_interval_bin(100, 1000, ϵ)

sigma_cortes(88, 232, 136)


n = 232
m = 126
k = 88
cl = 0.95
ϵ = 1-cl

cl, cu = confidence_interval_cortes(k, n, m, ϵ)
expected_auc(k, n, m)

# employ the e' = f(e) trick to obtain tighter bounds
k = 100
N = 1000
cl = 0.95
ϵ = 1-cl
nvec = 400:500
cis = map(n->confidence_interval_cortes(k, n, N-n, ϵ),nvec)


cl, cu = confidence_interval_cortes(25, 100, 400, ϵ)
cl, cu = confidence_interval_cortes(26, 100, 400, ϵ)

cl, cu = confidence_interval_cortes(50, 300, 700, ϵ)

cl, cu = confidence_interval_cortes(26, 10, 100, ϵ)

n = 10
m = 100
k = 26
ϵ = 0.05

confidence_interval_cortes(88, 232, 136, 0.05)
confidence_interval_cortes(182, 469, 231, 0.05)
confidence_interval_cortes(39, 164, 139, 0.05)
confidence_interval_cortes(74, 247, 2190, 0.05)

cl, cu = confidence_interval_cortes(100, 1000, 600, ϵ)

# how to compare the CI?
# small number of positives, large number of negatives
# change k - number of errors
# how to translate auc confidence intervals to auc@5 confidence intervals?
# i could take an roc curve, fix it at fpr=0.05
# compute the confidence bounds for one TPR using some 1D distribution
# and compute the auc ci based on cortes
# or, we could compute it for some changing parameter

# i want to see the width of auc ci for changing contamination rate
ϵ = 0.05
m = 200
n = 10
kvec = 0:10
aucs = map(k->expected_auc(k, n, m), kvec)
cis = map(k->confidence_interval_cortes(k, n, m, ϵ), kvec)

ϵ = 0.05
m = 20
n = 400
kvec = 0:2:20
aucs = map(k->expected_auc(k, n, m), kvec)
sigmas = map(k->sigma_cortes(k, n, m), kvec)
# sigma se zmensuje podle poctu samplu

# k = pocet spatne serazenych skore
map(k->expected_auc(n+1, n, m), kvec)

# musim zjistit std pro ten binomialni/gaussovsky model - pro ten druhy je to easy

using PyPlot

a = 0.8
b = 1.8
x = 0.0:0.01:1.0
y = roc_curve_parametrized.(x, a, b)
plot(x,y)

# zde se pokusim udelat odhad pro a,b v tom bigaussovskem modelu pro odhad roc
using EvalCurves

N2 = 1000
N = 2*N2
m0 = -1
sd0 = 1.1
m1 = 1
sd1 = 0.8
X0 = m0 .+ sd0*randn(N2)
X1 = m1 .+ sd1*randn(N2)
figure()
hist(X0, alpha=0.5, 20)
hist(X1, alpha=0.5, 20)

y = vcat(zeros(N2), ones(N2));
scores = vcat(X0, X1);
roc = EvalCurves.roccurve(scores, y)

a = abs(m1-m0)/sd1
b = sd0/sd1
xvec = 0:0.01:1.0
yvec = roc_curve_parametrized.(xvec, a, b)
figure()
scatter(roc...,s=2)
plot(xvec, yvec, c="r")

# mle odhad a,b je v https://onlinelibrary.wiley.com/doi/pdf/10.1002/(SICI)1097-0258(19980515)17:9%3C1033::AID-SIM784%3E3.0.CO;2-Z?casa_token=CpGaL20diaYAAAAA:Tus3_8Q6UAH70xoU__RNuDtUIdwuJSkMvXDIS5oUgp3wo-MzmLPhiB6pCYXOOgGMF-V9mW6ZyQnCdsA

