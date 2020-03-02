# here we try to compute bootstrapping confidence intervals
using UCI
using PyPlot

include("functions.jl")
savepath = "/home/vit/vyzkum/measure_evaluation/ci_data_plots"
# ci 
α = 0.05
k = 7
nbins = 30

# get data
# make changes in UCI so that the original data labels are also available
# also do train/test/validation splits
dataset = "ionosphere"
difficulties = nothing
#125

dataset = "waveform-1"
difficulties = nothing
# 1696

dataset = "yeast" 
difficulties = [:easy, :medium]

dataset = "wine" 
difficulties = nothing

dataset = "pima-indians"
difficulties = [:easy, :medium]

dataset = "abalone"
difficulties = [:easy, :medium]
#51

dataset = "statlog-satimage"
difficulties = [:easy, :medium]

dataset =  "blood-transfusion"
difficulties = [:easy, :medium]

dataset =  "libras"
difficulties = [:easy, :medium]
#143

dataset =  "sonar"
difficulties = [:easy, :medium]
# 91

dataset =  "breast-cancer-wisconsin"
difficulties = [:easy, :medium]
# 91

#data, normal_labels, anomaly_labels = UCI.get_data(dataset)
#X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8; seed = 123)

data, normal_labels, anomaly_labels = UCI.get_loda_data(dataset);
X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8; seed = 123, difficulty=difficulties);

# now compute it
scores = knn_scores(X_tr, X_tst, k);
roc = EvalCurves.roccurve(scores, y_tst);
auc = EvalCurves.auc(roc...)
m0 = mean(scores[y_tst.==0])
std0 = std(scores[y_tst.==0])
m1 = mean(scores[y_tst.==1])
std1 = std(scores[y_tst.==1])
a,b = roc_parameters(m0, std0, m1, std1)

# plot empirical roc
if false
	figure(figsize=(10,10))

	subplot(221)
	plot(roc...)
	title("AUC=$auc")
	ylim([0,1.02])
	xlim([-.02,1])

	# scatter empirical points + parametrized roc
	subplot(222)
	scatter(roc..., s=1)
	ylim([0,1.02])
	xlim([-.02,1])

	# replace this with a proper fit!
	xvec=0:0.01:1
	yvec = roc_curve_parametrized.(xvec,a,b)
	plot(xvec, yvec,"r")
	title("a=$a, b=$b")

	subplot(223)
	hist(scores[y_tst.==1],color="r", alpha=0.4,nbins,density=true,label="pos");
	hist(scores[y_tst.==0],color="b", alpha=0.4,nbins,density=true,label="neg");
	xvec_hist = range(minimum(scores), maximum(scores), length=1000);
	density0 = normal_density.(xvec_hist, m0, std0);
	density1 = normal_density.(xvec_hist, m1, std1);
	plot(xvec_hist, density0, "b")
	plot(xvec_hist, density1, "r")
	legend();
end

#### BOOTSTRAPPING ####
# this was just a small visualisation
# now do bootstrapping ci for auc, partial auc, fpr@tpr
# first generate N new testing datasets
N = 1000
sample_inds(n) = sample(1 :n, n);

fn = joinpath(savepath, dataset*".jld2")
# if data is alread present, laod it
if isfile(fn)
	bs_data = load(fn);
	bs_scores = bs_data["scores"];
	bs_rocs = bs_data["rocs"];
else	
# else recreate it
	bs_inds = hcat([sample_inds(length(y_tst)) for _ in  1:N]...);
	bs_scores = mapslices(is -> vec(knn_scores(X_tr, X_tst[:,is], k)), bs_inds, dims=2);
	bs_rocs = map(n->EvalCurves.roccurve(bs_scores[:,n], y_tst[bs_inds[:,n]]), 1:N);
	save(fn, Dict("scores"=>bs_scores, "rocs"=>bs_rocs));
end

# now compute the statistics
bs_auc_mean, bs_auc_ci = mean_and_ci(bs_rocs, α, EvalCurves.auc)

# do the same for fpr levels
fprs = Array(0.01:0.005:0.99);
bs_tpr_means, bs_tpr_cis = mean_and_ci(bs_rocs, α, fprs, EvalCurves.tpr_at_fpr)
bs_pauc_means, bs_pauc_cis = mean_and_ci(bs_rocs, α, fprs, EvalCurves.auc_at_p)

# make the plot 
xvec_hist = range(minimum(scores), maximum(scores), length=1000);
density0 = normal_density.(xvec_hist, m0, std0);
density1 = normal_density.(xvec_hist, m1, std1);
plot_cis_etc(roc, auc, a, b, fprs, y_tst, scores, xvec_hist, density0, density1,
	bs_tpr_means, bs_tpr_cis, bs_pauc_means, bs_pauc_cis)
ff = joinpath(savepath, dataset*"_bootstrapping.png")
savefig(ff)

#### SAMPLING CI #########
scores0 = scores[y_tst.==0];
scores1 = scores[y_tst.==1];

# optimal number of components selection via BIC
Nmax = 3
crit = "max"
if crit == "BIC"
	mix0 = select_optimal_gmm(scores0, Nmax, BIC)
	mix1 = select_optimal_gmm(scores1, Nmax, BIC)
else
	mix0 = select_optimal_gmm(scores0, Nmax, max_n)
	mix1 = select_optimal_gmm(scores1, Nmax, max_n)
end

# now plot it
#figure()
n0 = mix0.n
n1 = mix1.n
#hist(scores[y_tst.==1],color="r", alpha=0.4,nbins,density=true);
#hist(scores[y_tst.==0],color="b", alpha=0.4,nbins,density=true);
xvec_hist = collect(range(minimum(scores), maximum(scores), length=1000))
p0 = pdf(mix0, xvec_hist)
p1 = pdf(mix1, xvec_hist)
#plot(xvec_hist, p0,"b",label="n0=$n0")
#plot(xvec_hist, p1,"r",label="n1=$n1")
#legend()

# test sampling
Ns = 10000
if false
	figure()
	samples0 = sample(mix0, Ns)
	samples1 = sample(mix1, Ns)
	hist(samples0,color="b", alpha=0.4,nbins,density=true)
	hist(samples1,color="r", alpha=0.4,nbins,density=true)
	#hist(scores[y_tst.==1],color="brown", alpha=0.4,nbins,density=true)
	#hist(scores[y_tst.==0],color="g", alpha=0.4,nbins,density=true)
	plot(xvec_hist, p0,"b",label="n0=$n0")
	plot(xvec_hist, p1,"r",label="n1=$n1")

	# how do we get the cis from this? we can just resample it forever
	figure()
	tr = roc_from_samples(mix0, mix1, 200)
	plot(roc...)
	plot(tr...)
end

# do a pseudo-boostrapping again
N1 = sum(y_tst)
N0 = length(y_tst)-N1
gmm_rocs = [roc_from_samples(mix0, mix1, (N0, N1)) for _ in 1:N]

#
gmm_auc_mean, gmm_auc_ci = mean_and_ci(gmm_rocs, α, EvalCurves.auc)
mean_and_ci(bs_rocs, α, fprs, EvalCurves.auc_at_p)

# do the same for fpr levels
fprs = Array(0.01:0.005:0.99);
gmm_tpr_means, gmm_tpr_cis = mean_and_ci(gmm_rocs, α, fprs, EvalCurves.tpr_at_fpr)
gmm_pauc_means, gmm_pauc_cis = mean_and_ci(gmm_rocs, α, fprs, EvalCurves.auc_at_p)

# also do GMM 1k
N1 = N0 = 1000
gmm_rocs_1k = [roc_from_samples(mix0, mix1, (N0, N1)) for _ in 1:N]
gmm_auc_mean_1k, gmm_auc_ci_1k = mean_and_ci(gmm_rocs_1k, α, EvalCurves.auc)
fprs = Array(0.01:0.005:0.99);
gmm_tpr_means_1k, gmm_tpr_cis_1k = mean_and_ci(gmm_rocs_1k, α, fprs, EvalCurves.tpr_at_fpr)
gmm_pauc_means_1k, gmm_pauc_cis_1k = mean_and_ci(gmm_rocs_1k, α, fprs, EvalCurves.auc_at_p)

# do the plot
xvec_hist = collect(range(minimum(scores), maximum(scores), length=1000));
density0 = pdf(mix0, xvec_hist);
density1 = pdf(mix1, xvec_hist)
plot_cis_etc(roc, auc, a, b, fprs, y_tst, scores, xvec_hist, density0, density1,
	gmm_tpr_means, gmm_tpr_cis, gmm_pauc_means, gmm_pauc_cis)
ff = joinpath(savepath, dataset*"_gmm.png")
savefig(ff)

# now compare the two
xvec_hist = collect(range(minimum(scores), maximum(scores), length=1000));
density0 = pdf(mix0, xvec_hist);
density1 = pdf(mix1, xvec_hist)
compare_cis(roc, auc, a, b, fprs, y_tst, scores, density0, density1,
	bs_tpr_means, bs_tpr_cis, gmm_tpr_means, gmm_tpr_cis,
	bs_pauc_means, bs_pauc_cis, gmm_pauc_means, gmm_pauc_cis)
ff = joinpath(savepath, dataset*"_comparison.png")
savefig(ff)

# we can get CI smaller than Kerekes by adding more GMM samples
# now add the analytical confidence intervals
Np = sum(y_tst)
α = 0.05
analytical_tpr_cis = map(x->kerekes_ci(Np, x, α), bs_tpr_means)

figure(figsize=(8,8))
subplot(221)
title("TPR@FPR")
fill_between(fprs, [x[1] for x in bs_tpr_cis], [x[2] for x in bs_tpr_cis], alpha=0.3, label="BS")
fill_between(fprs, [x[1] for x in gmm_tpr_cis], [x[2] for x in gmm_tpr_cis], alpha=0.3, label="GMM")
fill_between(fprs, [x[1] for x in gmm_tpr_cis_1k], [x[2] for x in gmm_tpr_cis_1k], alpha=0.3, label="GMM-1k")
fill_between(fprs, [x[1] for x in analytical_tpr_cis], [x[2] for x in analytical_tpr_cis], alpha=0.3, label="Kerekes")
ylabel("TPR")
legend(loc="lower right")

subplot(223)
plot(fprs, [x[2]-x[1] for x in bs_tpr_cis]./bs_tpr_means, label="BS")
plot(fprs, [x[2]-x[1] for x in gmm_tpr_cis]./gmm_tpr_means, label="GMM")
plot(fprs, [x[2]-x[1] for x in gmm_tpr_cis_1k]./gmm_tpr_means_1k, label="GMM-1k")
plot(fprs, [x[2]-x[1] for x in analytical_tpr_cis]./bs_tpr_means, label="Kerekes")
xlabel("FPR")
ylabel("relative CI width")
legend()

# also add analytical cis for pauc
Np = sum(y_tst)
Nn = length(y_tst) - Np
α = 0.05
analytical_pauc_cis = map(x->ma_ci(x[1], x[2], a, b, Nn, Np, α), zip(bs_pauc_means, fprs))

subplot(222)
title("AUC@FPR, Np=$Np, Nn=$Nn")
fill_between(fprs, [x[1] for x in bs_pauc_cis], [x[2] for x in bs_pauc_cis], alpha=0.3, label="BS")
fill_between(fprs, [x[1] for x in gmm_pauc_cis], [x[2] for x in gmm_pauc_cis], alpha=0.3, label="GMM")
fill_between(fprs, [x[1] for x in gmm_pauc_cis_1k], [x[2] for x in gmm_pauc_cis_1k], alpha=0.3, label="GMM-1k")
fill_between(fprs, [x[1] for x in analytical_pauc_cis], [x[2] for x in analytical_pauc_cis], alpha=0.3, label="Ma")
ylabel("AUC@FPR")
legend(loc="lower right")

subplot(224)
plot(fprs, [x[2]-x[1] for x in bs_pauc_cis]./bs_pauc_means, label="BS")
plot(fprs, [x[2]-x[1] for x in gmm_pauc_cis]./gmm_pauc_means, label="GMM")
plot(fprs, [x[2]-x[1] for x in gmm_pauc_cis_1k]./gmm_pauc_means_1k, label="GMM-1k")
plot(fprs, [x[2]-x[1] for x in analytical_pauc_cis]./bs_pauc_means, label="Ma")
xlabel("FPR")
ylabel("relative CI width")
legend()
tight_layout()
ff = joinpath(savepath, dataset*"_analytical_vs_empirical.png")
savefig(ff)

# ok so this is a bit weird since there is quite a discontinuity at fpr=0.1 
# since the poisson and gauss cis are quite different
if false
	Np = 200
	ttpr = collect(0:0.01:1.0)
	tcintervals = map(x->bin_ci(Np,x,α),ttpr)
	plot(ttpr, [x[2] for x in tcintervals]-ttpr, label="bin")

	tncintervals = map(x->kerekes_ci(Np,x,α),ttpr)
	plot(ttpr, [x[2] for x in tncintervals]-ttpr, label="combined")
	legend()
end