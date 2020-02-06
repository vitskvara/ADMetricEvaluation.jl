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

# pack this nicely in a package and test it against the old measures
