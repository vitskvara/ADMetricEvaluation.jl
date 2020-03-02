using Distances
using EvalCurves
using JLD2
using FileIO
using GaussianMixtures
using Statistics
using Distributions
using StatsBase
import StatsBase.sample

# these confidence intervals are base on Kerekes - Receiver Operating Characteristic Curve Confidence Intervals and Regions
critval(α, dist) = invlogccdf(dist, log(α))
critf(α, df1, df2) = quantile(FDist(df1, df2), 1-α)
critchi2(α, df) = quantile(Chisq(df), 1-α)
critt(α, df) = quantile(TDist(df), 1-α)

function bin(N, θ, α)
	θL = N*θ/(N*θ+(N-N*θ+1)*critf(α/2, 2*(N-N*θ+1), 2*N*θ))
	f = critf(α/2, 2*(N*θ+1), 2*(N-N*θ))
	θU = (N*θ+1)*f/(N-N*θ+(N*θ+1)*f)
	return θL, θU
end

function bin_ci(N, θ, α)
	if θ == 0
		return (0.0, 0.0)
	elseif θ == 1
		return (1.0, 1.0)
	else
		return bin(N, θ, α) 
	end
end

function poisson(N, θ, α) 
	if θ > 0
		return critchi2(1-α/2, 2*N*θ)/(2*N), critchi2(α/2, 2*(N*θ+1))/(2*N)
	else
		return 0.0, critchi2(α, 2*(N*θ+1))/(2*N)
	end
end
function poisson_ci(N, θ, α)
	if θ == 0
		return (0.0, 0.0)
	elseif θ == 1
		return (1.0, 1.0)
	else
		return poisson(N, θ, α) 
	end
end

function gauss(N, θ, α)
	s = sqrt(θ*(1-θ))*critt(α/2, N-1)/sqrt(N)
	θ-s, θ+s
end

function gauss_ci(N, θ, α)
	if θ == 0
		return (0.0, 0.0)
	elseif θ == 1
		return (1.0, 1.0)
	else
		return gauss(N, θ, α) 
	end
end

function kerekes_ci(N, θ, α)
	if N <= 100
		return bin_ci(N, θ, α)
	elseif θ < 0.1
		return poisson_ci(N, θ, α)
	else
		return gauss_ci(N, θ, α)
	end
end

# this should work for exponential distributions
function sigma_hanley(A::Real, n::Int, m::Int)
	Pxxy = A/(2-A)
	Pxyy = 2*A^2/(1+A)
	return sqrt(1/(m*n)*(A*(1-A)+(m-1)*(Pxxy-A^2)+(n-1)*(Pxyy-A^2)))
end

sigma_max(A::Real, n::Int, m::Int) = sqrt(A*(1-A)/min(m,n))

function expected_auc(k::Int, n::Int, m::Int)
	l = k/(m+n)
	ea = 1-l 
	m == n && return ea
	a = sum(map(i->binomial(BigInt(m+n),BigInt(i)),0:k-1))
	b = sum(map(i->binomial(BigInt(m+n+1),BigInt(i)),0:k))
	ea - (n-m)^2*(m+n+1)/(4*m*n)*(l-a/b)
end

function sigma_cortes(k::Int, n::Int, m::Int)
	# compute Z_(1:4)
	function Zi(i::Int, k::Int, n::Int, m::Int)
		a = sum(map(j->binomial(BigInt(m+n+1-i),BigInt(j)),0:k-i))
		b = sum(map(j->binomial(BigInt(m+n+1),BigInt(j)),0:k))
		a/b
	end 
	Z = map(i->Zi(i,k,n,m),1:4)

	# now the rest
	T = 3*((m-n)^2 + m + n) + 2
	Q0 = (m+n+1)*T*k^2 + ((-3*n^2+3*m*n+3*m+1)*T - 12*(3*m*n+m+n) - 8)*k + 
		(-3*m^2+7*m+10*n+3*n*m+10)*T - 4*(3*m*n+m+n+1)
	Q1 = T*k^3 + 3*(m-1)*T*k^2 + ((-3*n^2+3*m*n-3*m+8)*T - 6*(6*m*n+m+n))*k + 
		(-3*m^2+7*(m+n)+3*m*n)*T - 2*(6*m*n+m+n)

	# and the result is:
	σ2 = (m+n+1)*(m+n)*(m+n-1)*T*((m+n-2)*Z[4] - (2*m-n+3*k-10)*Z[3])/72 + 
		(m+n+1)*(m+n)*T*(m^2-n*m+3*k*m-5*m+2*k^2-n*k+12-9*k)*Z[2]/48 -
		(m+n+1)^2*(m-n)^4*Z[1]^2/16 - 
		(m+n+1)*Q1*Z[1]/72 + 
		k*Q0/144
	σ2 = σ2/(m^2*n^2)
	sqrt(σ2)
end

function error_interval_bin(k::Int, N::Int, ϵ::Real)
	si = 2*sqrt((1-sqrt(1-ϵ))*N)
	return k/N - 1/si, k/N + 1/si
end

# this one is more suitable for large N
function error_interval_gauss(k::Int, N::Int, ϵ::Real)
	p = (1-sqrt(1-ϵ))/2
	s = quantile(Normal(), 1-p)/(2*sqrt(N))
	return k/N - s, k/N + s
end

function confidence_interval_cortes(k::Int, n::Int, m::Int, ϵ::Real)
	N = m+n
	el, eu = (N < 100) ? error_interval_bin(k, N, ϵ) : error_interval_gauss(k, N, ϵ)
	kl, ku = max(0, floor(Int, el*N)), ceil(Int, eu*N)
	As = map(k->expected_auc(k, n, m), kl:ku) 
	σs = map(k->sigma_cortes(k, n, m), kl:ku)
	α1 = max(minimum(As - σs/sqrt(ϵ)), 0.0)
	α2 = maximum(As + σs/sqrt(ϵ))
	return α1, α2
end

# this is based on https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.5777?casa_token=k-CRRr4v3KAAAAAA:gSEX7Ul3DhKFWRGCgcor8qkPnritKWZYXZZEhavNDPlTQZJ1egFEOlfeXJITdg9mLV_qnMNJCwj91z0
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.2103?casa_token=aI9RfkC-gMQAAAAA:bHLwJoanfXnV_azieJDc7z6Tts-T_xEJ2K2QkxzIgIgIVG-CUc8luduYCVia6oBLMus68bQtIW-XeDg
roc_curve_parametrized(x::Real, a::Real, b::Real) = cdf(Normal(), a + b*quantile(Normal(), x))
# these can be instead derived from the empirical mle fit 
var_a(a::Real, b::Real, n::Int, m::Int) = (n*(a^2+2)+2*m*b^2)/(2*n*m)
var_b(b::Real, n::Int, m::Int) = (n+m)*b^2/(2*n*m)
cov_ab(a::Real, b::Real, n::Int, m::Int) = a*b/(2*m)
dAda(a::Real, b::Real, h::Real) = exp(-a^2/(2*(1+b^2)))*cdf(Normal(), h)/sqrt(2*pi*(1+b^2))
dAdb(a::Real, b::Real, h::Real) = 
	-exp(-a^2/(2*(1+b^2)))*(exp(-h^2/2)/(2*pi*(1+b^2)) + a*b*cdf(Normal(), h)/sqrt(2*pi*(1+b^2)^3))
hf(e::Real, a::Real, b::Real) = sqrt(1+b^2)*(quantile(Normal(), e) + a*b/(1+b^2))
var_auc(va::Real, vb::Real, da::Real, db::Real, cab::Real) = da^2*va + db^2*vb + 2*da*db*cab
# n = no negatives, m = no positives
function var_pauc(e::Real, a::Real, b::Real, n::Int, m::Int)
	va = var_a(a, b, n, m)
	vb = var_b(b, n, m)
	cab = cov_ab(a, b, n, m)
	h = hf(e, a, b) 
	da = dAda(a, b, h)
	db = dAdb(a, b, h)
	var_auc(va, vb, da, db, cab)
end
roc_parameters(m0::Real, std0::Real, m1::Real, std1::Real) = abs(m1-m0)/std1, std0/std1
function ma_ci(pauc::Real, e::Real, a::Real, b::Real, n::Int, m::Int, α::Real)
	std = sqrt(var_pauc(e::Real, a::Real, b::Real, n::Int, m::Int))
	q = quantile(Normal(), 1-α/2)
	ul, up = max(0.0, pauc-q*std), min(1.0, pauc+q*std)
end

# define a knn fit
function knn_score(X_tr::Matrix, x::Vector, k::Int)  
	d = sort(colwise(Euclidean(), x, X_tr))
	(length(d) >= k) ? nothing : error("selected k=$k too big")
	mean(d[1:k])
end
knn_scores(X_tr::Matrix, X_tst::Matrix, k::Int) = vec(mapslices(x->knn_score(X_tr,x,k), X_tst, dims=1))

normal_density(x::Real, μ::Real, σ::Real) = 1/(σ*sqrt(2*pi))*exp(-1/2*(x-μ)^2/σ^2)

# bootstrapping stuff
function bs_percentile_ci(samples::Vector, α::Real) 
	sorted = sort(samples)
	n = length(samples)
	return sorted[floor(Int,n*α/2)], sorted[floor(Int,n*(1-α/2))]
end

function mean_and_ci(rocs::Vector, α::Real, measure::Function, args...)
	vals = map(x->measure(x..., args...), rocs)
	return mean(vals), bs_percentile_ci(vals, α)
end
function mean_and_ci(rocs::Vector, α::Real, fprs::Vector, measure::Function, args...)
	vals = map(fpr->mean_and_ci(rocs, α, measure, fpr, args...), fprs)
	return [val[1] for val in vals], [val[2] for val in vals]
end

# GMM fitting
pdf(gmm::GMM, x::Vector) = reduce(+,map(i->normal_density.(x, gmm.μ[i], sqrt(gmm.Σ[i])), 1:gmm.n).*gmm.w)
pdf(gmm::GMM, x::Real) = pdf(gmm, [x])[1]
BIC(gmm::GMM, x::Vector) = log(length(x))*2*gmm.n - sum(log.(pdf(gmm, x)))
max_n(gmm::GMM, x::Vector) = -gmm.n
function select_optimal_gmm(scores::Vector, Nmax::Int, criterion)
	gmms = [GMM(n, scores) for n in 1:Nmax]
	critvals = map(m->criterion(m, scores), gmms)
	return gmms[argmin(critvals)]
end
sample(gmm::GMM) = (n=sample(1:gmm.n, Weights(gmm.w)); randn()*sqrt(gmm.Σ[n])+gmm.μ[n])
sample(gmm::GMM, N::Int) = [sample(gmm) for _ in 1:N]

function roc_from_samples(gmm0::GMM, gmm1::GMM, Ns::Union{Int,Tuple,Vector})
	Ns = (length(Ns) == 1) ? (Ns, Ns) : Ns
	scores = vcat(sample(gmm0, Ns[1]), sample(gmm1, Ns[2]))
	y = vcat(zeros(Ns[1]), ones(Ns[2]))
	return EvalCurves.roccurve(scores, y)
end

function plot_cis_etc(roc, auc, a, b, fprs, y_tst, scores, xvec_hist, density0, density1,
	tpr_means, tpr_cis, pauc_means, pauc_cis)
	figure(figsize=(16,8))
	# scatter empirical points + parametrized roc
	subplot(241)
	scatter(roc..., s=1)
	ylim([0,1.02])
	xlim([-.02,1])
	# replace this with a proper fit!
	xvec=0:0.01:1;
	par_roc = roc_curve_parametrized.(xvec,a,b);
	plot(xvec, par_roc,"r")
	auc = round(auc,digits=3)
	a = round(a,digits=3)
	b = round(b,digits=3)
	title("AUC=$auc, a=$a, b=$b")

	subplot(245)
	hist(scores[y_tst.==1],color="r", alpha=0.4,nbins,density=true,label="pos");
	hist(scores[y_tst.==0],color="b", alpha=0.4,nbins,density=true,label="neg");
	plot(xvec_hist, density0, "b")
	plot(xvec_hist, density1, "r")
	legend()

	subplot(242)
	title("TPR@FPR 95% CI")
	plot(fprs, tpr_means)
	fill_between(fprs, [x[1] for x in tpr_cis], [x[2] for x in tpr_cis], alpha=0.3)
	xlim([0,0.2])
	ylabel("TPR@FPR")

	subplot(246)
	plot(fprs, [x[2]-x[1] for x in tpr_cis])
	xlim([0,0.2])
	xlabel("FPR")
	ylabel("CI width")

	subplot(243)
	title("AUC@FPR 95% CI")
	plot(fprs, pauc_means)
	fill_between(fprs, [x[1] for x in pauc_cis], [x[2] for x in pauc_cis], alpha=0.3)
	xlim([0,0.2])
	ylabel("AUC@FPR")

	subplot(247)
	plot(fprs, [x[2]-x[1] for x in pauc_cis])
	xlim([0,0.2])
	xlabel("FPR")
	ylabel("CI width")

	subplot(244)
	title("TPR@FPR vs AUC@FPR")
	plot(fprs, tpr_means, label="TPR@FPR")
	fill_between(fprs, [x[1] for x in tpr_cis], [x[2] for x in tpr_cis], alpha=0.3)
	plot(fprs, pauc_means, label="AUC@FPR")
	fill_between(fprs, [x[1] for x in pauc_cis], [x[2] for x in pauc_cis], alpha=0.3)
	xlim([0,0.2])
	legend()

	subplot(248)
	plot(fprs, [x[2]-x[1] for x in tpr_cis]./tpr_means, label="TPR@FPR")
	plot(fprs, [x[2]-x[1] for x in pauc_cis]./pauc_means, label="AUC@FPR")
	xlabel("FPR")
	ylabel("relative CI width")
	xlim([0,0.2])
	tight_layout()
end

function compare_cis(roc, auc, a, b, fprs, y_tst, scores, density0, density1,
	bs_tpr_means, bs_tpr_cis, gmm_tpr_means, gmm_tpr_cis,
	bs_pauc_means, bs_pauc_cis, gmm_pauc_means, gmm_pauc_cis)
	figure(figsize=(8,8))
	# scatter empirical points + parametrized roc
	subplot(221)
	scatter(roc..., s=1)
	ylim([0,1.02])
	xlim([-.02,1])
	# replace this with a proper fit!
	xvec=0:0.01:1;
	par_roc = roc_curve_parametrized.(xvec,a,b);
	plot(xvec, par_roc,"r")
	auc = round(auc,digits=3)
	a = round(a,digits=3)
	b = round(b,digits=3)
	title("AUC=$auc, a=$a, b=$b")

	subplot(223)
	title("GMM fit of scores")
	hist(scores[y_tst.==1],color="r", alpha=0.4,nbins,density=true,label="pos");
	hist(scores[y_tst.==0],color="b", alpha=0.4,nbins,density=true,label="neg");
	plot(xvec_hist, density0, "b")
	plot(xvec_hist, density1, "r")
	legend()

	subplot(222)
	title("95% confidence interval widths")
	plot(fprs, [x[2]-x[1] for x in bs_tpr_cis], label="TPR@FPR-bootstrap")
	plot(fprs, [x[2]-x[1] for x in gmm_tpr_cis], label="TPR@FPR-gmm")
	plot(fprs, [x[2]-x[1] for x in bs_pauc_cis], label="AUC@FPR-bootstrap")
	plot(fprs, [x[2]-x[1] for x in gmm_pauc_cis], label="AUC@FPR-gmm")
	xlabel("FPR")
	ylabel("CI width")
	xlim([0,0.2])
	legend()

	subplot(224)
	plot(fprs, [x[2]-x[1] for x in bs_tpr_cis]./bs_tpr_means, label="TPR@FPR-bootstrap")
	plot(fprs, [x[2]-x[1] for x in gmm_tpr_cis]./gmm_tpr_means, label="TPR@FPR-gmm")
	plot(fprs, [x[2]-x[1] for x in bs_pauc_cis]./bs_pauc_means, label="AUC@FPR-bootstrap")
	plot(fprs, [x[2]-x[1] for x in gmm_pauc_cis]./gmm_pauc_means, label="AUC@FPR-gmm")
	xlabel("FPR")
	ylabel("relative CI width")
	xlim([0,0.2])
	legend()
	tight_layout()
end
