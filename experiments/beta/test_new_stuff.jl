include("test_beta_alternatives_functions.jl")
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
fprs = [0.01, 0.05]
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
row_measures = [:auc, :auc_weighted, :auc_at_1, :tpr_at_1, :bauc_at_1, :lauc_at_1, :measure_at_1,
		:auc_at_5, :tpr_at_5, :bauc_at_5, :lauc_at_5, :measure_at_5]
column_measures = [:auc_at_1, :auc_at_5, :prec_at_1, :prec_at_5,
		:tpr_at_1, :tpr_at_5]

dataset = "ecoli"
subdataset = "ecoli-cp-imL"

dataset = "ecoli"
subdataset = "ecoli-cp-imL"
seed = 1
X_tr, y_tr, X_val, y_val, X_tst, y_tst = get_data(dataset, subdataset, seed)

model = OCSVM_model(0.1)
ScikitLearn.fit!(model, Array(transpose(X_tr)))
score_fun(X) = -ScikitLearn.decision_function(model, Array(transpose(X)))
scores = score_fun(X_val)
fprvec, tprvec = roccurve(scores, y_val)

fpr = 0.05
beta_auc(scores, y_val, fpr, 1000)

fprs = EvalCurves.fpr_distribution(scores, y_val, 0.05, 1000, 0.5)
α, β = EvalCurves.estimate_beta_params(fprs)
roc = EvalCurves.roccurve(scores, y_val)
interp_len = max(1001, length(roc[1]))
roci = EvalCurves.linear_interpolation(roc..., n=interp_len)

# weights are given by the beta pdf and are centered on the trapezoids
dx = (roci[1][2] - roci[1][1])/2
xw = roci[1][1:end-1] .+ dx
w = exp.(EvalCurves.beta_logpdf.(xw, α, β))

q  = 1/2

figure()
subplot(121)
title("ROC")
plot(roc...)
subplot(122)
hist(fprs, density = true)
plot(xw, w)
plot(xw, w.^q)

#try a different sampling strategy
# or a different (flatter) distribution


α, β = (2,2)
w = exp.(EvalCurves.beta_logpdf.(xw, α, β))

auc(xw, w.^2)

# so instead of beta weights, try weighing by the distributions from 
# Kerekes - ROC confidence interavals and regions
binomial_pdf(nd::Int,p::Real,n::Int) = binomial(n, nd)*p^nd*(1-p)^(n-nd)
gaussian_pdf(x::Real, μ::Real, σ::Real) = 1/(sqrt(2*pi)*σ)*exp(-(x-μ)^2/(2*σ^2))
gaussian_pdf(nd::Int,p::Real,n::Int) = 1/(sqrt(2*pi)*sqrt(n*p*(1-p)))*exp(-(nd-n*p)^2/(2*n*p*(1-p)))*n
gaussian_pdf(nd::Real,p::Real,n::Int) = 1/(sqrt(2*pi)*sqrt(n*p*(1-p)))*exp(-(nd-n*p)^2/(2*n*p*(1-p)))*n
poisson_pdf(nd::Int,p::Real,n::Int) = (n*p)^nd/factorial(nd)*exp(-n*p)
function weights_kerekes(x, p, n)
	# nd = n*p
	nd = round(Int, x*n)
	if n <= 100
		return binomial_pdf(nd,p,n)
#	elseif p < 0.1
#		return poisson_pdf(nd,p,n)
	else
		nd = x*n
		return gaussian_pdf(nd,p,n)
	end
end

w = weights_kerekes.(xw, 0.05, 200)

w = gaussian_pdf.(xw, 0.05, 0.01)

