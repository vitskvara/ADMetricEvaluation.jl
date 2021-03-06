"""
    threshold_at_fpr_sample(scores::Vector, y_true::Vector, fpr::Real, d::Real[; warns=true])

Subsample the input `scores` to obtain a random threshold value for a given `fpr` value.
Argument `d` is the relative number of the subsampled scores.
"""
function threshold_at_fpr_sample(scores::Vector, y_true::Vector, fpr::Real, d::Real; warns=true)
    @assert 0 <= d <= 1
    n = length(y_true)
    nn = floor(Int, d*n)
    sample_inds = StatsBase.sample(1:n, nn, replace=false)
    EvalCurves.threshold_at_fpr(scores[sample_inds], y_true[sample_inds], fpr, warns=warns)
end

"""
    fpr_distribution(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int, d::Real=0.5[; warns=true])

Computes the distribution of false positive rates around given `fpr`.
"""
function fpr_distribution(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int, d::Real=0.5; warns=true)
    thresholds = map(i->threshold_at_fpr_sample(scores, y_true, fpr, d; warns=warns), 1:nsamples)
    fprs = map(x->EvalCurves.fpr_at_threshold(scores, y_true, x), thresholds)
end

"""
    localized_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int[; d::Real=0.5,
        normalize=false])

A localized AUC. Samples the distribution of false positive rate values around given `fpr`. Then
integrates the ROC curve only between found minimum and maximum value.

# Arguments
* `scores`: A numerical vector of sample scores, higher value indicates higher
probability of belonginig to the positive class
* `y_true`: True binary labels
* `fpr`: Value of FPR of interest
* `nsamples`: Number of FPR samples to be fitted
* `d`: A ratio of the size of the resampling set from which the distribution samples are drawn from
* `normalize`: Normalize the output?

"""
function localized_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, 
    normalize = true, warns=true)
    # first sample fprs and get parameters of the beta distribution
    fprs = fpr_distribution(scores, y_true, fpr, nsamples, d, warns=warns)
    
    # filter out NaNs
    fprs = fprs[.!isnan.(fprs)]
    (length(fprs) == 0) ? (return NaN) : nothing

    # check consistency
    if !_check_sampled_fpr_consistency(fpr, fprs; nsigma=2)
        warns ? (@warn "the requested fpr is out of the sampled fpr distribution, returning NaN") : nothing
        return NaN
    end

    # now continue
    roc = EvalCurves.roccurve(scores, y_true)
    EvalCurves.partial_auc(roc..., maximum(fprs), minimum(fprs), normalize=normalize)
end

"""
    estimate_beta_params(x::Vector)

Given a set of samples `x`, estimate the parameters of the one-dimensional Beta(α,β) distribution
as in `https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters`. 
"""
function estimate_beta_params(x::Vector)
    μ, σ2 = mean_and_var(x)
    if σ2 < μ*(1-μ)
        α = μ*(μ*(1-μ)/σ2 - 1)
        β = (1-μ)*(μ*(1-μ)/σ2 - 1)
    else
        α, β = NaN, NaN
    end
    return α, β
end

"""
    beta_logpdf(x::Real, α::Real, β::Real)

Log-pdf of Beta(α, β) distribution. Equals to log of x^(α-1)*(1-x)^(β-1)*Γ(α+β)/Γ(α)*Γ(β).
"""
beta_logpdf(x::Real, α::Real, β::Real) = 
    (α-1)*log(x) + (β-1)*log(1-x) + loggamma(α+β) - loggamma(α) - loggamma(β)

"""
    beta_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5)

Computes βAUC - an integral over ROC weighted by a Beta distribution of false positive rates
around `fpr`.

# Arguments
* `scores`: A numerical vector of sample scores, higher value indicates higher
probability of belonginig to the positive class
* `y_true`: True binary labels
* `fpr`: Value of FPR of interest
* `nsamples`: Number of FPR samples to be fitted
* `d`: A ratio of the size of the resampling set from which the distribution samples are drawn from

"""
function beta_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    # first sample fprs and get parameters of the beta distribution
    fprs = fpr_distribution(scores, y_true, fpr, nsamples, d, warns=warns)
    # filter out NaNs
    fprs = fprs[.!isnan.(fprs)]
    (length(fprs) == 0) ? (return NaN) : nothing

    # check for consistency
    if !_check_sampled_fpr_consistency(fpr, fprs; nsigma=2)
        warns ? (@warn "the requested fpr is out of the sampled fpr distribution, returning NaN") : nothing
        return NaN
    end

    # now continue
    α, β = estimate_beta_params(fprs)

    # compute roc
    roc = EvalCurves.roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = exp.(beta_logpdf.(xw, α, β))
    wauroc = EvalCurves.auc(roci..., w)
end

function _check_sampled_fpr_consistency(fpr, fprs; nsigma=3)
    # test whether the fprs actually make any sense by a nsigma test
    m, s2 = mean_and_var(fprs)
    return (m-nsigma*sqrt(s2) < fpr < m+nsigma*sqrt(s2))
end

"""
    linear_interpolation(x::Vector,y::Vector;n=nothing,dx=nothing)

Linearly interpolate `x` and `y`.
"""
function linear_interpolation(x::Vector,y::Vector;n=nothing,dx=nothing)
    (n==nothing && dx==nothing) ? 
        error("Support one of the keyword arguments - number of steps `n` or step length `dx`") : nothing
    _x = (dx == nothing) ?
        collect(range(minimum(x), maximum(x), length=n)) : collect(range(minimum(x), maximum(x), step=dx))
    n = length(_x)
    _y = zeros(n)
    # now for an element in _x, compute the element in _y as a linear interpolation between the 
    # two elements in y
    for i in 1:n
        if i==1
            _y[i] = y[1]
        elseif i==n
            _y[i] = y[end]
        else
            ri = findfirst(_x[i].<x)
            li = findlast(_x[i].>=x)
            _y[i] = y[li] + (y[ri] - y[li])/(x[ri]-x[li])*(_x[i]-x[li])
        end
    end
    return _x, _y
end

"""
    gaussian_pdf(nd::Real,p::Real,n::Int)

Gaussian pdf that is an approximation of the binomial distribution.
"""
gaussian_pdf(nd::Real,p::Real,n::Int) = 1/(sqrt(2*pi)*sqrt(n*p*(1-p)))*exp(-(nd-n*p)^2/(2*n*p*(1-p)))*n

"""
    gauss_auc(scores::Vector, y_true::Vector, fpr::Real)

An integral of the ROC curve where the weights are given by a Gaussian with width based on number of samples 
and location on the FPR axis.
"""
function gauss_auc(scores::Vector, y_true::Vector, fpr::Real)
    n = length(y_true) - sum(y_true) # number of negative samples

    # compute roc
    roc = EvalCurves.roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = gaussian_pdf.(xw.*n, fpr, n)

    wauroc = EvalCurves.auc(roci..., w)
end

"""
    empirical_histogram_weights(x::Vector, samples::Vector, rounding=8)

Weights given by an empirical histogram of fprs. Used in `hist_auc`.
"""
function empirical_histogram_weights(x::Vector, samples::Vector, rounding=8)
    # do some rounding to ensure equalities
    x = round.(x, digits=rounding)
    dxs = round.(x[2:end].-x[1:end-1], digits=rounding)
    samples = round.(samples, digits = rounding)

    N = length(dxs)
    w = zeros(N)
    for i in 1:N
        if dxs[i] != 0
            w[i] = sum(x[i] .<= samples .< x[i+1])/length(samples)/dxs[i]
        end
    end
    return w
end

"""
    hist_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)

An integral of the ROC curve where the weights are given by an empirical histogram of FPR values sampled
around given `fpr`.
"""
function hist_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    # first sample fprs and get parameters of the beta distribution
    fprs = fpr_distribution(scores, y_true, fpr, nsamples, d, warns=warns)
    # filter out NaNs
    fprs = fprs[.!isnan.(fprs)]
    (length(fprs) == 0) ? (return NaN) : nothing

    # check for consistency
    if !_check_sampled_fpr_consistency(fpr, fprs; nsigma=2)
        warns ? (@warn "the requested fpr is out of the sampled fpr distribution, returning NaN") : nothing
        return NaN
    end

    # compute roc
    roc = EvalCurves.roccurve(scores, y_true)
    
    # compute the histogram weights
    w = empirical_histogram_weights(roc[1], fprs)

    # compute the integral
    wauroc = EvalCurves.auc(roc..., w)
end
