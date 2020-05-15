function bootstrapped_measure(score_fun::Function, measure::Function, X_test::Array, 
	y_test::Vector, N_samples::Int)
	m = size(X_test,2)
	inds = map(n->StatsBase.sample(1:m,m), 1:N_samples)
	scores = map(is -> score_fun(X_test[:,is]), inds)
	rocs = map(x -> EvalCurves.roccurve(x[1], y_test[x[2]]), zip(scores, inds))
	vals = map(r -> measure(r...), rocs)
	return mean(vals)
end

# gmm bootstrapping
StatsBase.sample(gmm::GMM) = (n=StatsBase.sample(1:gmm.n, Weights(gmm.w)); randn()*sqrt(gmm.Σ[n])+gmm.μ[n])
StatsBase.sample(gmm::GMM, N::Int) = [StatsBase.sample(gmm) for _ in 1:N]

function fit_gmms_sample_rocs(scores::Vector, y_test::Vector, 
	N_samples::Int, N_components::Int, 
	N_class_samples::Union{Nothing,Int,Tuple,Vector}=nothing)
	# fit two gmm models (maybe fit more and try the best according to a criterion, 
	# but that is slow and more components are usually better)
	local mix0, mix1
	@suppress begin
		mix0 = GMM(N_components, scores[y_test.==0])
		mix1 = GMM(N_components, scores[y_test.==1])
	end
	
	# how many of each class should be sampled
	if N_class_samples == nothing
		N1 = sum(y_test)
		N0 = length(y_test) - N1
	elseif length(N_class_samples) == 1
		N1 = N0 = N_class_samples
	else
		N0, N1 = N_class_samples
	end

	# now sample scores, compute rocs
	scores = [vcat(StatsBase.sample(mix0, N0), StatsBase.sample(mix1, N1)) for _ in 1:N_samples]
	y_samples = vcat(zeros(N0), ones(N1))
	rocs = map(s->EvalCurves.roccurve(s, y_samples), scores)
end

measure_mean(rocs::Vector, measure::Function) = mean(map(r->measure(r...), rocs))

# GMM component selection criterions
# this tends to select 1 component
"""
	BIC(gmm::GMM, x::Vector)

Bayesian information criterion for selecting the optimal number of components of GMM.
"""
BIC(gmm::GMM, x::Vector) = log(length(x))*2*gmm.n - sum(log.(pdf(gmm, x)))

"""
	max_n(gmm::GMM, x::Vector)

Selects the model with alrgest number of components.
"""
max_n(gmm::GMM, x::Vector) = -gmm.n

"""
	normal_density(x::Real, μ::Real, σ::Real)

Density of a normal distribution.
"""
normal_density(x::Real, μ::Real, σ::Real) = 1/(σ*sqrt(2*pi))*exp(-1/2*(x-μ)^2/σ^2)

"""
	pdf(gmm::GMM, x)

The pdf of a GMM model.
"""
pdf(gmm::GMM, x::Vector) = reduce(+,map(i->normal_density.(x, gmm.μ[i], sqrt(gmm.Σ[i])), 1:gmm.n).*gmm.w)
pdf(gmm::GMM, x::Real) = pdf(gmm, [x])[1]

"""
	gmm_fit(scores::Vector, max_components::Int; verb=false, kwargs...)

Fits a 1-D data vector with a GMM model. Starts with `max_components` and progresses down from this if 
the fit fails.
"""
function gmm_fit(scores::Vector, max_components::Int; verb=false, kwargs...)
    # basically, try fitting gmms from max components down to 1
    for nc in max_components:-1:1
        gmm = try
            if !verb
                @suppress begin
                    GMM(nc, scores; kwargs...)
                end
            else
                GMM(nc, scores; kwargs...)
            end
        catch e    
            @warn(e)
            nothing
        end
        (gmm != nothing) ? (return gmm) : nothing
    end
end
function gmm_fit(scores::Vector, y_true::Vector, max_components::Int; verb=false, kwargs...)
    gmm0 = gmm_fit(scores[y_true.==0], max_components, verb=verb, kwargs...)
    gmm1 = gmm_fit(scores[y_true.==1], max_components, verb=verb, kwargs...)
    return gmm0, gmm1
end

"""
	tpr_at_fpr_gmm(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; 
        min_samples::Int=1000, nc::Int=5, warns = true)

GMM version of tpr@fpr.
"""
function tpr_at_fpr_gmm(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; 
        min_samples::Int=1000, nc::Int=5, warns = true)
    # fit the gmms
    gmm0, gmm1 = gmm_fit(scores, y_true, nc);
    if (gmm0 == nothing) || (gmm1 == nothing)
        return NaN
    end
    
    # sample scores
    N = max(min_samples, length(scores))
    
    # get rocs and the respective tpr values
    ts = map(_->EvalCurves.tpr_at_fpr(roccurve(vcat(StatsBase.sample(gmm0, N), StatsBase.sample(gmm1, N)), 
                vcat(zeros(N), ones(N)))..., fpr), 1:nrepeats)
    mean(ts)
end

"""
	auc_at_fpr_gmm(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; 
        min_samples::Int=1000, nc::Int=3, warns = true)

GMM version of AUC@fpr.
"""
function auc_at_fpr_gmm(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; 
        min_samples::Int=1000, nc::Int=3, warns = true)
    # fit the gmms
    gmm0, gmm1 = gmm_fit(scores, y_true, nc);
    if (gmm0 == nothing) || (gmm1 == nothing)
        return NaN
    end
    
    # sample scores
    N = max(min_samples, length(scores))
    
    # get rocs and the respective pauc values
    paucs = map(_->EvalCurves.auc_at_p(roccurve(vcat(StatsBase.sample(gmm0, N), StatsBase.sample(gmm1, N)), 
                vcat(zeros(N), ones(N)))..., fpr; normalize=true), 1:nrepeats)
    mean(paucs)
end

# bootstrapping
bootstrap_sample_inds(N::Int) = StatsBase.sample(1:N, N)
"""
	tpr_at_fpr_bootstrap(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; warns=false)

Bootstrap version of tpr@fpr.
"""
function tpr_at_fpr_bootstrap(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; warns=false)
    # get the indices
    inds = map(_->bootstrap_sample_inds(length(y_true)), 1:nrepeats)
    
    # get rocs and the respective tpr values
    ts = map(x->EvalCurves.tpr_at_fpr(roccurve(scores[x], y_true[x])..., fpr), inds)
    mean(ts)
end

"""
	auc_at_fpr_bootstrap(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; warns=false)

Bootstrap version of AUC@fpr.
"""
function auc_at_fpr_bootstrap(scores::Vector, y_true::Vector, fpr::Real, nrepeats::Int; warns=false)
    # sample bootstrap indices
    inds = map(_->bootstrap_sample_inds(length(y_true)), 1:nrepeats)
    
    # get rocs and the respective pauc values
    aucs = map(x->EvalCurves.auc_at_p(roccurve(scores[x], y_true[x])..., fpr; normalize=true), inds)
    mean(aucs)
end
