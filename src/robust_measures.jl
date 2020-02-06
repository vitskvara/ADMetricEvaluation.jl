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

gmm_bootstrapped_measure(rocs::Vector, measure::Function) = mean(map(r->measure(r...), rocs))

# GMM component selection criterions
# this tends to select 1 component
BIC(gmm::GMM, x::Vector) = log(length(x))*2*gmm.n - sum(log.(pdf(gmm, x)))
max_n(gmm::GMM, x::Vector) = -gmm.n
