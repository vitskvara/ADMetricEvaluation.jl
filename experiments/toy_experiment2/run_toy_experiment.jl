# run as `julia run_toy_experiment,jl p fpr [N]`
# where p is the ratio of the first class in the imbalanced classifier
# and fpr is the requested fpr level, and N is the maximum number of samples
# julia run_toy_experiment,jl 0.49 0.5
# julia run_toy_experiment,jl 0.51 0.5
# julia run_toy_experiment,jl 0.985 0.01
# julia run_toy_experiment,jl 0.995 0.01


using PyPlot, Statistics, EvalCurves, ADMetricEvaluation
using ProgressMeter
using FileIO, BSON

p = Meta.parse(ARGS[1])
fpr = Meta.parse(ARGS[2])
N = (length(ARGS) == 3) ? Int(Meta.parse(ARGS[3])) : 10000

function scores1_negative(p1::Real)
	i = Int(rand() > p1) + 1
	return rand()*0.5 + [-1,1][i]
end
scores1_positive() = rand()*0.5
scores1_and_labels(n::Int, p1::Real) = 
	vcat(map(x->scores1_negative(p1), 1:n), map(x->scores1_positive(), 1:n)),
	vcat(zeros(Int, n), ones(Int, n))

scores2_negative() = rand()
scores2_positive() = rand() - 0.1
scores2_and_labels(n::Int) = 
	vcat(map(x->scores2_negative(), 1:n), map(x->scores2_positive(), 1:n)),
	vcat(zeros(Int, n), ones(Int, n))

f = "toy_data_other_p-$(p)_fpr-$(fpr)_N-$(N).bson"
if N == 10000
	ns =  unique(floor.(Int, 10 .^ (range(0,4, length=122))))
elseif N == 100000
	ns =  unique(floor.(Int, 10 .^ (range(0,5, length=114))))
else
	# this will not ensure 100 unique points
	ns =  unique(floor.(Int, 10 .^ (range(0,log10(N), length=100))))
end
result = []
for n in ns
	println(n)
	res = Dict(
			:tpr1 => Float64[],
			:bstpr1 => Float64[],
			:pauc1 => Float64[],
			:pauc21 => Float64[],
			:gauc1 => Float64[],
			:tpr2 => Float64[],
			:bstpr2 => Float64[],
			:pauc2 => Float64[],
			:pauc22 => Float64[],
			:gauc2 => Float64[]
			)
	@showprogress for i in 1:100
		sls = (scores1_and_labels(n, p), scores2_and_labels(n))
		rocs = map(x->roccurve(x...), sls)
		tprs = map(x->EvalCurves.tpr_at_fpr(x..., fpr), rocs)	
		bstprs = map(x->ADMetricEvaluation.tpr_at_fpr_bootstrap(x..., fpr, 1000), sls)
		paucs = map(x->EvalCurves.auc_at_p(x..., fpr, normalize=true), rocs)
		paucs2 = map(x->EvalCurves.auc_at_p(x..., 2*fpr, normalize=true), rocs)
		gaucs = map(x->ADMetricEvaluation.gauss_auc(x..., fpr), sls)
		push!(res[:tpr1], tprs[1])
		push!(res[:tpr2], tprs[2])
		push!(res[:bstpr1], bstprs[1])
		push!(res[:bstpr2], bstprs[2])
		push!(res[:pauc1], paucs[1])
		push!(res[:pauc2], paucs[2])
		push!(res[:pauc21], paucs2[1])
		push!(res[:pauc22], paucs2[2])
		push!(res[:gauc1], gaucs[1])
		push!(res[:gauc2], gaucs[2])
	end
	push!(result, res)
end
save(f, :result => result, :ns => ns)
println("succesfuly saved the result to $(f)")
