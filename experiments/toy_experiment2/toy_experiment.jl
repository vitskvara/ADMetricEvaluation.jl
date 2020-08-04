using PyPlot, Statistics, EvalCurves, ADMetricEvaluation
using ProgressMeter
using FileIO, BSON
using DrWatson

function scores1_negative(p1::Real)
	i = Int(rand() > p1) + 1
	return rand()*0.5 + [-1,1][i]
end
scores1_positive() = rand()*0.5
scores1_and_labels(n::Int, p1::Real) = 
	vcat(map(x->scores1_negative(p1), 1:n), map(x->scores1_positive(), 1:n)),
	vcat(zeros(Int, n), ones(Int, n))

scores2_negative() = rand()*0.8
scores2_positive() = rand() - 0.2
scores2_and_labels(n::Int) = 
	vcat(map(x->scores2_negative(), 1:n), map(x->scores2_positive(), 1:n)),
	vcat(zeros(Int, n), ones(Int, n))

# sample it
n = 1000000
p = 0.5
scores1, labels1 = scores1_and_labels(n, p)
scores2, labels2 = scores2_and_labels(n)

# plot it
figure()
subplot(121)
title("p = $(p)")
hist(scores1[labels1 .== 0], 50, alpha=0.5, label = "negative", density = true)
hist(scores1[labels1 .== 1], 50, alpha=0.5, label = "positive", density = true)
legend()

subplot(122)
title("TPR@0.5 = 0.4")
hist(scores2[labels2 .== 0], 50, alpha=0.5, label = "negative", density = true)
hist(scores2[labels2 .== 1], 50, alpha=0.5, label = "positive", density = true)
legend()
tight_layout()
savefig("distributions_toy.png")

# other test 
n = 10000
fpr = 0.5
sls = (scores1_and_labels(n, p), scores2_and_labels(n))
rocs = map(x->roccurve(x...), sls)
tprs = map(x->EvalCurves.tpr_at_fpr(x..., fpr), rocs)	

# load the data here
p = 0.51
fpr = 0.5
f = "toy_data_$(p).bson"
if isfile(f)
	using DrWatson
	@unpack result, ns = load(f)
else # or compute them
	#ns = vcat(collect(1:49), collect(50:10:99), collect(100:100:1000))
	ns =  unique(floor.(Int, 10 .^ (range(0,4, length=122))))
	#ns =  unique(floor.(Int, 10 .^ (range(0,5, length=114))))
	result = []
	for n in ns
		println(n)
		res = Dict(
				:tpr1 => Float64[],
				:bstpr1 => Float64[],
				:tpr2 => Float64[],
				:bstpr2 => Float64[]
			)
		@showprogress for i in 1:100
			sls = (scores1_and_labels(n, p), scores2_and_labels(n))
			rocs = map(x->roccurve(x...), sls)
			tprs = map(x->EvalCurves.tpr_at_fpr(x..., fpr), rocs)	
			bstprs = map(x->ADMetricEvaluation.tpr_at_fpr_bootstrap(x..., fpr, 1000), sls)
			push!(res[:tpr1], tprs[1])
			push!(res[:tpr2], tprs[2])
			push!(res[:bstpr1], bstprs[1])
			push!(res[:bstpr2], bstprs[2])
		end
		push!(result, res)
	end
	save(f, :result => result, :ns => ns)
end

i = 60
n = ns[i]
subplot(221)
title("TPR1")
hist(result[i][:tpr1])
subplot(222)
title("TPR2")
hist(result[i][:tpr2])
subplot(223)
title("BSTPR1")
hist(result[i][:bstpr1])
subplot(224)
title("BSTPR2")
hist(result[i][:bstpr2])
savefig("histograms_p_$(p)_n_$(n).png")


mbstpr1 = [mean(d[:bstpr1]) for d in result]
mbstpr2 = [mean(d[:bstpr2]) for d in result]
mtpr1 = [mean(d[:tpr1]) for d in result]
mtpr2 = [mean(d[:tpr2]) for d in result]

# here compute the probability that bs tpr > tpr
# do it on ns
p1 = [mean(d[:bstpr1] .> d[:tpr1]) for d in result]
p2 = [mean(d[:bstpr2] .> d[:tpr2]) for d in result]

pp1 = [mean(d[:tpr1] .> d[:tpr2]) for d in result]
pp2 = [mean(d[:bstpr1] .> d[:bstpr2]) for d in result]

# plot it again
n = 1000000
scores1, labels1 = scores1_and_labels(n, p)
scores2, labels2 = scores2_and_labels(n)

# plot it
figure(figsize=(4,13))
subplot(511)
title("p = $(p)")
hist(scores1[labels1 .== 0], 50, alpha=0.5, label = "negative", density = true)
hist(scores1[labels1 .== 1], 50, alpha=0.5, label = "positive", density = true)
legend()

subplot(512)
title("TPR@0.5 = 0.4")
hist(scores2[labels2 .== 0], 50, alpha=0.5, label = "negative", density = true)
hist(scores2[labels2 .== 1], 50, alpha=0.5, label = "positive", density = true)
legend()

subplot(513)
plot(log10.(ns), mtpr1, label = "TPR 1")
plot(log10.(ns), mbstpr1, label = "BS TPR 1")
plot(log10.(ns), mtpr2, label = "TPR 2")
plot(log10.(ns), mbstpr2, label = "BS TPR 2")
xlabel("log10(n)")
ylabel("TPR@0.5")
legend()

subplot(514)
plot(log10.(ns), p1, label = "P(BSTPR1 > TPR1)")
plot(log10.(ns), p2, label = "P(BSTPR2 > TPR2)")
xlabel("log10(n)")
legend()

subplot(515)
plot(log10.(ns), pp1, label = "P(TPR1 > TPR2)")
plot(log10.(ns), pp2, label = "P(BSTPR1 > BSTPR2)")
xlabel("log10(n)")
legend()

tight_layout()
savefig("toy_problem_p_$(p).png")


# histograms
data49 = load("toy_data_0.49.bson")
result49, ns49 = data49[:result], data49[:ns]
data51 = load("toy_data_0.51.bson")
result51, ns51 = data51[:result], data51[:ns]

i = 55
n = ns51[i]

function score_histograms(result, spt)
	suptitle(spt)
	subplot(221)
	title("TPR1")
	hist(result[:tpr1])
	subplot(222)
	title("TPR2")
	hist(result[:tpr2])
	subplot(223)
	title("BSTPR1")
	hist(result[:bstpr1])
	subplot(224)
	title("BSTPR2")
	hist(result[:bstpr2])
end

figure()
score_histograms(result49[i], "n=$(ns49[i]), p=0.49")
figure()
score_histograms(result51[i], "n=$(ns51[i]), p=0.51")

sum(result51[i][:bstpr1] .< 0.6)

# now repeat for gauss_auc
p = 0.51
fpr = 0.5
f = "toy_data_other_$(p).bson"
if isfile(f)
	using DrWatson
	@unpack result, ns = load(f)
else # or compute them
	#ns = vcat(collect(1:49), collect(50:10:99), collect(100:100:1000))
	ns =  unique(floor.(Int, 10 .^ (range(0,4, length=122))))
	#ns =  unique(floor.(Int, 10 .^ (range(0,5, length=114))))
	result = []
	for n in ns
		println(n)
		res = Dict(
				:tpr1 => Float64[],
				:pauc1 => Float64[],
				:pauc21 => Float64[],
				:gauc1 => Float64[],
				:tpr2 => Float64[],
				:pauc2 => Float64[],
				:pauc22 => Float64[],
				:gauc2 => Float64[]
				)
		@showprogress for i in 1:100
			sls = (scores1_and_labels(n, p), scores2_and_labels(n))
			rocs = map(x->roccurve(x...), sls)
			tprs = map(x->EvalCurves.tpr_at_fpr(x..., fpr), rocs)	
			paucs = map(x->EvalCurves.auc_at_p(x..., fpr, normalize=true), rocs)
			paucs2 = map(x->EvalCurves.auc_at_p(x..., 2*fpr, normalize=true), rocs)
			gaucs = map(x->ADMetricEvaluation.gauss_auc(x..., fpr), sls)
			push!(res[:tpr1], tprs[1])
			push!(res[:tpr2], tprs[2])
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
end

function plot_result(result, m)
	m1 = Symbol("$(m)1")
	m2 = Symbol("$(m)2")
	mx1 = [mean(d[m1]) for d in result]
	mx2 = [mean(d[m2]) for d in result]
	mtpr1 = [mean(d[:tpr1]) for d in result]
	mtpr2 = [mean(d[:tpr2]) for d in result]

	# here compute the probability that bs tpr > tpr
	# do it on ns
	p1 = [mean(d[m1] .> d[:tpr1]) for d in result]
	p2 = [mean(d[m2] .> d[:tpr2]) for d in result]

	pp1 = [mean(d[:tpr1] .> d[:tpr2]) for d in result]
	pp2 = [mean(d[m1] .> d[m2]) for d in result]

	# plot it again
	n = 1000000
	scores1, labels1 = scores1_and_labels(n, p)
	scores2, labels2 = scores2_and_labels(n)

	# plot it
	figure(figsize=(4,13))
	subplot(511)
	title("p = $(p)")
	hist(scores1[labels1 .== 0], 50, alpha=0.5, label = "negative", density = true)
	hist(scores1[labels1 .== 1], 50, alpha=0.5, label = "positive", density = true)
	legend()

	subplot(512)
	title("TPR@0.5 = 0.4")
	hist(scores2[labels2 .== 0], 50, alpha=0.5, label = "negative", density = true)
	hist(scores2[labels2 .== 1], 50, alpha=0.5, label = "positive", density = true)
	legend()

	subplot(513)
	plot(log10.(ns), mtpr1, label = "TPR 1")
	plot(log10.(ns), mx1, label = "$(m) 1")
	plot(log10.(ns), mtpr2, label = "TPR 2")
	plot(log10.(ns), mx2, label = "$(m) 2")
	xlabel("log10(n)")
	ylabel("TPR@0.5")
	legend()

	subplot(514)
	plot(log10.(ns), p1, label = "P($(m)1 > TPR1)")
	plot(log10.(ns), p2, label = "P($(m)2 > TPR2)")
	xlabel("log10(n)")
	legend()

	subplot(515)
	plot(log10.(ns), pp1, label = "P(TPR1 > TPR2)")
	plot(log10.(ns), pp2, label = "P($(m)1 > $(m)2)")
	xlabel("log10(n)")
	legend()

	tight_layout()
	savefig("toy_problem_$(m)_p_$(p).png")
end

plot_result(result, "gauc")
plot_result(result, "pauc")
plot_result(result, "pauc2")

# also, plot what the rocs look like for different models1/2
function roc_scores(scores, labels, fpr)
	roc = roccurve(scores, labels)
	tpr = EvalCurves.tpr_at_fpr(roc..., fpr)
	pauc = EvalCurves.auc_at_p(roc..., fpr, normalize=true)
	pauc2 = EvalCurves.auc_at_p(roc..., 2*fpr, normalize=true)
	gauc = ADMetricEvaluation.gauss_auc(scores, labels, fpr)
	return roc, tpr, pauc, pauc2, gauc
end
	
n = 10
s = 2.5
px = range(0,1,length=500)
figure()
suptitle("n=$(n)")
subplot(221)
title("p=0.49")
scores, labels = scores1_and_labels(n, 0.49)
roc, tpr, pauc, pauc2, gauc = roc_scores(scores, labels, fpr)
plot(roc...)
text(0.6, 0.5, "tpr=$(round(tpr,digits=2))\npauc=$(round(pauc,digits=2))\npauc2=$(round(pauc2,digits=2))\ngauc=$(round(gauc,digits=2))")
n0 = sum(labels .== 0)
py = ADMetricEvaluation.gaussian_pdf.(px .* n0, fpr, n0)/s
plot(px,py)
subplot(222)
title("p=0.51")
scores, labels = scores1_and_labels(n, 0.51)
roc, tpr, pauc, pauc2, gauc = roc_scores(scores, labels, fpr)
plot(roc...)
text(0.6, 0.5, "tpr=$(round(tpr,digits=2))\npauc=$(round(pauc,digits=2))\npauc2=$(round(pauc2,digits=2))\ngauc=$(round(gauc,digits=2))")
n0 = sum(labels .== 0)
py = ADMetricEvaluation.gaussian_pdf.(px .* n0, fpr, n0)/s
plot(px,py)
subplot(223)
title("p=0.49")
scores, labels = scores2_and_labels(n)
roc, tpr, pauc, pauc2, gauc = roc_scores(scores, labels, fpr)
plot(roc...)
text(0.6, 0.2, "tpr=$(round(tpr,digits=2))\npauc=$(round(pauc,digits=2))\npauc2=$(round(pauc2,digits=2))\ngauc=$(round(gauc,digits=2))")
n0 = sum(labels .== 0)
py = ADMetricEvaluation.gaussian_pdf.(px .* n0, fpr, n0)/s
plot(px,py)
savefig("integral_measures_n=$(n).png")
