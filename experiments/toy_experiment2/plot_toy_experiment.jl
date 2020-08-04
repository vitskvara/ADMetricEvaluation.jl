# run as `julia plot_toy_experiments.jl` in the dir where the data is
using FileIO, BSON
using DrWatson
using PyPlot, Statistics

files = readdir()
files = filter(x->occursin("bson", x) && occursin("p-", x) && occursin("N-", x), files)

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

function create_plot(f)
	splits = split(f, "_")
	p = Meta.parse(split(splits[4], "-")[2])
	fpr = Meta.parse(split(splits[5], "-")[2])
	N = Meta.parse(split(split(splits[6], "-")[2], ".")[1])
	data = load(f)
	result, ns = data[:result], data[:ns]

	# plot it again
	n = 1000000
	scores1, labels1 = scores1_and_labels(n, p)
	scores2, labels2 = scores2_and_labels(n)

	figure(figsize = (12,13))
	for (icol, m) in enumerate(["bstpr", "gauc", "pauc2"])
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

		# plot it
		subplot(5,3,icol)
		title("p=$(p), fpr=$(fpr), N=$(N)")
		hist(scores1[labels1 .== 0], 50, alpha=0.5, label = "negative", density = true)
		hist(scores1[labels1 .== 1], 50, alpha=0.5, label = "positive", density = true)
		legend()

		subplot(5,3,icol+3)
		hist(scores2[labels2 .== 0], 50, alpha=0.5, label = "negative", density = true)
		hist(scores2[labels2 .== 1], 50, alpha=0.5, label = "positive", density = true)
		legend()

		subplot(5,3,icol+6)
		plot(log10.(ns), mtpr1, label = "TPR 1")
		plot(log10.(ns), mx1, label = "$(m) 1")
		plot(log10.(ns), mtpr2, label = "TPR 2")
		plot(log10.(ns), mx2, label = "$(m) 2")
		xlabel("log10(n)")
		ylabel("TPR@$(fpr)")
		legend()

		subplot(5,3,icol+9)
		plot(log10.(ns), p1, label = "P($(m)1 > TPR1)")
		plot(log10.(ns), p2, label = "P($(m)2 > TPR2)")
		xlabel("log10(n)")
		legend()

		subplot(5,3,icol+12)
		plot(log10.(ns), pp1, label = "P(TPR1 > TPR2)")
		plot(log10.(ns), pp2, label = "P($(m)1 > $(m)2)")
		xlabel("log10(n)")
		legend()		
	end
	tight_layout()
	savefig("toy_problem_N-$(N)_fpr-$(fpr)_p-$(p).png")
end

map(create_plot, files)