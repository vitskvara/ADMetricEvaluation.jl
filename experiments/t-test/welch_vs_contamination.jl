include("plot_functions.jl")
contamination = 0.01
outpath = joinpath("/home/vit/vyzkum/measure_evaluation/welch_cont", 
	string(contamination))
mkpath(outpath)

base_dataset = "statlog-satimage"
sub = "1-3"
model_names = [:OCSVM, :kNN]
model_params = [[0.5], [21, :kappa]]

base_dataset = "yeast"
sub = "CYT-VAC"

base_dataset = "ecoli"
sub = "cp-im"

# some stuff
if model_names[1] != model_names[2]
	model_names_diff = copy(model_names)
else
	model_names_diff = map(x->Symbol(string(x[2])*"$(x[1])"), enumerate(model_names))
end

# get data
dataset_fname = base_dataset*"-"*sub
data, normal_labels, anomaly_labels = UCI.get_data(base_dataset);
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels);
data = filter(x->x[2]==sub,subdatasets)[1][1];
X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8, 0.01, seed=1, 
	standardize=true)
np = sum(y_tst)
nn = length(y_tst) - np

# construct models
models = map(x->eval(Symbol(String(x[1])*"_model"))(x[2]...), 
	zip(model_names, model_params))
model_fname = reduce(connect_und, map(x->reduce(connect_minus,vcat(x[1],x[2])),
		zip(model_names, model_params)))

function resample_test(X_tst, y_tst, contamination::Real)
	np = sum(y_tst)
	nn = length(y_tst) - np
	contamination
	(np/nn < contamination) ? error("too little anomalies in given test set") : nothing
	npn = floor(Int, nn*contamination)
	return hcat(X_tst[:,y_tst.==0], X_tst[:,y_tst.==1][:,rand(1:np,npn)]),
		vcat(zeros(Int,nn), ones(Int,npn))
end

fpr = 0.05
ks = collect(1:50)
conts = collect(range(0,min(1,np/nn),length=101))
n = ks[end]
α = 0.05
fname = joinpath(outpath, dataset_fname*"_"*model_fname*".jld2")
if isfile(fname)
	d = load(fname)
	measure_vals = d["measure_vals"]
	pauc_means, pauc_stds, pauc_welch, pauc_dfs, pauc_critvals, pauc_pvals = 
		d["pauc_stats"][:means], d["pauc_stats"][:stds], d["pauc_stats"][:welch],
		d["pauc_stats"][:dfs], d["pauc_stats"][:critvals], d["pauc_stats"][:vals]
	tpr_means, tpr_stds, tpr_welch, tpr_dfs, tpr_critvals, tpr_pvals = 
		d["tpr_stats"][:means], d["tpr_stats"][:stds], d["tpr_stats"][:welch],
		d["tpr_stats"][:dfs], d["tpr_stats"][:critvals], d["tpr_stats"][:vals]
else
	measure_vals = Dict{Symbol, Any}(
		:k => ks,
		:contamination => conts
		)
	for (model, model_name) in zip(models, model_names)
		atp = []
		tprtp = []
		r = []
		for k in ks
			auc_at_p = []
			tpr_at_p = []
			rocs = []
			X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8, 0.01, seed=k, 
				standardize=true)	
			ScikitLearn.fit!(model, Array(transpose(X_tr)))
			for c in conts
				_X_tst, _y_tst = resample_test(X_tst, y_tst, c)
				roc  = EvalCurves.roccurve(score_fun(model, _X_tst), _y_tst)
				push!(auc_at_p, EvalCurves.auc_at_p(roc...,fpr, normalize=true))
				push!(tpr_at_p, EvalCurves.tpr_at_fpr(roc..., fpr))
				push!(rocs, roc)
			end
			push!(atp, auc_at_p)
			push!(tprtp, tpr_at_p)
			push!(r, rocs)
		end
		measure_vals[model_name] = Dict(
			:auc_at_p => hcat(atp...),
			:tpr_at_p => hcat(tprtp...),
			:roc => r
			)
	end

	pauc_means, pauc_stds, pauc_welch, pauc_dfs, pauc_critvals, pauc_pvals = 
		get_statistics(:auc_at_p,n,α)
	tpr_means, tpr_stds, tpr_welch, tpr_dfs, tpr_critvals, tpr_pvals = 
			get_statistics(:tpr_at_p,n,α)
	save(fname, "measure_vals", measure_vals, 
			"tpr_stats", Dict(
				:means => tpr_means, 
				:stds => tpr_stds, 
				:welch => tpr_welch, 
				:dfs => tpr_dfs, 
				:critvals => tpr_critvals, 
				:vals => tpr_pvals
				),
			"pauc_stats", Dict(
				:means => pauc_means, 
				:stds => pauc_stds, 
				:welch => pauc_welch, 
				:dfs => pauc_dfs, 
				:critvals => pauc_critvals, 
				:vals => pauc_pvals
				))
end

# plot
colors = ["b", "g"]
figure(figsize=(8,8))
suptitle(dataset_fname*" "*model_fname*", np=$np, nn=$nn")

subplot(221)
for (i, model_name) in enumerate(model_names)
	plot(conts, pauc_means[i], c=colors[i], label=model_name)
	fill_between(conts, 
		vec(pauc_means[i].+1.96*pauc_stds[i]), 
		vec(pauc_means[i].-1.96*pauc_stds[i]), 
		color=colors[i], alpha=0.2)
end
xlabel("contamination")
ylabel("AUC@$(fpr)")
ylim([0,1])
legend(frameon=false)

subplot(222)
for (i, model_name) in enumerate(model_names)
	plot(conts, tpr_means[i], c=colors[i], label=model_name)
	fill_between(conts, 
		vec(tpr_means[i].+1.96*tpr_stds[i]), 
		vec(tpr_means[i].-1.96*tpr_stds[i]), 
		color=colors[i], alpha=0.2)
end
xlabel("contamination")
ylabel("TPR@$(fpr)")
ylim([0,1])

subplot(223)
plot(conts, abs.(pauc_welch), label="AUC@$(fpr)")
plot(conts, abs.(tpr_welch), label="TPR@$(fpr)")
plot(conts, abs.(pauc_critvals), "--", label="AUC@$(fpr) critval")
plot(conts, abs.(tpr_critvals), "--", label="AUC@$(fpr) critval")
xlabel("contamination")
ylabel("Welch statistic")
legend(frameon=false)

tight_layout(rect=[0, 0.03, 1, 0.95])
fname = joinpath(outpath, dataset_fname*"_"*model_fname*".png")
savefig(fname)
