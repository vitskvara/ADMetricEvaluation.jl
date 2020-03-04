include("../t-test/plot_functions.jl")
include("functions.jl")

contamination = 0.01
outpath = joinpath("/home/vit/vyzkum/measure_evaluation/tukey", 
	string(contamination))
mkpath(outpath)

base_dataset = "statlog-satimage"
sub = "1-3"
model_names = [:OCSVM, :kNN, :IF]
model_params = [[0.5], [21, :kappa], [50]]

if model_names[1] != model_names[2]
	model_names_diff = copy(model_names)
else
	model_names_diff = map(x->Symbol(string(x[2])*"$(x[1])"), enumerate(model_names))
end

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

# get the scores for models
ks = collect(1:50)
fprs = collect(range(0.00, 0.99, length=100))
fname = joinpath(outpath, dataset_fname*"_"*model_fname*".jld2")
measure_vals = compute_measures(models, model_names_diff, fprs, ks, fname, contamination)
save(fname, "measure_vals", measure_vals)

# compute welch score
n = ks[end]
α = 0.05
mpairs = pairs(model_names)
mpair_symbols = map(m->Symbol(string(m[1])*"-"*string(m[2])), mpairs)
pauc_welch = map(x->get_statistics(measure_vals,:auc_at_p,n,α,x)[3], mpairs)
tpr_welch = map(x->get_statistics(measure_vals,:tpr_at_p,n,α,x)[3], mpairs)

figure()
subplot(211)
title("AUC@FPR")
for (pauc,m) in zip(pauc_welch, mpairs)
	plot(fprs, abs.(pauc), label = string(m[1])*"-"*string(m[2]))
end
legend()
subplot(212)
title("TPR@FPR")
for (tpr,m) in zip(tpr_welch, mpairs)
	plot(fprs, abs.(tpr), label = string(m[1])*"-"*string(m[2]))
end
legend()

# get all the samples
fpr = 0.05
ifpr = findfirst(fprs .== fpr)
pauc_population = Dict(zip(model_names, 
	map(m->measure_vals[m][:auc_at_p][ifpr,:], model_names_diff)))
tpr_population = Dict(zip(model_names, 
	map(m->measure_vals[m][:tpr_at_p][ifpr,:], model_names_diff)))

figure()
subplot(211)
title("AUC@$fpr")
for m in model_names
	hist(pauc_population[m], label=string(m), density=true, alpha=0.3)
end
subplot(212)
title("TPR@$fpr")
for m in model_names
	hist(tpr_population[m], label=string(m), density=true, alpha=0.3)
end
legend()

# now compute the tukey statistic
population = tpr_population
pop_vals = [population[k] for k in keys(population)]
total_mean = mean(vcat(pop_vals...))
total_var = var(vcat(pop_vals...))
group_means = map(mean, pop_vals)
group_vars = map(var, pop_vals)
mean_var = mean(group_vars)
within_var = sum((group_means .- total_mean).^2*n)/(length(vcat(pop_vals...))-length(group_means))

msw = (n-1)*sum(group_vars)

pauc_ts = get_tukey_stats(:auc_at_p)
tpr_ts = get_tukey_stats(:tpr_at_p)

figure()
suptitle("Tukey statistic")
subplot(211)
title("AUC@FPR")
for m in mpair_symbols
	plot(pauc_ts[m][:fpr], pauc_ts[m][:vals], label=string(m))
end
xlabel("FPR")
xlim([0,1])
plot(pauc_ts[mpair_symbols[1]][:fpr], mean(hcat([pauc_ts[k][:vals] for k in keys(pauc_ts)]...), dims=2),
	label = "mean")
legend()
subplot(212)
title("TPR@FPR")
for m in mpair_symbols
	plot(tpr_ts[m][:fpr], tpr_ts[m][:vals], label=string(m))
end
xlabel("FPR")
xlim([0,1])
plot(tpr_ts[mpair_symbols[1]][:fpr], mean(hcat([tpr_ts[k][:vals] for k in keys(tpr_ts)]...), dims=2),
	label = "mean")
legend()
tight_layout(rect=[0, 0.03, 1, 0.95])

# also compute tukey q stat	
tukey_q(group_means, group_vars, repeat([n], length(group_means)))

pauc_tq = get_tukey_qs(:auc_at_p)
tpr_tq = get_tukey_qs(:tpr_at_p)

figure()
title("Tukey q stat")
plot(pauc_tq[:fpr], pauc_tq[:vals], label="AUC@FPR")
plot(tpr_tq[:fpr], tpr_tq[:vals], label="TPR@FPR")
xlabel("FPR")
xlim([0,1])
legend()

# now plot everything over and over
figure()
suptitle("Comparison of discriminability criterions")
subplot(211)
title("AUC@FPR")
plot(fprs, mean(abs.(hcat(pauc_welch...)), dims=2), label="Mean Welch statistic")
plot(pauc_ts[mpair_symbols[1]][:fpr], 
	mean(hcat([pauc_ts[k][:vals] for k in keys(pauc_ts)]...), dims=2), 
	label = "Mean Tukey statistic")
plot(pauc_tq[:fpr], pauc_tq[:vals], label="Tukey q statistic")
xlabel("FPR")
xlim([0,1])
legend(frameon=false)
subplot(212)
title("TPR@FPR")
plot(fprs, mean(abs.(hcat(tpr_welch...)), dims=2), label="Mean Welch statistic")
plot(tpr_ts[mpair_symbols[1]][:fpr], 
	mean(hcat([tpr_ts[k][:vals] for k in keys(tpr_ts)]...), dims=2), 
	label = "Mean Tukey statistic")
plot(tpr_tq[:fpr], tpr_tq[:vals], label="Tukey q statistic")
xlabel("FPR")
xlim([0,1])
legend(frameon=false)
tight_layout(rect=[0, 0.03, 1, 0.95])
