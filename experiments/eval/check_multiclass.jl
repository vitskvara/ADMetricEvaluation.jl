using ADMetricEvaluation
using DataFrames
using Statistics
using PyPlot
using DataFramesMeta
using LinearAlgebra

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
include("../utils/visualize_fit.jl")
savepath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_multiclass_plots"
#savefigs = false
savefigs = true

#############################
dataset = "breast-tissue"
measure = :vol_at_5
model = "LOF"
subclass_by = "adi-car"
seed = 1

sensitivity_dfs = ADMetricEvaluation.multiclass_sensitivities(data_path, dataset, measure)
stats = ADMetricEvaluation.multiclass_sensitivities_stats(data_path, dataset, measure) 
joined_stats = ADMetricEvaluation.join_multiclass_sensitivities_stats(stats)

subclass_row = filter(x->x[Symbol("max/loss")].== subclass_by, sensitivity_dfs[Symbol(model)])
subclass_on = string(argmax(subclass_row[2:end]))
show(subclass_row)

mmeasure = Symbol(string(measure)*"_mean")
printcols = [:dataset, :model, :params, :auc_mean, :auc_weighted_mean, :vol_at_5_mean]
if !(mmeasure in printcols)
	push!(printcols, mmeasure)
end
bydf = @linq ADMetricEvaluation.get_agregdf(data_path, dataset, subclass_by) |> 
	where(:model .== model)
bydf = bydf[.!isnan.(bydf[mmeasure]),:]
show(bydf[printcols])
ondf = @linq ADMetricEvaluation.get_agregdf(data_path, dataset, subclass_on) |> 
	where(:model .== model)
ondf = ondf[.!isnan.(ondf[mmeasure]),:]
show(ondf[printcols])
# show the actual selected params in both dataframes
bymaxind = argmax(bydf[mmeasure])
paramsby = bydf[:params][bymaxind]
println("\nParams selected by:")
show(DataFrame(bydf[bymaxind, :][printcols]))

onmaxind = argmax(ondf[mmeasure])
paramson = ondf[:params][onmaxind]
println("\nSelected on and best params:")
show(append!(ondf[ondf[:params] .== paramsby,:],ondf[onmaxind,:])[printcols])

# extract params
function extract_params(param_str, model)
	if model=="kNN"
		spars = string.(split(param_str, " "))
		spars = map(x->string(split(x, "=")[2]), spars)
		return [Int(Meta.parse(spars[2])), Meta.parse(spars[1])]
	elseif model=="LOF"
		spar = string(split(param_str, "=")[2])
		return [Int(Meta.parse(spar))]
	elseif model=="OCSVM"
		spar = string(split(param_str, "=")[2])
		return [Float64(Meta.parse(spar))]
	elseif model=="IF"
		spar = string(split(param_str, "=")[2])
		return [Int(Meta.parse(spar))]
	end
end

# plot and save the best by fit
basefigname1 = joinpath(savepath,"$(dataset)-$(subclass_by)_$(model)_$(string(measure))_best.png")
visualize_fit(dataset, model, extract_params(paramsby, model), subclass_by, 0, 0.05, seed;
	figname = (savefigs ? basefigname1 : ""))

# plot and save the selected on fit
basefigname2 = joinpath(savepath,"$(dataset)-$(subclass_on)_$(model)_$(string(measure))_selected.png")
visualize_fit(dataset, model, extract_params(paramsby, model), subclass_on, 0, 0.05, seed;
	figname = (savefigs ? basefigname2 : ""))

# plot and save the best by fit
basefigname3 = joinpath(savepath, "$(dataset)-$(subclass_on)_$(model)_$(string(measure))_best.png")
visualize_fit(dataset, model, extract_params(paramson, model), subclass_on, 0, 0.05, seed;
	figname = (savefigs ?  basefigname3 : ""))
show()

if savefigs
	outfile = joinpath(savepath, "$(dataset)-$(subclass_by)-$(subclass_on)_$(model)_$(string(measure)).png")
	cmd = `convert $basefigname1 $basefigname2 $basefigname3 -append $outfile`
	run(cmd)
	run(`rm $basefigname1`)
	run(`rm $basefigname2`)
	run(`rm $basefigname3`)
end