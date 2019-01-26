using ADMetricEvaluation
using PyPlot
using DataFramesMeta
using Statistics
using CSV
using IterTools

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
all_datasets = readdir(data_path)
multiclass_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_data"
datasets = readdir(multiclass_path)	
push!(datasets, "isolet")
measure = :auc

dataset_chars = CSV.read("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv")
# first create the subclass column
function get_dataset(s, datasets)
	inds = map(x->occursin(x,s), datasets)
	if !any(inds)
		return ""
	else
		return datasets[inds][1]
	end
end
rename!(dataset_chars, :dataset => :subdataset)
insertcols!(dataset_chars, 1, :dataset => map(x->get_dataset(x, all_datasets), dataset_chars[:subdataset]))
function remove_prefix(str, prefix)
	s = replace(str, prefix=>"")
	if s==""
		return s
	end
	if s[1] == '-'
		s=s[2:end]
	end
	return s
end
insertcols!(dataset_chars, 3, :subclass => map(x->remove_prefix(x[1], x[2]), 
	zip(dataset_chars[:subdataset],dataset_chars[:dataset])))

# now create a comprehensive df with combinations of subclasses in rows together with their 
# losses and characteristics (e.g. difference of clusterdnesses)
# subclass x subclass x model

get_sensitivity(by::String, on::String, sensdf::DataFrame) = 
	sensdf[sensdf[Symbol("max/loss")] .== by, :][Symbol(on)][1]

subdfs = []
for dataset in datasets
	subclassdf = @linq dataset_chars |> where(:dataset.==dataset, :iteration.==1)
	subclasses = unique(subclassdf[:subclass])
	p = vec(collect(Iterators.product(subclasses, subclasses)))
	df = DataFrame()
	df[:by_class] = [x[1] for x in p]
	df[:on_class] = [x[2] for x in p]
	insertcols!(df, 1, :dataset => dataset)
	df[:by_clust] = join( df,rename(subclassdf, :subclass=>:by_class), on = :by_class, makeunique = true)[:clusterdness]
	df[:on_clust] = join( df,rename(subclassdf, :subclass=>:on_class), on = :on_class, makeunique = true)[:clusterdness]
	df[:clust_diff] = df[:by_clust] - df[:on_clust]
	# now get the sensitivities
	sens_dfs = ADMetricEvaluation.multiclass_sensitivities(data_path, dataset, measure)
	models = collect(keys(sens_dfs))
	dfs = []
	for model in models
		_df = deepcopy(df)
		insertcols!(_df, 2, :model=>string(model))
		_df[:loss] = map(x->get_sensitivity(x[1], x[2], sens_dfs[model]), 
			zip(_df[:by_class], _df[:on_class]))
		push!(dfs, _df)
	end
	push!(subdfs, vcat(dfs...))
end
allsubdf = vcat(subdfs...)
allsubdf[:rel_clust_diff] = allsubdf[:clust_diff]./allsubdf[:on_clust]
allsubdf[:rel_max_clust_diff] = allsubdf[:clust_diff]./max.(allsubdf[:by_clust], allsubdf[:on_clust])
function scatter_clust_vs_loss(df; kwargs...)
	_df = @linq df |> where(.!(:by_clust.==:on_clust), .!(:model.=="LOF"))
	_df = @linq _df |> where(.!(:loss.==0.0))
	_df = @linq _df |> where(:loss.>=2.0)
	scatter(abs.(_df[:rel_max_clust_diff]), _df[:loss]/100; kwargs...)
	#scatter(log.(_df[:by_clust]), _df[:loss]/100; kwargs...)
end
scatter_clust_vs_loss(allsubdf, s=5, alpha=1)
