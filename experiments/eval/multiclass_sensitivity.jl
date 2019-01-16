using DataFrames
using ADMetricEvaluation
using Statistics

#data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data_contaminated"

function sensitivity_df(dataset, measure)
	# dataset = "pendigits"
	dfs = ADMetricEvaluation.loaddata(dataset, data_path)
	aggregdfs = []
	for df in dfs
		_df = ADMetricEvaluation.average_over_folds(df)
		ADMetricEvaluation.drop_cols!(_df)
		push!(aggregdfs, _df)
	end
	aggregdf = vcat(aggregdfs...)
	map(x->rename!(aggregdf, x=>Symbol(split(string(x),"_mean")[1])), names(aggregdf)[4:end])
	# now create the subclass df
	insertcols!(aggregdf, 2, :subclass=>map(x->split(x, dataset)[2][2:end], aggregdf[:dataset]))
	subclasses = unique(aggregdf[:subclass])
	sensdf = DataFrame(Symbol("max/loss")=>String[])
	for subclass in subclasses
		sensdf[Symbol(subclass)] = Float64[]
	end
	models = unique(aggregdf[:model])
	select_hyperparams(df, subclass, measure, models) = map(x->argmax(df[measure][df[:model].==x]), models)
	for rowclass in subclasses
		global rowvec = Array{Any,1}()
		push!(rowvec, string(rowclass))
		for columnclass in subclasses
			# we select hyperparams on one subclass
			rowsubdf = aggregdf[aggregdf[:subclass].==rowclass,:]
			hyperparams_inds = select_hyperparams(rowsubdf, rowclass, measure, models)
			columnsubdf = aggregdf[aggregdf[:subclass].==columnclass,:]
			# and evaluate the difference between the measure on row and column df
			rowmean = Statistics.mean(map(x->rowsubdf[measure][rowsubdf[:model].==x[1]][x[2]] ,zip(models,hyperparams_inds)))
			columnmean = Statistics.mean(map(x->columnsubdf[measure][columnsubdf[:model].==x[1]][x[2]] ,zip(models,hyperparams_inds)))
			push!(rowvec, abs(rowmean-columnmean)/columnmean*100.0)
		end
		push!(sensdf, rowvec)
	end
	return sensdf
end