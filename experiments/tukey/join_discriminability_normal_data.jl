using CSV, DataFrames
indiscpath = ARGS[1]
#indiscpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_post"
normalpath = replace(replace(indiscpath, "discriminability"=>"f1"), "_post"=>"")
outpath = replace(indiscpath, "post"=>"joined")

function joindfs(dataset, f)
	indiscdf = CSV.read(joinpath(indiscpath, dataset, f))
	normaldf = CSV.read(joinpath(normalpath, dataset, f))
	
	# remove some rows and columns
	iters = unique(normaldf[!,:iteration])
	indiscdf = filter(r->(r[:iteration] in iters), indiscdf)
	colnames = names(indiscdf)
	filter!(x->!(x in [:auc, :auc_weighted, :prec_at_1, :prec_at_5,
		:f1_at_1, :f1_at_5,  :auc_at_1, :auc_at_5, :tpr_at_1, :tpr_at_5]), 
		colnames)
	indiscdf = indiscdf[!,colnames]

	#
	filter!(x->!(occursin("auc", string(x)) | occursin("tpr", string(x))), colnames)
	outdf = join(normaldf, indiscdf, on = colnames)
	CSV.write(joinpath(outpath, dataset, f), outdf)
end

datasets = readdir(indiscpath)
for dataset in datasets
	mkpath(joinpath(outpath, dataset))
	for f in readdir(joinpath(indiscpath, dataset))
		try
			joindfs(dataset, f)
		catch e
			if is(e, ArgumentError)
				nothing
			else
				rethrow(e)
			end
		end
	end
end