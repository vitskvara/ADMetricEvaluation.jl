include("plot_dataset.jl")
data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
datasets = readdir(data_path)
outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_dataset_plots"
mkpath(outpath)

for dataset in datasets
	_subdatasets = unique(map(x->vcat(split(x,"_")[1:end-1]...)[1], 
		readdir(joinpath(data_path,dataset))))
	for (n, subdataset) in enumerate(_subdatasets)
		try
			f = correlation_grid_plot(dataset, n, data_path)
			savefig(joinpath(outpath, "$subdataset.png"))
			close()
			println("$subdataset saved!")
		catch
			println("$subdataset not saved!")
		end
	end
end