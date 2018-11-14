using ADMetricEvaluation
using PyPlot

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
datasets = readdir(data_path)
outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_dataset_plots"
dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
mkpath(outpath)

for dataset in datasets[1:2]
	_subdatasets = unique(map(x->vcat(split(x,"_")[1:end-1]...)[1], 
		readdir(joinpath(data_path,dataset))))
	for (n, subdataset) in enumerate(_subdatasets)
		#try
			f = ADMetricEvaluation.correlation_grid_plot(dataset, n, data_path;
				dataset_info = dataset_info)
			savefig(joinpath(outpath, "$subdataset.png"))
			close()
			println("$subdataset saved!")
		#catch
		#	println("$subdataset not saved!")
		#end
	end
end