using ADMetricEvaluation
using PyPlot

host = gethostname()
if host == "vit"
	data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
	outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_dataset_plots_knn"
elseif host == "axolotl.utia.cas.cz"
	data_path = "/home/skvara/work/anomaly_detection/data/metric_evaluation/umap_data"
	outpath = "/home/skvara/work/anomaly_detection/data/metric_evaluation/umap_dataset_plots_knn"
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
end

datasets = readdir(data_path)
mkpath(outpath)
models = ["kNN"]
for dataset in datasets
	_subdatasets = unique(map(x->vcat(split(x,"_")[1:end-1]...)[1], 
		readdir(joinpath(data_path,dataset))))
	for (n, subdataset) in enumerate(_subdatasets)
		try
			f = ADMetricEvaluation.single_dataset_corr_grid(dataset, n, data_path;
				dataset_info = dataset_info, models=models)
			savefig(joinpath(outpath, "$subdataset.png"))
			close()
			println("$subdataset saved!")
		catch
			println("$subdataset not saved!")
		end
	end
end
