using ADMetricEvaluation
using PyPlot

host = gethostname()
if host == "vit"
	data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/synthetic_data"
	dataset_info = ""
	outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/synthetic_plots"
elseif host == "axolotl.utia.cas.cz"
	data_path = "/home/skvara/work/anomaly_detection/data/metric_evaluation/synthetic_data"
	outpath = "/home/skvara/work/anomaly_detection/data/metric_evaluation/synthetic_plots"
	dataset_info = ""
end

datasets = readdir(data_path)
mkpath(outpath)
models = ["kNN", "LOF", "OCSVM", "IF"]

for dataset in datasets
	_subdatasets = unique(map(x->vcat(split(x,"_")[1:end-1]...)[1], 
		readdir(joinpath(data_path,dataset))))
	for (n, subdataset) in enumerate(_subdatasets)
		try
			f = ADMetricEvaluation.single_dataset_corr_grid(dataset, n, data_path;
				dataset_info = dataset_info, models = models)
			savefig(joinpath(outpath, "$subdataset.png"))
			close()
			println("$subdataset saved!")
		catch
			println("$subdataset not saved!")
		end
	end
end