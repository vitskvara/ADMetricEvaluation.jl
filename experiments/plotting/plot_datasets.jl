using ADMetricEvaluation
using PyPlot

host = gethostname()
#data_type = "umap_data_contaminated"
data_type = "full_data_contaminated"
if host == "vit"
	data_path = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation", data_type)
	dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
	outpath = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation", data_type*"_plots")
elseif host == "axolotl.utia.cas.cz"
	data_path = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation", data_type)
	outpath = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation", data_type*"_plots")
	dataset_info = "/home/skvara/work/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
end

datasets = readdir(data_path)
mkpath(outpath)

for dataset in datasets
	_subdatasets = unique(map(x->vcat(split(x,"_")[1:end-1]...)[1], 
		readdir(joinpath(data_path,dataset))))
	for (n, subdataset) in enumerate(_subdatasets)
		try
			f = ADMetricEvaluation.single_dataset_corr_grid(dataset, n, data_path;
				dataset_info = dataset_info)
			savefig(joinpath(outpath, "$subdataset.png"))
			close()
			println("$subdataset saved!")
		catch
			println("$subdataset not saved!")
		end
	end
end