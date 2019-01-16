import ADMetricEvaluation
ADME = ADMetricEvaluation
using PyPlot

data_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_data"
dataset_info = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/dataset_overview.csv"
outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_other_plots"
mkpath(outpath)

model = "OCSVM"
alldf = ADME.load_all_by_model(data_path, model)
param = :gamma
metric = :vol_at_5
uvals = unique(alldf[param])

figure()
for val in uvals
	plt[:hist](alldf[metric][alldf[param].==val],10,label="$val",density = true, histtype = "step", lw=5,
		alpha=0.5)
end
title("$param $metric")
legend()
show()