import ADMetricEvaluation
ADME = ADMetricEvaluation
using Plots
plotly()

data_path = "/home/vit/vyzkum/anomaly_detection/data/UCI/umap"
dataset = "wall-following-robot"
raw_data = ADME.get_umap_data(dataset, data_path)
multiclass_data = ADME.create_multiclass(raw_data...)

for (i, (data, label)) in enumerate(multiclass_data)
	println("$dataset-$label")
#	plot(title="$dataset-$label")
#	scatter!(data.normal[1,:], data.normal[2,:], label="normal")
#	scatter!(data.medium[1,:], data.medium[2,:], label="medium")
#	gui()
end

data = multiclass_data[1][1]
