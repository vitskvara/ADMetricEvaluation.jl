import ADMetricEvaluation
ADME = ADMetricEvaluation
using Test 

data_path = "/home/vit/vyzkum/anomaly_detection/data/UCI/umap"

dataset = "abalone"
xa = ADME.get_umap_data(dataset, data_path)
ya = ADME.create_multiclass(xa...)

dataset = "cardiotocography"
xc = ADME.get_umap_data(dataset, data_path)
yc = ADME.create_multiclass(xc...)
	
dataset = "yeast"
xy = ADME.get_umap_data(dataset, data_path)
yy = ADME.create_multiclass(xy...)

zaa = ADME.split_data(ya[1][1], 0.8)
zae = ADME.split_data(ya[1][1], 0.8, difficulty = :easy)
zaem = ADME.split_data(ya[1][1], 0.8, difficulty = [:easy, :medium])

@testset "DATA" begin
	@test typeof(xa[1]) == ADME.ADDataset
	@test length(ya) == 1

	@test typeof(xc[1]) == ADME.ADDataset
	@test length(yc) > 1
	@test yc[1][2] == "2-9"

	@test yy[1][2] == "CYT-MIT"

	@test size(zaa[1],2) == size(zaem[1],2) == size(zae[1],2)
	@test size(zaa[2],2) > size(zaem[2],2) > size(zae[2],2)
end

