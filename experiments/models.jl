using ScikitLearn
using kNN

@sk_import svm : OneClassSVM
@sk_import ensemble : IsolationForest
@sk_import neighbors : LocalOutlierFactor

OCSVM_model(γ="auto") = OneClassSVM(gamma = γ)
IF_model(n_estimators = 100) = IsolationForest(n_estimators = n_estimators, contamination = "auto", behaviour = "new")
LOF_model(n_neighbors = 20) =  LocalOutlierFactor(n_neighbors = n_neighbors, novelty = true, contamination = "auto")

mutable struct kNN_model
	k::Int
	m::Symbol
	t::Symbol
	knn::kNN.KNNAnomaly
end

kNN_model(k::Int, metric::Symbol) = 
	kNN_model(k, metric, :KDTree, kNN.KNNAnomaly(Array{Float32,2}(undef,1,0), metric, :KDTree))
# create a sklearn-like fit function
ScikitLearn.fit!(knnm::kNN_model, X) = (knnm.knn = kNN.KNNAnomaly(Array(transpose(X)), knnm.m, knnm.t)) 
ScikitLearn.decision_function(knnm::kNN_model, X) = -kNN.predict(knnm.knn, Array(transpose(X)), knnm.k)
