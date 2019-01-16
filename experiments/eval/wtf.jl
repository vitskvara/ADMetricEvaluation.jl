using ScikitLearn
using UCI
using ADMetricEvaluation
include("../models.jl")

raw_data = UCI.get_umap_data("page-blocks")
multiclass_data = UCI.create_multiclass(raw_data...)
X_train, y_train, X_test, y_test = UCI.split_data(multiclass_data[2][1], 0.8, 0.05; seed = 1, standardize=true)
m = kNN_model(7, :gamma)
ScikitLearn.fit!(m, Array(transpose(X_train)))
score_fun(X) = -ScikitLearn.decision_function(m, Array(transpose(X))) 
scores = score_fun(X_test)
precision_at_p(score_fun, X_test, y_test, 0.05)
prec = ADMetricEvaluation.precision_at_p(score_fun, X_test, y_test, 0.05)
# how is precision computed
y=y_test
X=X_test
p=0.05
N_a = sum(y)
N_n = length(y) - N_a
N = size(X,2)
@assert N == length(y)
k = Int(floor(N*p/(1-p)))
	