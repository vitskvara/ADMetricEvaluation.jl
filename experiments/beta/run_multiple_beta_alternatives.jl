include("test_beta_alternatives_functions.jl")

svpath = "/home/vit/vyzkum/measure_evaluation/beta_alternatives/gauss_auc"
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_beta_contaminated-0.00"

datasets = readdir(orig_path)
fprs = [0.01, 0.05]

gaussian_pdf(nd::Real,p::Real,n::Int) = 1/(sqrt(2*pi)*sqrt(n*p*(1-p)))*exp(-(nd-n*p)^2/(2*n*p*(1-p)))*n
function gauss_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, warns=true)
    n = length(y_true) - sum(y_true) # number of negative samples

    # compute roc
    roc = roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = EvalCurves.linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = gaussian_pdf.(xw.*n, fpr, n)

    wauroc = auc(roci..., w)
end

measuref = gauss_auc
for dataset in datasets
	subsets = get_subsets(dataset)
	for subdataset in subsets
		results = save_measure_test_results(dataset, subdataset, measuref, svpath, fprs, orig_path)
	end
end
