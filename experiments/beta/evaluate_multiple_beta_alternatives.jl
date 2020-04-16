include("test_beta_alternatives_functions.jl")

svpath = "/home/vit/vyzkum/measure_evaluation/beta_alternatives/gauss_auc"
orig_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"
filter_ratio = 0.2

datasets = readdir(orig_path)
results = []
for dataset in datasets
	subsets = get_subsets(dataset)
	for subdataset in subsets
		res = load_and_join(dataset, subdataset, orig_path, svpath)
		(res[1] != nothing) ? push!(results, res) : nothing		 
	end
end
alldf_val = vcat([x[1] for x in results]...)
alldf_tst = vcat([x[2] for x in results]...)

row_measures = [:auc, :auc_weighted, :auc_at_1, :tpr_at_1, :bauc_at_1, :lauc_at_1, :measure_at_1,
		:auc_at_5, :tpr_at_5, :bauc_at_5, :lauc_at_5, :measure_at_5]
column_measures = [:auc_at_1, :auc_at_5, :prec_at_1, :prec_at_5,
		:tpr_at_1, :tpr_at_5]
fprs = [0.01, 0.05]
target_measure = :measure_at_5
fprs100 = map(x->round(Int, x*100), fprs)
measure_loss_df = rel_measure_loss(alldf_val, alldf_tst, row_measures, column_measures, fprs100, 
	target_measure, filter_ratio)

showall(alldf_val[!,[:dataset, :params, :tpr_at_5, :auc_at_5, :bauc_at_5, :measure_at_5]])


measure_loss_df_gauss = deepcopy(measure_loss_df)
measure_loss_df_gauss[7,:measure] = :gauss_auc_at_1
measure_loss_df_gauss[12,:measure] = :gauss_auc_at_5

measure_loss_df_gauss

# find the best/worst cases in the new measure
target_measure = :measure_at_5
datasets = unique(alldf_val[!,:dataset])
filtered_datasets = 
	filter(x->sum(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,target_measure])) > length(filter(r->r[:dataset]==x, alldf_val)[!,target_measure])*(1-filter_ratio), datasets)

if length(filtered_datasets) == 0
	nothing
else
	alldf_val_filtered = filter(r->!(r[:dataset] in filtered_datasets), alldf_val)
	alldf_tst_filtered = filter(r->!(r[:dataset] in filtered_datasets), alldf_tst)
end

measure_dict_val_filtered = Dict(zip(row_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val_filtered,alldf_tst_filtered,x,column_measures),row_measures)))

# here we construct the df that will show the best and worst results
row_measure = :measure_at_5
col_measure = :tpr_at_5

# this computes the mean loss by dataset
x1f = measure_dict_val_filtered[row_measure][!,col_measure]
x2f = measure_dict_val_filtered[col_measure][!,col_measure]
dxf = x2f-x1f
newdf = deepcopy(measure_dict_val_filtered[row_measure])
newdf[!,:dx] = dxf
sort(newdf, cols=:dx)

filter(r->r[:dataset]=="ecoli-cp-imL", alldf_val_filtered)[!,[:dataset, :params, :tpr_at_5, :auc_at_5, :bauc_at_5, :measure_at_5]]