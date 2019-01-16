function average_over_folds(df)
	if df[:model][1] == "IF"
		return aggregate(df, [:dataset, :model, :num_estimators], Statistics.mean)
	elseif df[:model][1] == "LOF"
		return aggregate(df, [:dataset, :model, :num_neighbors], Statistics.mean)
	elseif df[:model][1] == "OCSVM"
		return aggregate(df, [:dataset, :model, :gamma], Statistics.mean)
	elseif df[:model][1] == "kNN"
		return aggregate(df, [:dataset, :model, :metric, :k], Statistics.mean)
	end
end

function drop_cols!(df)
	if df[:model][1] == "IF"
		deletecols!(df, :num_estimators)
	elseif df[:model][1] == "LOF"
		deletecols!(df, :num_neighbors)
	elseif df[:model][1] == "OCSVM"
		deletecols!(df, :gamma)
	elseif df[:model][1] == "kNN"
		deletecols!(df, :metric)
		deletecols!(df, :k)	
	end
end

function pareto_optimal_params(df, metrics)
	X = convert(Array,df[metrics])'
	weight_mask = fill(false, length(metrics))
	weight_mask[findall(x->x == :auc_weighted_mean, metrics)[1]] == true
	pareto_optimal_i = MultiObjective.pareto_best_index(X, weight_mask)
	return df[pareto_optimal_i,:] 
end

function rank_models(data_path::String; 
	metrics = [:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5],
	models = ["kNN", "IF", "LOF", "OCSVM"],
	pareto_optimal = false
	)
	# first get the aggregated collection of all data
	datasets = readdir(data_path)
	res = []
	for dataset in datasets
		dfs = loaddata(dataset, data_path)
		aggregdfs = []
		for df in dfs
			_df = average_over_folds(df)
			if pareto_optimal
				_df = pareto_optimal_params(_df, map(x->Symbol(string(x)*"_mean"), metrics))
			end
			drop_cols!(_df)
			push!(aggregdfs, _df)
		end
		push!(res, vcat(aggregdfs...))
	end
	if !pareto_optimal
		alldf = aggregate(vcat(res...), [:dataset, :model], maximum)
	else
		alldf = vcat(res...)
	end

	datasets = unique(alldf[:dataset])
	rankdf = DataFrame(:dataset=>String[], :metric=>Any[], :kNN=>Float64[], :LOF=>Float64[],
		:IF=>Float64[], :OCSVM=>Float64[])
	for dataset in datasets
		if pareto_optimal
			aggmetrics = map(x->Symbol(string(x)*"_mean"), metrics)
		else
			aggmetrics = map(x->Symbol(string(x)*"_mean_maximum"), metrics)
		end
		for metric in aggmetrics  
			vals = alldf[alldf[:dataset].==dataset, [:model, metric]]
			vals[:rank]=rankvals(vals[metric])
			try
				push!(rankdf, [dataset, metric, 
					vals[:rank][vals[:model].=="kNN"][1],
					vals[:rank][vals[:model].=="LOF"][1],
					vals[:rank][vals[:model].=="IF"][1],
					vals[:rank][vals[:model].=="OCSVM"][1]
					])
			catch
				nothing
			end
		end 
	end
	return rankdf, alldf
end

function rankvals(x::Vector, rev=true)
    j = 1
    tiec = 0 # tie counter
    y = Float64[]
    # create ranks
    sx = sort(x, rev=rev)
    is = sortperm(x, rev=rev)
    for _x in sx
        # this decides ties
        nties = size(x[x.==_x],1) - 1
        if nties > 0
            push!(y, (sum((j-tiec):(j+nties-tiec)))/(nties+1))
            tiec +=1
            # restart tie counter
            if tiec > nties
                tiec = 0
            end
        else
            push!(y,j)
        end
        j+=1
    end
    return y[sortperm(is)]
end


function model_ranks_stats(data_path, metrics=[:auc, :auc_weighted, :auc_at_5, :prec_at_5, :tpr_at_5, :vol_at_5, :auc_at_1, :prec_at_1,
:tpr_at_1, :vol_at_1])
	rankdf, alldf = rank_models(data_path, metrics = metrics)

	Nm = length(metrics)

	ranks_mean = DataFrame(:metric=>String[],:kNN=>Float64[], :LOF=>Float64[], :IF=>Float64[], :OCSVM=>Float64[])
	ranks_sd = DataFrame(:metric=>String[],:kNN=>Float64[], :LOF=>Float64[], :IF=>Float64[], :OCSVM=>Float64[])
	figure(figsize=(10,5))
	global ind = 1
	for metric in (map(x->Symbol(string(x)*"_mean_maximum"), metrics))
		subplot(1,Nm,ind)
		mus = []
		sds = []
		for model in [:kNN, :LOF, :IF, :OCSVM]
			x = rankdf[model][rankdf[:metric].==metric]
			mu=Statistics.mean(x)
			std=Statistics.std(x)
			push!(mus, mu)
			push!(sds, std)
			plt[:hist](x, 20, label=string(model), alpha=1, histtype="step")
		end
		push!(ranks_mean, vcat([string(metric)], mus))
		push!(ranks_sd, vcat([string(metric)], sds))
		legend()
		xlabel(metric)
		global ind+=1
	end
	return ranks_mean, ranks_sd
end