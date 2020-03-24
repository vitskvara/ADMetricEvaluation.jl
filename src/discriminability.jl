function pairs(v::Vector)
	out = []
	for x in v
		for y in v
			if x!=y && !((y,x) in out)
				push!(out, (x,y))
			end
		end
	end
	out
end

welch_test_statistic(μ1::Real, μ2::Real, σ1::Real, σ2::Real, n1::Int, n2::Int) = 
	(μ1 - μ2)/sqrt(σ1^2/n1 + σ2^2/n2)
function welch_df(σ1::Real, σ2::Real, n1::Int, n2::Int)
	df = (σ1^2/n1 + σ2^2/n2)^2/(σ1^4/(n1^2*(n1-1)) + σ2^4/(n2^2*(n2-1)))
	isnan(df) ? NaN : floor(Int, df)
end
crit_t(α::Real, df::Int) = quantile(TDist(df), 1-α)
crit_t(α::Real, df::Real) = isnan(df) ? NaN : quantile(TDist(df), 1-α)
welch_critval(α::Real, df::Real) = crit_t(α/2, df)
welch_critval(α::Real, σ1::Real, σ2::Real, n1::Int, n2::Int) = 
	welch_critval(α, welch_df(σ1, σ2, n1, n2))
welch_pval(t::Real, df::Int) = 1-cdf(TDist(df), t) # onesided version
welch_pval(t::Real, df::Real) = isnan(df) ? NaN : 1-cdf(TDist(df), t) # onesided version

tukey_test_statistic(μ1::Real, μ2::Real, msw::Real, n::Int) = (max(μ1,μ2)-min(μ1,μ2))/sqrt(msw/n)
# k = number of groups, df = (N - k) = (total samples - k)
crit_srd(α::Real, k::Real, df::Real) = (isnan(k) | isnan(df)) ? NaN : quantile(StudentizedRange(df, k), 1-α)
tukey_critval(α::Real, k::Real, df::Real) = crit_srd(α/2, k, df)

# R = mean ranks, n = number of datasets, k = number of models
friedman_test_statistic(R::Vector,n::Int,k::Int) = 12*n/(k*(k+1))*(sum(R.^2) - k*(k+1)^2/4)
# k = number of models
crit_chisq(α::Real, df::Int) = quantile(Chisq(df), 1-α)
friedman_critval(α::Real, k::Int) = crit_chisq(α/2, k-1)

# adjusted friedman
function adjusted_friedman_test_statistic(R::Vector, n::Int, k::Int)
	F = friedman_test_statistic(R,n,k)
	((n-1)*F)/(n*(k-1)-F)
end
crit_f(α::Real, df1::Int, df2::Int) = quantile(FDist(df1, df2), 1-α)
adjusted_friedman_critval(α::Real, n::Int, k::Int) = crit_f(α/2, k-1, (k-1)*(n-1))

function get_tukey_stats(measure)
	tk_stats = Dict(zip(mpair_symbols, [Dict(:vals=>[], :fpr=>[]) for _ in mpairs]))
	for ifpr in 1:length(fprs)
		population = Dict(zip(model_names, 
			map(m->measure_vals[m][measure][ifpr,:], model_names_diff)))
		pop_vals = [population[k] for k in keys(population)]
		
		group_means = Dict(zip(model_names, map(mean, pop_vals)))
		group_vars = map(var, pop_vals)
		msw = (n-1)*sum(group_vars)
		for (m, ms) in zip(mpairs, mpair_symbols)
			tk_stat = tukey_test_statistic(group_means[m[1]], group_means[m[2]], msw, n)
			if !isnan(tk_stat)
				push!(tk_stats[ms][:vals], tk_stat)
				push!(tk_stats[ms][:fpr], fprs[ifpr])
			end
		end
	end
	return tk_stats
end

function tukey_q_statistic(means::Vector, vars::Vector, ns::Vector)
	imax, imin = argmax(means), argmin(means)
	var = ((ns[imax]-1)*vars[imax] + (ns[imin]-1)*vars[imin])/(ns[imax]+ns[imin]-2)
	(means[imax]-means[imin])/sqrt(var/(ns[imax]+ns[imin]))
end

function nan_tukey_q(means::Vector, vars::Vector, ns::Vector)
	naninds = isnan.(means) .| isnan.(vars)
	tukey_q_statistic(means[.!naninds], vars[.!naninds], ns)
end

function get_tukey_qs(measure)
	tk_qs = Dict(:vals=>[], :fpr=>[])
	for ifpr in 1:length(fprs)
		pop_vals = map(m->measure_vals[m][measure][ifpr,:], model_names_diff)
		group_means = map(mean, pop_vals)
		group_vars = map(var, pop_vals)
		tk_q = nan_tukey_q(group_means, group_vars, repeat([n], length(group_means)))

		if !isnan(tk_q)
			push!(tk_qs[:vals], tk_q)
			push!(tk_qs[:fpr], fprs[ifpr])
		end
	end
	return tk_qs
end

nanmean(x, args...;kwargs...) = mean(x[.!isnan.(x)], args...; kwargs...)
nanvar(x, args...;kwargs...) = var(x[.!isnan.(x)], args...; kwargs...)
function nanrowmean(x)
	z = x[.!vec(mapslices(y->any(isnan.(y)), x, dims=2)),:]
	return (length(z) == 0) ? repeat([NaN], size(x,2)) : mean(z, dims=1)
end
function nanrowmedian(x)
	z = x[.!vec(mapslices(y->any(isnan.(y)), x, dims=2)),:]
	return (length(z) == 0) ? repeat([NaN], size(x,2)) : median(z, dims=1)
end
function nanargmax(x) 
	_x = x[.!isnan.(x)]
	length(_x) > 0 ? findfirst(x .== maximum(_x)) : nothing
end

remove_appendix!(df, cols, appendix) = 
	map(c->rename!(df, Symbol(string(c)*"_$appendix") => c), cols)

function fpr_levels(df, measure)
	colnames = string.(names(df))
	colnames = filter!(c->occursin(measure, c), colnames)
	fprs = map(c->split(c, "_")[end], colnames)
	parse.(Float64, fprs)/100
end

function create_outpath(inpath, dataset)
	outpath = joinpath(replace(inpath, "_pre"=>"_post"), dataset)
	mkpath(outpath)
	outpath
end

function extract_subsets(alldf, dataset)
	subsets = unique(alldf[!,:dataset])
	if length(subsets) == 1 return [""] end
	map(x->x[2][2:end], split.(subsets, dataset))
end

function stat_lines(mean_vals_df, var_vals_df, meas_cols, nexp)
	# tukey q - easy
	tq = map(c->nan_tukey_q(mean_vals_df[!,c], var_vals_df[!,c], 
		repeat([nexp], length(mean_vals_df[!,c]))), meas_cols)

	# parwise tukey and welch statistic
	nr = size(mean_vals_df,1)
	nc = size(mean_vals_df,2) # first three columns are not interesting
	wtm = zeros(Float32, binomial(nr,2), nc) # welch stat matrix
	ttm = zeros(Float32, binomial(nr,2), nc) # tukey stat matrix
	l = 0
	for i in 1:nr-1
		for j in i+1:nr
			l += 1
			for k in 1:nc
				# welch statistic
				wtm[l,k] = abs(welch_test_statistic(mean_vals_df[i,k], mean_vals_df[j,k], 
					var_vals_df[i,k], var_vals_df[j,k], nexp, nexp))
				# tukey statistic
				valid_vars = var_vals_df[.!isnan.(var_vals_df[:,k]),k]
				msw = mean(valid_vars)
				ttm[l,k] = tukey_test_statistic(mean_vals_df[i,k], mean_vals_df[j,k], msw, nexp)
			end
		end
	end
	
	# get the means and medians
	wt_mean = vec(nanrowmean(wtm))
	wt_med = vec(nanrowmedian(wtm))
	tt_mean = vec(nanrowmean(ttm))
	tt_med = vec(nanrowmedian(ttm))

	return tq, wt_mean, wt_med, tt_mean, tt_med
end

function optimal_fprs(df, measure, max_fpr=1.0)
	# determine some constants
	fprs = fpr_levels(df, measure)
	fprs = fprs[fprs.<=max_fpr]
	nfprs = length(fprs) # number of fpr levels of interes
	nexp = maximum(df[!,:iteration]) # number of experiments

	# get the dfs with means and variances
	colnames = names(df)
	meas_cols = filter(x->occursin(measure, string(x)), colnames)[1:nfprs]
	subdf = df[!,vcat([:model, :params, :iteration], meas_cols)]
	
	# now get the actual values
	mean_df = aggregate(subdf, [:model, :params], nanmean);
	remove_appendix!(mean_df, meas_cols, "nanmean");
	var_df = aggregate(subdf, [:model, :params], nanvar);
	remove_appendix!(var_df, meas_cols, "nanvar");
	mean_vals_df = mean_df[!,meas_cols] # these two only contain the value columns
	var_vals_df = var_df[!,meas_cols]

	# get 
	tq, wt_mean, wt_med, tt_mean, tt_med = 
		stat_lines(mean_vals_df, var_vals_df, meas_cols, nexp)
	
	# finally, find the optimal fpr value for each criterion
	tq_maxi, wt_mean_maxi, wt_med_maxi, tt_mean_maxi, tt_med_maxi = 
		map(nanargmax, (tq, wt_mean, wt_med, tt_mean, tt_med))
	return Dict(
			:tukey_q => Dict(:val=>tq[tq_maxi], :fpr=>fprs[tq_maxi]),
			:tukey_mean => Dict(:val=>tt_mean[tt_mean_maxi], :fpr=>fprs[tt_mean_maxi]),
			:tukey_median => Dict(:val=>tt_med[tt_med_maxi], :fpr=>fprs[tt_med_maxi]),
			:welch_mean => Dict(:val=>wt_mean[wt_mean_maxi], :fpr=>fprs[wt_mean_maxi]),
			:welch_median => Dict(:val=>wt_med[wt_med_maxi], :fpr=>fprs[wt_med_maxi])
		),
		fprs
end

function crit_vals(α, var_vals_df, meas_cols, nexp)
	# get sizes
	nr = size(var_vals_df,1)
	nc = size(var_vals_df,2) # first three columns are not interesting
	
	# tukey q and statistic critval - easy, its the same for all columns
	tqc = tukey_critval(α, nr, (nexp-1)*nr)

	# parwise tukey and welch statistic
	wcm = zeros(Float32, binomial(nr,2), nc) # welch crit_val matrix
	l = 0
	for i in 1:nr-1
		for j in i+1:nr
			l += 1
			for k in 1:nc
				# welch statistic
				wcm[l,k] = welch_critval(α, var_vals_df[i,k], var_vals_df[j,k], nexp, nexp)
			end
		end
	end
	
	return tqc, wcm
end

function optimal_fprs_critvals(df, measure, α)
	# determine some constants
	fprs = fpr_levels(df, measure)
	nexp = maximum(df[!,:iteration]) # number of experiments

	# get the dfs with means and variances
	colnames = names(df)
	meas_cols = filter(x->occursin(measure, string(x)), colnames)
	subdf = df[!,vcat([:model, :params, :iteration], meas_cols)]
	
	# now get the actual values
	mean_df = aggregate(subdf, [:model, :params], nanmean);
	remove_appendix!(mean_df, meas_cols, "nanmean");
	var_df = aggregate(subdf, [:model, :params], nanvar);
	remove_appendix!(var_df, meas_cols, "nanvar");
	mean_vals_df = mean_df[!,meas_cols] # these two only contain the value columns
	var_vals_df = var_df[!,meas_cols]

	# get sizes
	nr = size(var_vals_df,1)
	nc = size(var_vals_df,2) # first three columns are not interesting
	
	# get the criterion values
	tq, wt_mean, wt_med, tt_mean, tt_med = 
		stat_lines(mean_vals_df, var_vals_df, meas_cols, nexp)

	# get the critvals
	tqc, wcm = crit_vals(α, var_vals_df, meas_cols, nexp)

	# now get the indices
	tqi = findfirst(tqc .< tq)
	ttmeani = findfirst(tqc .< tt_mean)
	ttmedi = findfirst(tqc .< tt_med) 

	wcrit_vec = vec(nanrowmean(wcm))
	wtmeani = findfirst(wcrit_vec .< wt_mean)
	wtmedi = findfirst(wcrit_vec .< wt_med)
	
	tqi = tqi == nothing ? 1 : tqi
	ttmeani = ttmeani == nothing ? 1 : ttmeani
	ttmedi = ttmedi == nothing ? 1 : ttmedi
	wtmeani = wtmeani == nothing ? 1 : wtmeani
	wtmedi = wtmedi == nothing ? 1 : wtmedi

	return Dict(
			:tukey_q => Dict(:val=>tq[tqi], :fpr=>fprs[tqi]),
			:tukey_mean => Dict(:val=>tt_mean[ttmeani], :fpr=>fprs[ttmeani]),
			:tukey_median => Dict(:val=>tt_med[ttmedi], :fpr=>fprs[ttmedi]),
			:welch_mean => Dict(:val=>wt_mean[wtmeani], :fpr=>fprs[wtmeani]),
			:welch_median => Dict(:val=>wt_med[wtmedi], :fpr=>fprs[wtmedi])
		),
		fprs
end

# this should add columns to the new df
function add_cols(df, opt_fprs, measures)
	# initialize the new df
	colnames = names(df)
	newdf = df[!, filter(x->!any(occursin.(measures, string(x))), colnames)]
	for m in [:auc_at_1, :auc_at_5, :tpr_at_1, :tpr_at_5]
		newdf[!,m] = df[!,m]
	end

	# now add the optimized columns
	for (measure, opt_fpr) in zip(measures, opt_fprs)
		for k in keys(opt_fpr)
			fpr = opt_fpr[k][:fpr]
			fpr_int = round(Int,fpr*100)
			ocolname = Symbol(measure*"_"*string(fpr_int))
			ncolname = Symbol(string(k)*"_$(measure)")
			newdf[!,ncolname] = df[!,ocolname] 
			ncolname = Symbol(string(k)*"_$(measure)_fpr")
			newdf[!,ncolname] = repeat([fpr], size(newdf,1))
		end
	end	
	return newdf
end

function collect_dfs(inpath)
	infiles = readdir(inpath)
	dfs = map(x->CSV.read(joinpath(inpath, x)), infiles)
	alldf = vcat(map(df->drop_cols!(merge_param_cols!(copy(df))), dfs)...);
	alldf, dfs, infiles
end

function process_discriminability_data(inpath, dataset, max_fpr, use_critvals=false, α=0.05)
	outpath = create_outpath(inpath, dataset)
	inpath = joinpath(inpath, dataset)
	# get individual datasets
	alldf, dfs, infiles = collect_dfs(inpath)
	subsets = unique(alldf[!,:dataset])
	measures = ["auc_at", "tpr_at"]

	for subset in subsets
		# get the df of interest
		subsetdf = filter(r->r[:dataset]==subset, alldf)
		# compute the optimal fpr level according to different measure and criterions
		if use_critvals
			opt_fprs = map(x->optimal_fprs_critvals(subsetdf, x, α)[1], measures)
		else
			opt_fprs = map(x->optimal_fprs(subsetdf, x, max_fpr)[1], measures)
		end

		# create new dfs with added columns - measures at the selected fpr
		newdfs = map(x->add_cols(x, opt_fprs, measures), 
			filter(x->occursin(subset, x[1,:dataset]), dfs))
		outfiles = joinpath.(outpath, filter(x->occursin(subset,x),infiles))
		# hopefully the two arrays are alwas sorted the same way
		map(x->CSV.write(x[1], x[2]), zip(outfiles, newdfs))
	end
end
