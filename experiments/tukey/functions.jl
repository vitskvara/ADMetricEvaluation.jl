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

tukey_stat(m1, m2, msw, n) = (max(m1,m2)-min(m1,m2))/sqrt(msw/n)

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
			tk_stat = tukey_stat(group_means[m[1]], group_means[m[2]], msw, n)
			if !isnan(tk_stat)
				push!(tk_stats[ms][:vals], tk_stat)
				push!(tk_stats[ms][:fpr], fprs[ifpr])
			end
		end
	end
	return tk_stats
end

function tukey_q(means::Vector, vars::Vector, ns::Vector)
	imax, imin = argmax(means), argmin(means)
	var = ((ns[imax]-1)*vars[imax] + (ns[imin]-1)*vars[imin])/(ns[imax]+ns[imin]-2)
	(means[imax]-means[imin])/sqrt(2*var/(ns[imax]+ns[imin]))
end


function get_tukey_qs(measure)
	tk_qs = Dict(:vals=>[], :fpr=>[])
	for ifpr in 1:length(fprs)
		pop_vals = map(m->measure_vals[m][measure][ifpr,:], model_names_diff)
		group_means = map(mean, pop_vals)
		group_vars = map(var, pop_vals)
		tk_q = tukey_q(group_means, group_vars, repeat([n], length(group_means)))

		if !isnan(tk_q)
			push!(tk_qs[:vals], tk_q)
			push!(tk_qs[:fpr], fprs[ifpr])
		end
	end
	return tk_qs
end

nanmean(x, args...;kwargs...) = mean(x[.!isnan.(x)], args...; kwargs...)
nanvar(x, args...;kwargs...) = var(x[.!isnan.(x)], args...; kwargs...)
remove_appendix!(df, cols, appendix) = 
	map(c->rename!(df, Symbol(string(c)*"_$appendix") => c), cols)

function fpr_levels(df, measure)
	colnames = string.(names(df))
	colnames = filter!(c->occursin(measure, c), colnames)
	fprs = map(c->split(c, "_")[end], colnames)
	parse.(Float64, fprs)/100
end