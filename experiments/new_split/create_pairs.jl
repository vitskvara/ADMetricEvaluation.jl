using UCI, StatsBase, DelimitedFiles

datasets = ["letter-recognition", "page-blocks", "pendigits", "statlog-satimage", "wall-following-robot", "waveform-1", 
    "waveform-2"]

allcouples = []
for dataset in datasets
	data = UCI.get_data(dataset)

	# get the list of subdatasets
	subdatasets = unique(vcat(data[2], data[3]))
	# hsuffle them
	subdatasets = sample(subdatasets, length(subdatasets),replace=false)
	# now create the couples
	couples = []
	for i in 1:floor(Int,length(subdatasets)/2)
		push!(couples, vcat([dataset], subdatasets[(2*i-1):(2*i)]))
	end
	push!(allcouples, couples)
end
allcouples = vcat(allcouples...)

f = "data_list.txt"
writedlm(f, allcouples)
