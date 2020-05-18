using UCI, DelimitedFiles

dataset = "letter-recognition"
normal_class = "T"
anomalous_class = "Z"

function get_new_split_data(dataset, normal_class, anomalous_class)
	normal_raw = UCI.get_data(dataset, normal_class)
	anomalous_raw = UCI.get_data(dataset, anomalous_class)
	# we have to check this in case the class is the same as the normal one in the original problem
	normal_X = (normal_raw[2][1] == normal_class) ? normal_raw[1].normal : normal_raw[1].medium
	anomalous_X = (anomalous_raw[2][1] == anomalous_class) ? anomalous_raw[1].normal : anomalous_raw[1].medium
	data = UCI.ADDataset(
		normal_X, 
		Array{Float32,2}(undef,0,0),
		anomalous_X,
		Array{Float32,2}(undef,0,0),
		Array{Float32,2}(undef,0,0)
		)
end

data = get_new_split_data(dataset, normal_class, anomalous_class)

# print the data sizes
data_info = readdlm("data_list.txt")
for i in 1:size(data_info,1)
	dataset = string(data_info[i,1])
	normal_class = string(data_info[i,2])
	anomalous_class = string(data_info[i,3])
	data = get_new_split_data(dataset, normal_class, anomalous_class)	
	println(data_info[i,:])
	println(size(data.normal))
	println(size(data.medium))
end




# this is how the data is split now
p = 0.6
contamination = 0.0
standardize = true
n_experiments = 10
test_contamination = nothing
data_path = ""

subdataset = "U-Z"
raw_data = UCI.get_data(dataset, path=data_path)
multiclass_data = UCI.create_multiclass(raw_data...)
data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]
X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, p, contamination;
	test_contamination = test_contamination, seed = iexp, standardize=standardize)
X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);



