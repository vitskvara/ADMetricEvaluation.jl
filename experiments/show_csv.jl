using CSV

file = ARGS[1]

df = CSV.read(file)
print(df)
