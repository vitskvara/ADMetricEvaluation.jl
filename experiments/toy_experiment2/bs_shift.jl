using FileIO, BSON
using DrWatson
using PyPlot, Statistics

f = "toy_data_0.51.bson"
data = load(f)
@unpack result, ns = data
m = "bstpr"
m1 = Symbol("$(m)1")
m2 = Symbol("$(m)2")

mx1 = [mean(d[m1]) for d in result]
mx2 = [mean(d[m2]) for d in result]
mtpr1 = [mean(d[:tpr1]) for d in result]
mtpr2 = [mean(d[:tpr2]) for d in result]
ns_shifted1 = ns/2
ns_shifted2 = ns*0.63

figure(figsize=(8,8))
plot(log10.(ns), mtpr1, label="tpr")
plot(log10.(ns), mx1, label="bstpr")
plot(log10.(ns_shifted1), mx1, label="bstpr (shift=0.5)")
legend()
savefig("shift=0.5.png")

figure(figsize=(8,8))
plot(log10.(ns), mtpr1, label="tpr")
plot(log10.(ns), mx1, label="bstpr")
plot(log10.(ns_shifted2), mx1, label="bstpr (shift=0.63)")
legend()
savefig("shift=0.63.png")
