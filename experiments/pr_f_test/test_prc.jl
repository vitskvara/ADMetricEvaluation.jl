using EvalCurves
using PyPlot

as = vcat([2.5, -0.1],randn(8))
Y = vcat([1,1], fill(0,8))

#rec, pr = EvalCurves.prcurve(as,Y; zero_rec=true) 
rec, pr = EvalCurves.prcurve(as,Y) 

figure()
plot(rec, pr)

length(unique(rec))


# how is it actually? the number off different recall values is not larger than the number of positive samples
as = vcat([2.5, -0.1, 0, 1],randn(8))
Y = vcat([1,1,1,1], fill(0,8))

#rec, pr = EvalCurves.prcurve(as,Y; zero_rec=true) 
rec, pr = EvalCurves.prcurve(as,Y) 

figure()
plot(rec, pr)

length(unique(rec))