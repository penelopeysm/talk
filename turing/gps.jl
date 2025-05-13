using AbstractGPs, LogExpFunctions, Turing

N = 10
xs = ColVecs(rand(2, N))
ys = mean.(xs) .< rand(length(xs))

@model function gp(points)
    var ~ Exponential(1)
    scale ~ Exponential(1)
    f = GP(var * with_lengthscale(SEKernel(), scale))
    preds ~ f(points, 1e-8)
    y ~ product_distribution(BernoulliLogit.(preds))
end

model = dense_gp(xs) | (; y = ys)

sample(model, NUTS(), 1000)

sample(model, Gibbs(:preds => ESS(),
             (:var, :scale) => externalsampler(HitAndRun(SliceSteppingOut(4.0)))),
       1000)
