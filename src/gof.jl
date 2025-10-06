"""
    aic(fit) -> Real

Akaike Information Criterion for a least-squares fit.

# Definition
`AIC = 2k + N*log(σ²)`, where:
- `k = length(coef(fit))` is the number of fitted parameters,
- `N = nobs(fit)` is the number of observations,
- `σ² = rss(fit)/N` is the residual variance estimate.

# Returns
- Lower is better (relative comparison across models on the same dataset).
"""
function aic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return 2k + N*log(σ2)
end

"""
    aicc(fit) -> Real

Small-sample corrected AIC.

# Definition
`AICc = AIC + (2k(k+1)) / (N - k - 1)`.

# Returns
- Recommended when `N / k` is modest.
"""
function aicc(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    a = aic(fit)
    return a + (2k*(k+1)) / (N - k - 1)
end

"""
    bic(fit) -> Real

Bayesian Information Criterion for a least-squares fit.

# Definition
`BIC = k*log(N) + N*log(σ²)`, with
- `k = length(coef(fit))`,
- `N = nobs(fit)`,
- `σ² = rss(fit)/N`.

# Returns
- Lower is better; BIC penalizes model complexity more strongly than AIC.
"""
function bic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return k*log(N) + N*log(σ2)
end

"""
    bicc(fit) -> Real

Bias-corrected BIC variant.

# Definition
`BICc = N*log(σ²) + N*k*log(N)/(N - k - 2)`.

# Returns
- A more conservative penalty when `N` is not ≫ `k`.
"""
function bicc(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return N * log(σ2) + N * k * log(N) / (N - k - 2)
end

