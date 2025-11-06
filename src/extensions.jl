"""
    fcs_plot(fit, τ, g) -> fig, fit    
    fcs_plot(fit, ch) -> fig, fit
    fcs_plot(spec, τ, data, p0) -> fig, fit
    fcs_plot(spec, ch, p0) -> fig, fit
    fcs_plot(model, τ, data, p0) -> fig, fit
    fcs_plot(model, ch, p0) -> fig, fit
    
Plot the autocorrelation data and its fit by the input FCSModel or FCSModelSpec. 
Optionally, the weighted residuals between the data and fit are included as a second panel 
along the bottom.

# Example
```julia
using CairoMakie, LaTeXStrings, FCSFitting

# Synthetic example parameters and data: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

# Organize data into a channel for easier handling
channel = FCSChannel("sample", t, g, nothing)

fig, fit = fcs_plot(spec, channel, initial_parameters)
save("corr1.png", fig)
```

# Keyword arguments
- `residuals=true`: Include bottom residuals panel if `true`.
- `color1`, `color2`, `color3`: Plot colors for data, fit, and residuals, respectively.
- `kwargs...`: Passed to `fcs_fit`

# Notes
- Delegates to the internal `_fcs_plot` methods and subsequently to `fcs_fit`.
- Uses log-scaled τ axis.
"""
function fcs_plot end

function fcs_plot(fit::FCSFitResult, τ::AbstractVector, data::AbstractVector; 
                  residuals::Bool=true, color1=:deepskyblue3, 
                  color2=:orangered2, color3=:steelblue4, kwargs...)
    if residuals
        _fcs_plot(fit, τ, data, color1, color2, color3; kwargs...)
    else
        _fcs_plot(fit, τ, data, color1, color2; kwargs...)
    end
end

function fcs_plot(fit::FCSFitResult, ch::FCSChannel; kwargs...)
    return fcs_plot(fit, ch.τ, ch.G; kwargs...)
end

function fcs_plot(spec::FCSModelSpec, τ::AbstractVector, data::AbstractVector,
                  p0::AbstractVector; kwargs...)
    fit = fcs_fit(spec, τ, data, p0; kwargs...)
    return fcs_plot(fit, τ, data; kwargs...)
end

fcs_plot(spec::FCSModelSpec, ch::FCSChannel, p0::AbstractVector; kwargs...) = 
    fcs_plot(spec, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)

function fcs_plot(m::FCSModel, τ::AbstractVector, data::AbstractVector, 
                  p0::AbstractVector; kwargs...)
    fit = fcs_fit(m, τ, data, p0; kwargs...)
    return fcs_plot(fit, τ, data; kwargs...)
end

function fcs_plot(m::FCSModel, ch::FCSChannel, p0::AbstractVector; kwargs...)
    fit = fcs_fit(m, ch, p0; kwargs...)
    return fcs_plot(fit, ch; kwargs...)
end


"""
    _fcs_plot(spec, τ, G, θ0; kwargs...) -> fig, fit, scales

Requires `CairoMakie` (and `LaTeXStrings` for math labels).
"""
_fcs_plot(args...; kwargs...) =
    error("`_fcs_plot` requires CairoMakie (and LaTeXStrings). Load them: `using CairoMakie, LaTeXStrings`.")

"""
    resid_acf_plot(resid; kwargs...) -> fig

Requires `CairoMakie`. Activate by `using CairoMakie`.
"""
resid_acf_plot(args...; kwargs...) =
    error("`resid_acf_plot` requires CairoMakie. Load it: `using CairoMakie`.")

"""
    fcs_table(model, fit, scales; kwargs...) -> pretty output

Requires `PrettyTables`. Activate by `using PrettyTables`.
"""
fcs_table(args...; kwargs...) = 
    error("`fcs_table` requires PrettyTables. Load it: `using PrettyTables`.")

"""
    read_fcs(path; kwargs...) -> FCSData

Requires `DelimitedFiles`. Activate by `using DelimitedFiles` in the session.
"""
read_fcs(::Any; kwargs...) = 
    error("`read_fcs` requires DelimitedFiles. Load it first: `using DelimitedFiles`.")