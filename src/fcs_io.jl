struct FCSChannel
    name::String                 # e.g. "G_DD" or "G_DA"
    τ::Vector{Float64}           # lag times (s)
    G::Vector{Float64}           # correlation values
    σ::Union{Nothing,Vector{Float64}}  # std dev per lag (optional)
end

struct FCSData
    channels::Vector{FCSChannel}
    metadata::Dict{String,Any}   # sample, T, η, NA, λ, pinhole, detector, etc.
    source::String               # filepath or “in-memory”
end

function read_fcs(path::AbstractString; start_idx::Union{Nothing,Int}=nothing,
                  end_idx::Union{Nothing,Int}=nothing,
                  colspec=:auto, metadata=Dict{String,Any}())

    raw = readdlm(path)
    r1 = isnothing(start_idx) ? 1 : start_idx
    r2 = isnothing(end_idx)   ? size(raw,1) : end_idx
    M  = raw[r1:r2, :]

    # Basic inference: 1st col = τ, next 4 = Gs, next 4 = σs (if present)
    if colspec === :auto
        ncol = size(M,2)
        τ = vec(M[:,1])
        chans = FCSChannel[]
        # try pairs (G,σ) for columns 2..n
        i = 2
        k = 1
        while i <= ncol
            G = vec(M[:,i])
            σ = (i+4 <= ncol && ncol >= 9) ? vec(M[:, i+4]) : nothing
            push!(chans, FCSChannel("G[$k]", τ, G, σ))
            i += 1
            k += 1
            if k > 4 && ncol <= 9; break; end
        end
        return FCSData(chans, metadata, String(path))
    else
        # explicit mapping
        names = String[]
        chans = FCSChannel[]
        τ = vec(M[:, first(first(colspec)) == :τ ? last(first(colspec)) : error("τ col missing")])
        # build channels
        for tup in colspec
            sym, idx = tup
            if sym === :τ; continue; end
            if sym === :G
                name = get(tup, 3, "G")
                σidx  = get(tup, 4, nothing)
                σ = isnothing(σidx) ? nothing : vec(M[:, σidx])
                push!(chans, FCSChannel(name, τ, vec(M[:,idx]), σ))
            end
        end
        return FCSData(chans, metadata, String(path))
    end
end


"""
    infer_parameter_list(model_name, params; n_diff=nothing)

Infer the names of parameters used in the fitting based on the model name and
parameter vector length. The returned names follow the same ordering as the
model parameter vectors:
- base parameters
- all dynamic times (τ_dyn)
- all dynamic fractions (K_dyn)
"""
function infer_parameter_list(model_name::Symbol, params::AbstractVector; 
                              n_diff::Union{Nothing,Int}=nothing, 
                              diffusivity::Union{Nothing,Real}=nothing)
    L = length(params)
    column_names = String[]

    if model_name === :fcs_2d
        # base = 3 → τD, g0, offset
        m = _ndyn_from_len(L - 3)
        isnothing(diffusivity) ? 
        append!(column_names, ["Diffusion time τ_D [s]"]) : 
        append!(column_names, ["Diffusion time τ_D [s]", "Beam width w_0 [m]"])

        append!(column_names, [
            "Current amplitude G(0)",
            "Offset G(∞)",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_2d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_2d_mdiff"))
        n = n_diff
        base = 2n + 2  # τDs[1:n], wts[1:n], g0, offset
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i] [s]"      for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]"   for i in 1:n])
        append!(column_names, ["Current amplitude G(0)", "Offset G(∞)"])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_3d
        # base = 4 → τD, g0, offset, κ
        m = _ndyn_from_len(L - 4)
        isnothing(diffusivity) ? 
        append!(column_names, ["Diffusion time τ_D [s]"]) : 
        append!(column_names, ["Diffusion time τ_D [s]", "Beam width w_0 [m]"])

        append!(column_names, [
            "Current amplitude G(0)",
            "Offset G(∞)",
            "Structure factor κ",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_3d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_3d_mdiff"))
        n = n_diff
        base = 2n + 3  # τDs[1:n], wts[1:n], g0, offset, κ
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i] [s]"      for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]"   for i in 1:n])
        append!(column_names, [
            "Current amplitude G(0)",
            "Offset G(∞)",
            "Structure factor κ",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    else
        return String[]
    end

    return column_names
end

"""
    fcs_plot(model::Function, ch, θ0, color1, color2, color3; residuals=true, kwargs...)
"""
function fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector; 
                  residuals::Bool=true, color1=:deepskyblue3, 
                  color2=:orangered2, color3=:steelblue4, kwargs...) 
    if residuals
        _fcs_plot(model, ch, θ0, color1, color2, color3; kwargs...)
    else
        _fcs_plot(model, ch, θ0, color1, color2; kwargs...)
    end
end

"""
    _fcs_plot(model::Function, ch, θ0; fontsize=20, 
              color1=:deepskyblue3, color2=:orangered2, color3=:steelblue4, kwargs...)

Fit FCS data with `model` using `fcs_fit` and generate a plot of the fit AND the residuals. 
"""
function _fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector, color1::Symbol, 
                   color2::Symbol, color3::Symbol; fontsize::Int = 20, kwargs...)
    fit, scales = fcs_fit(model, ch.τ, ch.G, θ0; σ = ch.σ, kwargs...)

    fig = Figure(size=(700, 600), fontsize=fontsize)

    # Top panel for main correlation fitting
    Axis(fig[1,1];
         xticklabelsvisible = false,
         ylabel = L"\mathrm{Correlation} \; G(\tau)",
         ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)],
         xscale = log10, height = 400, width = 600)

    scatter!(ch.τ, ch.G; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)
    lines!(ch.τ, model(ch.τ, fit.param .* scales; diffusivity = get(kwargs, :diffusivity, nothing)); 
           linewidth=3, color=color2, alpha=0.9)

    # Bottom panel for residuals plot
    Axis(fig[2,1];
         xlabel = L"\mathrm{Logarithmic\ lag\ time}\; \log_{10}{\tau}",
         ylabel = L"\mathrm{Residuals}",
         xscale = log10, height = 100, width = 600,
         xtickformat = xs -> [L"%$(log10(xs[i]))" for i in eachindex(xs)],
         ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)])

    scatterlines!(ch.τ, fit.resid; color=color3, markersize=5, strokewidth=1, alpha=0.7)

    return fig, fit, scales
end
"""
    _fcs_plot(model::Function, ch, θ0; fontsize=20, 
              color1=:deepskyblue3, color2=:orangered2, kwargs...)

Fit FCS data with `model` using `fcs_fit` and generate a plot of the fit WITHOUT the residuals. 
"""
function _fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector, color1::Symbol, 
                   color2::Symbol; fontsize::Int = 20, kwargs...)
    fit, scales = fcs_fit(model, ch.τ, ch.G, θ0; σ = ch.σ, kwargs...)

    fig = Figure(size=(700, 600), fontsize=fontsize)

    # Top panel for main correlation fitting
    Axis(fig[1,1];
         xticklabelsvisible = false,
         ylabel = L"\mathrm{Correlation} \; G(\tau)",
         ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)],
         xscale = log10)

    scatter!(ch.τ, ch.G; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)
    lines!(ch.τ, model(ch.τ, fit.param .* scales; diffusivity = get(kwargs, :diffusivity, nothing)); 
           linewidth=3, color=color2, alpha=0.9)

    return fig, fit, scales
end

"""
    fcs_table(model::Function, lag_times, data, θ0; backend::Symbol=:html, kwargs...)

Fit FCS data with `model` using `fcs_fit` and generate a table of the fitted parameters and the goodness of fit, BIC.    
"""
function fcs_table(model::Function, lag_times::AbstractVector, data::AbstractVector, θ0::AbstractVector; 
                   backend::Symbol=:html, kwargs...)
    fit, scales = fcs_fit(model, lag_times, data, θ0; kwargs...)
    fcs_table(model, fit, scales; backend)
end

"""
    fcs_table(fit::LsqFit.LsqFitResult, scales::AbstractVector; backend::Symbol=:html)

Generate a table of the fitted parameters corresponding to an FCS `LsqFitResult`, `fit`, and the goodness of fit, BIC.    
"""
function fcs_table(model::Function, fit::LsqFit.LsqFitResult, scales::AbstractVector; 
                   backend::Symbol=:html, n_diff::Union{Nothing,Int}=nothing, 
                   diffusivity::Union{Nothing, Real}=nothing, gof_metric::Function=bic)
    vals = parameters(fit, scales)
    errs = errors(fit, scales)

    mname = nameof(model)  # Symbol if model is a named function
    model_sym = mname isa Symbol ? mname : :unknown
    parameter_list = infer_parameter_list(model_sym, vals; n_diff, diffusivity)

    if !isnothing(diffusivity) 
        insert!(vals, 1, τD(diffusivity, vals[1]))
        insert!(errs, 1, errs[1] * vals[1] / (2diffusivity)) #assuming no error in diffusivity
    end
    n = min(length(parameter_list), length(vals), length(errs))
    data = hcat(parameter_list[1:n], vals[1:n], errs[1:n])

    gof_val = gof_metric(fit)
    gof_line = " $(nameof(gof_metric)) " * @sprintf("= %.5g ", gof_val)

    column_labels = ["Parameters", "Values", "Std. Dev."]

    pretty_table(
        data;
        backend,
        column_labels = column_labels,
        source_notes = gof_line,
        source_note_alignment = :c,
        alignment = [:l, :r, :r],
        formatters = [(v,i,j)->(j ∈ (2,3) && v isa Number ? @sprintf("%.4g", v) : v)]
    )
end

# Some utility and goodness of fit functions
parameters(fit::LsqFit.LsqFitResult, scale) = fit.param .* scale
errors(fit::LsqFit.LsqFitResult, scale) = stderror(fit) .* scale
function aic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return 2k + N*log(σ2)
end
function aicc(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    a = aic(fit)
    return a + (2k*(k+1)) / (N - k - 1)
end
function bic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return k*log(N) + N*log(σ2)
end
function bicc(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return N * log(σ2) + N * k * log(N) / (N - k - 2)
end