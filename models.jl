module FCSModels

export τD_from_D, udc_2d, udc_2d_anom, udc_3d,
       fcs_2d, fcs_2d_mdiff, fcs_2d_anom_mdiff,
       fcs_3d, fcs_3d_mdiff


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    τD_from_D(D, w0)

Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
@inline τD_from_D(D::Real, w0::Real) = w0^2 / (4D)

# clamp into (ε, 1-ε) to keep fractions valid without NaNs/Infs in fits
@inline function clamp01(x::T) where {T}
    epsT = eps(T)
    return clamp(x, epsT, one(T) - epsT)
end

# broadcastable vectors from Real or AbstractVector
@inline _asvec(x::AbstractVector) = x
@inline _asvec(x::Real) = (x,)

# normalize mixture weights to sum to 1 without mutating the input
function _normalize_weights(ws::AbstractVector)
    T = promote_type(eltype(ws), Float64)
    w = T.(ws)
    s = sum(w)
    s > 0 ? (w ./ s) : fill(inv(length(w)), length(w))
end

# multiplicative blinking/dynamics factor Π_j [1 + K_j/(1-K_j) * exp(-t/τ_j)]
# returns 1 if no (τ,K) provided
function _dynamics_factor(t, τs::AbstractVector, Ks::AbstractVector)
    length(τs) == length(Ks) || throw(ArgumentError("τs and Ks must have same length"))
    if isempty(τs)
        return t isa AbstractVector ? ones(promote_type(eltype(t), Float64), length(t)) : one(promote_type(typeof(t), Float64))
    end
    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(τs), eltype(Ks))
        out = ones(T, length(t))
        @inbounds for i in eachindex(τs)
            K = clamp01(T(Ks[i]))
            τ = T(τs[i])
            @. out *= (one(T) + K/(one(T)-K) * exp(-t/τ))
        end
        return out
    else
        T = promote_type(typeof(t), eltype(τs), eltype(Ks))
        out = one(T)
        @inbounds for i in eachindex(τs)
            K = clamp01(T(Ks[i]))
            τ = T(τs[i])
            out *= (one(T) + K/(one(T)-K) * exp(-T(t)/τ))
        end
        return out
    end
end

# sum_i w_i * kernel(t, param_i), for t scalar or vector
function _mdiff(t, params::AbstractVector, weights::AbstractVector, kernel::Function)
    length(params) == length(weights) || throw(ArgumentError("params and weights must have same length"))
    w = _normalize_weights(weights)
    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(params), eltype(weights))
        out = zeros(T, length(t))
        @inbounds for i in eachindex(params)
            @. out += T(w[i]) * kernel(t, T(params[i]))
        end
        return out
    else
        T = promote_type(typeof(t), eltype(params), eltype(weights))
        s = zero(T)
        @inbounds for i in eachindex(params)
            s += T(w[i]) * kernel(T(t), T(params[i]))
        end
        return s
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Base kernels (unit-amplitude, zero-offset)
# ─────────────────────────────────────────────────────────────────────────────

"2D diffusion kernel: 1 / (1 + t/τD)"
@inline function udc_2d(t, τD) 
    if t isa AbstractVector 
        @. inv(1 + t/τD) 
    else
        inv(1 + t/τD)
    end
end

"2D anomalous diffusion kernel: 1 / (1 + (t/τD)^α)"
@inline function udc_2d_anom(t, τD, α)
    if t isa AbstractVector
        @. inv(1 + (t/τD)^α)
    else
        inv(1 + (t/τD)^α)
    end
end

"3D diffusion kernel with structure factor s = z0/w0: 1 / ((1 + t/τD) * sqrt(1 + t/(s^2 τD)))"
@inline function udc_3d(t, τD, s)
    if t isa AbstractVector
        @. inv( (1 + t/τD) * sqrt(1 + t/(s^2 * τD)) )
    else
        inv( (1 + t/τD) * sqrt(1 + t/(s^2 * τD)) )
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Public models (with amplitude g0 and offset, plus optional blinking)
# ─────────────────────────────────────────────────────────────────────────────

"""
    fcs_2d(t; τD, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Single-component 2D diffusion with optional multiplicative dynamics (triplet/blinking).
"""
function fcs_2d(t::Union{Real,AbstractVector{<:Real}}; τD::Real=1, g0::Real=1, offset::Real=0,
                 τ_dyn::Union{AbstractVector,Tuple,NTuple}=(), K_dyn::Union{AbstractVector,Tuple,NTuple}=())
    dyn = _dynamics_factor(t, collect(τ_dyn), collect(K_dyn))
    udc = udc_2d(t, τD)
    if t isa AbstractVector
        return @. offset + g0 * udc * dyn
    else
        return offset + g0 * udc * dyn
    end 
end

"""
    fcs_2d_mixture(t; τDs, weights, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Mixture of 2D diffusion components with weights that are normalized internally.
"""
function fcs_2d_mdiff(t::Union{Real,AbstractVector{<:Real}}; τDs::AbstractVector=[1],
                      weights::AbstractVector=[1], g0::Real=1, offset::Real=0,
                      τ_dyn::Union{AbstractVector,Tuple,NTuple}=(), 
                      K_dyn::Union{AbstractVector,Tuple,NTuple}=())
    dyn = _dynamics_factor(t, collect(τ_dyn), collect(K_dyn))
    mix = _mdiff(t, τDs, weights, (tt,τ)->udc_2d(tt,τ))
    if t isa AbstractVector
        return @. offset + g0 * mix * dyn
    else
        return offset + g0 * mix * dyn
    end 
end

"""
    fcs_2d_anom_mixture(t; τDs, αs, weights, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Mixture of 2D anomalous components with exponents αs (same length as τDs).
"""
function fcs_2d_anom_mdiff(t::Union{Real,AbstractVector{<:Real}}; τDs::AbstractVector=[1],
                           αs::AbstractVector=[1], weights::AbstractVector=[1], g0::Real=1, 
                           offset::Real=0, τ_dyn::Union{AbstractVector,Tuple,NTuple}=(), 
                           K_dyn::Union{AbstractVector,Tuple,NTuple}=())
    length(τDs) == length(αs) == length(weights) || throw(ArgumentError("τDs, αs, weights must have same length"))
    dyn = _dynamics_factor(t, collect(τ_dyn), collect(K_dyn))
    # small wrapper to pass paired params
    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(τDs), eltype(αs), eltype(weights))
        out = zeros(T, length(t))
        w = _normalize_weights(weights)
        @inbounds for i in eachindex(τDs)
            @. out += T(w[i]) * udc_2d_anom(t, T(τDs[i]), T(αs[i]))
        end
        return @. offset + g0 * out * dyn
    else
        T = promote_type(typeof(t), eltype(τDs), eltype(αs), eltype(weights))
        s = zero(T)
        w = _normalize_weights(weights)
        @inbounds for i in eachindex(τDs)
            s += T(w[i]) * udc_2d_anom(T(t), T(τDs[i]), T(αs[i]))
        end
        return offset + g0 * s * dyn
    end
end

"""
    fcs_3d(t; τD, s, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Single-component 3D diffusion with structure factor `s = z0/w0` and optional dynamics.
"""
function fcs_3d(t::Union{Real,AbstractVector{<:Real}}; τD::Real=1, 
                s::Real=1, g0::Real=1, offset::Real=0,
                τ_dyn::Union{AbstractVector,Tuple,NTuple}=(), 
                K_dyn::Union{AbstractVector,Tuple,NTuple}=())
    dyn = _dynamics_factor(t, collect(τ_dyn), collect(K_dyn))
    udc = udc_3d(t, τD, s)
    if t isa AbstractVector
        return @. offset + g0 * udc * dyn
    else
        return offset + g0 * udc * dyn
    end
end

"""
    fcs_3d_mixture(t; τDs, s, weights, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Mixture of 3D diffusion components sharing the same structure factor `s`.
"""
function fcs_3d_mdiff(t::Union{Real,AbstractVector{<:Real}}; τDs::AbstractVector=[1], 
                      s::Real=1, weights::AbstractVector=[1], g0::Real=1, offset::Real=0,
                      τ_dyn::Union{AbstractVector,Tuple,NTuple}=(), 
                      K_dyn::Union{AbstractVector,Tuple,NTuple}=())
    dyn = _dynamics_factor(t, collect(τ_dyn), collect(K_dyn))
    mix = _mdiff(t, τDs, weights, (tt,τ)->udc_3d(tt, τ, s))
    if t isa AbstractVector
        return @. offset + g0 * mix * dyn
    else
        return offset + g0 * mix * dyn
    end
end



# ─────────────────────────────────────────────────────────────────────────────
# Global multi-curve wrappers
# ─────────────────────────────────────────────────────────────────────────────

_as_tuple(x::Tuple) = x
_as_tuple(x::AbstractVector) = Tuple(x)

"""
    global_fcs_2d(ts; τD, curves) -> NTuple{N,Vector}

Evaluate 2D FCS on multiple curves with a *shared* diffusion time `τD`.

- `ts`: Tuple or Vector of lag-time vectors, one per curve.
- `curves`: Tuple or Vector of NamedTuples with keys:
    `g0::Real`, `offset::Real`, `τ_dyn::AbstractVector` (or empty tuple),
    `K_dyn::AbstractVector` (or empty tuple).

Returns an `NTuple` with one model vector per input curve, preserving order.
"""
function global_fcs_2d(ts; τD::Real, curves)
    tset = _as_tuple(ts)
    cparams = _as_tuple(curves)
    length(tset) == length(cparams) ||
        throw(ArgumentError("length(ts) must match length(curves)"))

    n = length(tset)
    return ntuple(i -> begin
        c = cparams[i]
        fcs_2d(tset[i];
               τD = τD,
               g0 = c.g0,
               offset= c.offset,
               τ_dyn = getfield(c, :τ_dyn),
               K_dyn = getfield(c, :K_dyn))
    end, n)
end

"""
    global_fcs_2d_mdiff(ts; τDs, weights, curves) -> NTuple{N,Vector}

Evaluate 2D *mixture* FCS on multiple curves with a shared set of `τDs` and `weights`.

- `ts`: Tuple/Vector of lag-time vectors.
- `τDs`: AbstractVector of diffusion times (shared).
- `weights`: AbstractVector of mixture weights (shared; normalized internally).
- `curves`: per-curve NamedTuples as in `global_fcs_2d`.

Returns an `NTuple` of model vectors.
"""
function global_fcs_2d_mdiff(ts; τDs::AbstractVector, weights::AbstractVector, curves)
    tset    = _as_tuple(ts)
    cparams = _as_tuple(curves)
    length(tset) == length(cparams) ||
        throw(ArgumentError("length(ts) must match length(curves)"))

    n = length(tset)
    return ntuple(i -> begin
        c = cparams[i]
        fcs_2d_mdiff(tset[i];
                     τDs = τDs,
                     weights = weights,
                     g0 = c.g0,
                     offset = c.offset,
                     τ_dyn = getfield(c, :τ_dyn),
                     K_dyn = getfield(c, :K_dyn))
    end, n)
end

"""
    global_fcs_2d_anom_mdiff(ts; τDs, αs, weights, curves) -> NTuple{N,Vector}

Evaluate 2D *anomalous* mixture on multiple curves with shared `τDs`, `αs`, and `weights`.
"""
function global_fcs_2d_anom_mdiff(ts; τDs::AbstractVector, αs::AbstractVector,
                                  weights::AbstractVector, curves)
    tset = _as_tuple(ts)
    cparams = _as_tuple(curves)
    length(tset) == length(cparams) ||
        throw(ArgumentError("length(ts) must match length(curves)"))

    n = length(tset)
    return ntuple(i -> begin
        c = cparams[i]
        fcs_2d_anom_mdiff(tset[i];
                          τDs = τDs,
                          αs = αs,
                          weights = weights,
                          g0 = c.g0,
                          offset = c.offset,
                          τ_dyn = getfield(c, :τ_dyn),
                          K_dyn = getfield(c, :K_dyn))
    end, n)
end

"""
    global_fcs_3d(ts; τD, curves) -> NTuple{N,Vector}

3D FCS on multiple curves with a shared `τD`. Each curve supplies its own
structure factor `s` (inside `curves`), plus `g0`, `offset`, and optional dynamics.
`curves[i]` must have keys: `s`, `g0`, `offset`, `τ_dyn`, `K_dyn`.
"""
function global_fcs_3d(ts; τD::Real, curves)
    tset = _as_tuple(ts)
    cparams = _as_tuple(curves)
    length(tset) == length(cparams) ||
        throw(ArgumentError("length(ts) must match length(curves)"))

    n = length(tset)
    return ntuple(i -> begin
        c = cparams[i]
        fcs_3d(tset[i];
               τD = τD,
               s = c.s,
               g0 = c.g0,
               offset = c.offset,
               τ_dyn = getfield(c, :τ_dyn),
               K_dyn = getfield(c, :K_dyn))
    end, n)
end


"""
    global_fcs_3d_mdiff(ts; τDs, weights, curves) -> NTuple{N,Vector}

3D *mixture* with shared `τDs`/`weights` across all curves.
Each curve supplies its own `s`, `g0`, `offset`, and optional dynamics.
"""
function global_fcs_3d_mdiff(ts; τDs::AbstractVector, weights::AbstractVector, curves)
    tset = _as_tuple(ts)
    cparams = _as_tuple(curves)
    length(tset) == length(cparams) ||
        throw(ArgumentError("length(ts) must match length(curves)"))

    n = length(tset)
    return ntuple(i -> begin
        c = cparams[i]
        fcs_3d_mdiff(tset[i];
                     τDs = τDs,
                     s = c.s,
                     weights = weights,
                     g0 = c.g0,
                     offset = c.offset,
                     τ_dyn = getfield(c, :τ_dyn),
                     K_dyn = getfield(c, :K_dyn))
    end, n)
end

end # module