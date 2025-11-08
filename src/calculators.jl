PosError(x) = ArgumentError(string(x, " must be positive."))
const w0_SIGN_ERROR = PosError("w0")
const κ_SIGN_ERROR = PosError("κ")
const D_SIGN_ERROR = PosError("Diffusivity")
const nd_ERROR = ArgumentError("The chosen diffuser must be at a positive index less than the total number of diffusers.")
const w0_REQUIRED_ERROR = ArgumentError("This model does not fix diffusivity, so you must provide `w0=` to compute D.")


"""
    τD(D, w0; scale="")

Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
function τD(D::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    D > 0 || throw(D_SIGN_ERROR)
    
    diff_time = w0^2 / (4D)
    haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
    return diff_time
end

"""
    τD(spec, fit; nd=1, scale="")

Get the nd-th diffusion time from a fitted model.

- If the spec used a **fixed diffusivity**, the corresponding slot in the fit is w₀,
  so we convert w₀ → τD.
- Otherwise, the slot is already τD so is scaled by `scale`
"""
function τD(spec::FCSModelSpec, fit::FCSFitResult; nd::Int = 1, scale::String = "")
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    θ = coef(fit)
    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)

    diff_slot = idx + nd
    if hasdiffusivity(spec) # slot holds w0
        w0 = θ[diff_slot]
        return τD(spec.diffusivity, w0; scale)
    else # slot already holds τD
        diff_time = θ[diff_slot]
        haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
        return diff_time
    end
end

"""
    diffusivity(τD, w0; scale="")

Convert diffusion time `τD` and beam waist `w0` to the diffusivity.
"""
function diffusivity(τD::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    τD > 0 || throw(PosError("τD"))

    diff = w0^2 / (4τD)
    # if the user says e.g. "μ", we scale the length unit; D is m^2/s → (prefix m)^2/s
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

"""
    diffusivity(spec; scale="")

Return the fixed diffusivity from the spec (error if it was not fixed).
"""
function diffusivity(spec::FCSModelSpec{D,S,OFF,true}; scale::String = "") where {D,S,OFF}
    diff = spec.diffusivity
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

"""
    diffusivity(spec, fit; nd=1, w0_known=nothing, scale="")

For models **without** fixed diffusivity, diffusion slots in the fit are τD’s.
To get D you must also provide w₀ (because τD → D needs w₀).

If your model used fixed diffusivity, use `diffusivity(spec)` instead.
"""
function diffusivity(spec::FCSModelSpec{D,S,OFF,false}, fit::FCSFitResult;
                     nd::Int=1, w0::Union{Nothing,Real}=nothing, scale::String="") where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)

    τd = τD(spec, fit; nd, scale = "")  # get τD in base units
    return diffusivity(τd, w0; scale)
end

"""
    Veff(w0, κ; scale="")

Calculate the effective volume from fitted FCS parameters.
"""
function Veff(w0::Real, κ::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    κ > 0 || throw(κ_SIGN_ERROR)

    vol = π^(3/2) * w0^3 * κ
    haskey(SI_PREFIXES, scale) && (vol *= SI_PREFIXES[scale]^3)
    return vol
end

"""
    Veff(spec, fit; nd=1, scale="")

3D-only. Pull κ and the nd-th w₀/τD from the fit and compute Veff.

- if diffusivity was fixed → diffusion slot is w₀ → we can compute Veff
- if diffusivity was free → diffusion slot is τD → user must give w₀
"""
function Veff(spec::FCSModelSpec{d3,S,OFF,true}, fit::FCSFitResult;
              nd::Int = 1, scale::String = "") where {S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    θ = coef(fit)

    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]
    w0 = θ[idx + nd]
    return Veff(w0, κ; scale)
end

function Veff(spec::FCSModelSpec{d3,S,OFF,false}, fit::FCSFitResult;
              w0::Union{Nothing,Real}=nothing, scale::String = "") where {S,OFF}
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]
    return Veff(w0, κ; scale)
end

"""
    Aeff(w0; scale="")

Calculate the area formed by the beam waist `w0`.
"""
function Aeff(w0::Real; scale::String="") 
    w0 > 0 || throw(w0_SIGN_ERROR)
    
    area = π * w0^2
    haskey(SI_PREFIXES, scale) && (area *= SI_PREFIXES[scale]^2)
    return area
end

"""
    Aeff(spec, fit; nd=1, scale="")

Extract w₀ from the fit and compute the confocal area.
"""
function Aeff(spec::FCSModelSpec{d2,S,OFF,true,NDIFF}, fit::FCSFitResult;
              nd::Int = 1, scale::String = "") where {S,OFF,NDIFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    θ = coef(fit)

    idx = 1
    !hasoffset(spec) && (idx += 1)
    w0 = θ[idx + nd]

    return Aeff(w0; scale)
end

"""
    concentration(g0, κ, w0; Ks=[], ics=[0], scale="L")

Estimate the **molar concentration** (in mol/L) from FCS fit parameters.

# Arguments
- `w0::Real`: Lateral 1/e² Gaussian waist of the detection PSF (meters).
- `κ::Real`: Axial structure factor `κ = wz / w0` (dimensionless).
- `g0::Real`: Fitted correlation amplitude **at τ→0** (dimensionless).
              In standard FCS models, the measured `g0` is inflated by blinking (“dark states”).
- `Ks::AbstractVector` (keyword): Dark-state **equilibrium fractions** for the kinetic terms
   used in the model (each in `[0,1)`), ordered exactly as in your dynamics kernel.
- `ics::AbstractVector{Int}` (keyword): Block sizes describing how `Ks` (and their times)
   are grouped into **independent** multiplicative blinking factors. For example,
   `ics = [2, 1]` means the first blinking block has 2 components, the second block has 1.
"""
function concentration(g0::Real, κ::Real, w0::Real; Ks::AbstractVector = Float64[],
                       ics::AbstractVector{Int} = Int[], scale::String="")
    g0 > 0 || throw(PosError("g0"))
    κ > 0 || throw(κ_SIGN_ERROR)
    w0 > 0 || throw(w0_SIGN_ERROR)
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == []) || throw(ArgumentError("If there are no dynamic fractions, ics must be empty.")) :
        sum(ics) == length(Ks) || throw(DYN_COMP_ERROR)

    # Blink prefactor at τ→0: B(0) = ∏_blocks (1 + Σ_i n_i),  n_i = K_i/(1-K_i)
    B0 = one(Float64)
    idx = 1
    for b in eachindex(ics)
        nb = ics[b]
        nb == 0 && continue

        s = 0.0
        @inbounds for j = 1:nb
            K = float(Ks[idx + j - 1])
            s += K / (1 - K)
        end
        B0 *= (1 + s)
        idx += nb
    end

    # base concentration: mol / L
    conc = B0 * 1e-3 / (g0 * AVAGADROS * Veff(w0, κ))
    if haskey(SI_PREFIXES, scale) # mol/L → (prefix)·mol/L
        conc *= SI_PREFIXES[scale]
    end
    return conc
end

"""
    concentration(spec, fit; nd=1, Ks=[], scale="L")

Pull g₀, κ, w₀ (in the fixed-diffusivity case), and Ks from the fit 
and determine the concentration.
"""
function concentration(spec::FCSModelSpec{d3,S,OFF,true}, fit::FCSFitResult;
                       nd::Int = 1, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx];  idx += 1
    w0 = θ[idx+nd-1]

    # total number of components in θ allocated to τD/ w₀ + weights
    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Kdyn = m == 0 ? Float64[] : collect(@view θ[idx+m:idx+2m-1])

    return concentration(g0, κ, w0; Ks=Kdyn, ics, scale)
end

function concentration(spec::FCSModelSpec{d3,S,OFF,false}, fit::FCSFitResult; nd::Int = 1, 
                       w0::Union{Nothing,Real}=nothing, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx];  idx += 1

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Kdyn = m == 0 ? Float64[] : collect(@view θ[idx+m:idx+2m-1])

    return concentration(g0, κ, w0; Ks=Kdyn, ics, scale)
end

"""
    surface_density(w0, g0; Ks=[], ics=[0], scale="")

Estimate the **molar surface density** (in mol/m^2) from FCS fit parameters.
Analogue to `concentration` when the a 2d fit is performed.
"""
function surface_density(g0::Real, w0::Real; Ks::AbstractVector = Float64[],
                         ics::AbstractVector{Int} = Int[], scale::String="")
    g0 > 0 || throw(PosError("g0"))
    w0 > 0 || throw(w0_SIGN_ERROR)
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == []) || throw(ArgumentError("If there are no dynamic fractions, ics must be empty.")) :
        sum(ics) == length(Ks) || throw(DYN_COMP_ERROR)

    # Blink prefactor at τ→0: B(0) = ∏_blocks (1 + Σ_i n_i),  n_i = K_i/(1-K_i)
    B0 = one(Float64)
    idx = 1
    for b in eachindex(ics)
        nb = ics[b]
        nb == 0 && continue

        s = 0.0
        @inbounds for j = 1:nb
            K = float(Ks[idx + j - 1])
            s += K / (1 - K)
        end
        B0 *= (1 + s)
        idx += nb
    end

    dens = B0 / (g0 * AVAGADROS * Aeff(w0))
    haskey(SI_PREFIXES, scale) && (dens *= SI_PREFIXES[scale])
    return dens
end

"""
    surface_density(spec, fit; nd=1, scale="")

2D analogue of `concentration(spec, fit, ...)`.

Pulls g₀ and w₀ (in the fixed-diffusivity case) from the fitted parameter
vector, then reconstructs the dynamic fractions from the tail of the vector, and
finally calls the base `surface_density(g0, w0; Ks, ics, scale)`.
"""
function surface_density(spec::FCSModelSpec{d2,S,OFF,true}, fit::FCSFitResult;
                         nd::Int = 1, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    w0 = θ[idx + nd - 1]

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Ks = m == 0 ? Float64[] : collect(@view θ[dyn_start + m : dyn_start + 2m - 1])

    return surface_density(g0, w0; Ks, ics, scale)
end

"""
    surface_density(spec, fit; nd=1, w0=..., scale="")

2D, **free diffusivity**: the diffusion slots hold τᴅ, not w₀, so the user
must supply `w0=` to convert to a surface density.
"""
function surface_density(spec::FCSModelSpec{d2,S,OFF,false}, fit::FCSFitResult;
                         nd::Int = 1, w0::Union{Nothing,Real} = nothing,
                         scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Ks = m == 0 ? Float64[] : collect(@view θ[dyn_start + m : dyn_start + 2m - 1])

    return surface_density(g0, w0; Ks, ics, scale)
end

"""
    hydrodynamic(D; T=293.0, η=1.0016e-3, scale="")
    hydrodynamic(τD, w0; T=293.0, η=1.0016e-3, scale="")

Calculate the effective hydrodynamic radius of a molecule using the Stokes-Einstein relation.

# Keyword Arguments
- `T=293.0`: Temperature (in Kelvin)
- `η=1.0016e-3`: Viscosity of water (Pa⋅s)
"""
function hydrodynamic(D::Real; T=293.0, η=1.0016e-3, scale::String="")
    D > 0 || throw(D_SIGN_ERROR)
    T > 0 || throw(PosError("Temperature"))
    η > 0 || throw(PosError("Viscosity"))

    rh = BOLTZMANN * T / (6π * η * D)
    haskey(SI_PREFIXES, scale) && (rh *= SI_PREFIXES[scale])
    return rh
end

hydrodynamic(τD::Real, w0::Real; kwargs...) =
    hydrodynamic(diffusivity(τD, w0); kwargs...)