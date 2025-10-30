module FCSFittingPrettyTablesExt

using PrettyTables
using LsqFit
import StatsAPI: aic, aicc, bic

import FCSFitting: FCSFitResult, sigstr, fcs_table, infer_parameter_names, 
                   τD, SI_PREFIXES, r2

# TODO: restructure and generalize here similar to how was done for fcs_plot
# might also be nice to make a FCSFitResult struct for containing spec + fit + scales :)
"""
    fcs_table(fit; backend=:html, gof_metric=bic, units=nothing)

Render a **parameter table** from an `FCSFitResult`.

# Arguments
- `fit::FCSFitResult` — Result from `fcs_fit`

# Keywords
- `backend::Symbol=:html` — `PrettyTables` backend (`:html`, `:unicode`, `:latex`, etc.).
- `gof_metric::Function=bic` — A function `gof_metric(fit)::Real` (e.g., `aic`, `aicc`, `bic`, `bicc`).
- `units::Union{Nothing, AbstractVector{String}}` — If provided, rescales parameter values to the 
                                                    corresponding SI prefix

# Output
Prints a table with columns:
- `"Parameters"` — Human-readable names from `infer_parameter_list(...)`,
- `"Values"` — `parameters(fit, scales)`,
- `"Std. Dev."` — `errors(fit, scales)`,

and a source note with the chosen GoF metric.

# Returns
- The return value of `pretty_table(...)` after printing the table.

# Notes
If `diffusivity` is provided, `τ_D` is computed and inserted at the top; the simple error propagation
assumes no uncertainty in `diffusivity`.
"""
function fcs_table(fit::FCSFitResult; 
                   backend::Symbol=:html, gof_metric::Function=bic,
                   units::Union{Nothing, AbstractVector{String}}=nothing)
    vals = coef(fit)
    errs = stderror(fit)
    
    # Build parameter list (names) in the same order as values
    parameter_list = infer_parameter_names(fit.spec, vals)

    # Trim to the common length
    n = min(length(parameter_list), length(vals), length(errs))
    parameter_list = parameter_list[1:n]
    vals = vals[1:n]
    errs = errs[1:n]

    # argument checks for SI prefix rescaling
    (units === nothing) && (units = fill("",n))
    length(units) < n && throw(ArgumentError("`units` length ($(length(units))) < number of displayed parameters ($n)."))
    # Validate keys and build multipliers
    bad = [u for u in units[1:n] if !haskey(SI_PREFIXES, u)]
    !isempty(bad) && throw(ArgumentError("Unknown SI prefixes in `units`."))
    # Apply scaling to values and errors
    multipliers = getindex.(Ref(SI_PREFIXES), units[1:n])
    @inbounds for i in 1:n
        vals[i] *= multipliers[i]
        errs[i] *= multipliers[i]
    end
    # Decorate displayed unit labels with the chosen prefix where present.
    # We only touch simple base units [s] and [m]; leave anything else as-is.
    @inbounds for i in 1:n
        u = units[i]
        if u != ""
            # Add prefix before the base symbol when found.
            # e.g. "[s]" -> "[μs]" and "[m]" -> "[nm]".
            parameter_list[i] = replace(parameter_list[i],
                "[s]" => "[" * u * "s]",
                "[m]" => "[" * u * "m]",
            )
        end
    end

    data = hcat(parameter_list, vals, errs)
    # evaluate goodness of fit metric and add it to the table
    gof_val = gof_metric(fit)
    gof_line = " $(nameof(gof_metric)) = $(sigstr(gof_val, 6)) "

    # PrettyTables call
    column_labels = ["Parameters", "Values", "Std. Dev."]

    pretty_table(
        data;
        backend,
        column_labels = column_labels,
        source_notes = gof_line,
        source_note_alignment = :c,
        alignment = [:l, :r, :r],
        formatters = [(v,i,j)->(j ∈ (2,3) && v isa Number ? sigstr(v, 4) : v)],
    )
end

end # module