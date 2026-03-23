@testset "naming" begin
    # SI_PREFIXES
    @test FluorescenceCorrelationFitting.SI_PREFIXES[""]  == 1.0
    @test FluorescenceCorrelationFitting.SI_PREFIXES["d"] == 1e1
    @test FluorescenceCorrelationFitting.SI_PREFIXES["c"] == 1e2
    @test FluorescenceCorrelationFitting.SI_PREFIXES["m"] == 1e3
    @test FluorescenceCorrelationFitting.SI_PREFIXES["μ"] == 1e6
    @test FluorescenceCorrelationFitting.SI_PREFIXES["u"] == 1e6
    @test FluorescenceCorrelationFitting.SI_PREFIXES["n"] == 1e9
    @test FluorescenceCorrelationFitting.SI_PREFIXES["p"] == 1e12


    # Parameter name inference helpers
    # Convenience aliases to constants used in names
    G0 = FluorescenceCorrelationFitting.G0_NAME
    OFF = FluorescenceCorrelationFitting.OFF_NAME
    AN = FluorescenceCorrelationFitting.ANOM_NAME
    ST = FluorescenceCorrelationFitting.STRUCT_NAME
    RT = FluorescenceCorrelationFitting.DIFFTIME_NAME
    WF = FluorescenceCorrelationFitting.DIFFFRAC_NAME
    BW = FluorescenceCorrelationFitting.BEAM_NAME
    DT = FluorescenceCorrelationFitting.DYNTIME_NAME
    DF = FluorescenceCorrelationFitting.DYNFRAC_NAME

    # 2D, Brownian, single diffuser, free offset
    spec_2d = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2)
    @test FluorescenceCorrelationFitting.expected_parameter_names(spec_2d) ==
        [G0, OFF, "$(RT) [s]", "$(DT) [1:m] [s]", "$(DF) [1:m]"]

    # 2D, Brownian, single diffuser, fixed offset (removed from p), still τD slots
    spec_2d_fixoff = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2, offset=0.0)
    @test FluorescenceCorrelationFitting.expected_parameter_names(spec_2d_fixoff)[1:1] == [G0]  # no OFF in front matter

    # 2D, diffusion given by w0 (fixed D in spec) → label should be Beam width
    spec_2d_w0 = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2, diffusivity=5e-11)
    # Only checking the base portion that changes wording
    base2d = FluorescenceCorrelationFitting.infer_parameter_names(spec_2d_w0, [0.0, 0.0, 0.0])  # g0, off fixed? no → expect G0, OFF, w0, (no dynamics)
    @test base2d[1:3] == [G0, OFF, "$(BW) [m]"]

    # 2D, anomalous (global α), n_diff=1
    spec_2d_ag = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2, anom=FluorescenceCorrelationFitting.globe)
    nd_ag = FluorescenceCorrelationFitting._no_dynamics_params(spec_2d_ag)
    @test nd_ag == [G0, OFF, "$(RT) [s]", AN]
    # With one dynamics block (τ_dyn1, K_dyn1)
    names_ag = FluorescenceCorrelationFitting.infer_parameter_names(spec_2d_ag, zeros(1 + 1 + 1 + 1 + 2))  # g0, off, τD, α, (τ,K)
    @test names_ag[end-1:end] == ["$(DT) 1 [s]", "$(DF) 1"]

    # 2D, anomalous per-pop, n_diff=2 (α1, α2) + 1 weight
    spec_2d_ap = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2, anom=FluorescenceCorrelationFitting.perpop, n_diff=2)
    nd_ap = FluorescenceCorrelationFitting._no_dynamics_params(spec_2d_ap)
    @test nd_ap == [G0, OFF, "$(RT) 1 [s]", "$(RT) 2 [s]", "$(AN) 1", "$(AN) 2", "$(WF) 1"]

    # 3D, Brownian, single diffuser
    spec_3d = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d3)
    nd_3d = FluorescenceCorrelationFitting._no_dynamics_params(spec_3d)
    @test nd_3d == [G0, OFF, ST, "$(RT) [s]"]

    # 3D, anomalous (global), with dynamics count inferred from params length
    spec_3d_ag = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d3, anom=FluorescenceCorrelationFitting.globe)
    # g0, off, κ, τD, α, (τ1,K1), (τ2,K2)  -> total 5 + 4 = 9
    names_3d_ag = FluorescenceCorrelationFitting.infer_parameter_names(spec_3d_ag, zeros(9))
    @test names_3d_ag[1:5] == [G0, OFF, ST, "$(RT) [s]", AN]
    @test names_3d_ag[6:9] == ["$(DT) 1 [s]", "$(DT) 2 [s]", "$(DF) 1", "$(DF) 2"]


    # sigstr
    @test FluorescenceCorrelationFitting.sigstr(0.0) == "0"
    @test FluorescenceCorrelationFitting.sigstr(Inf) == "Inf"
    @test FluorescenceCorrelationFitting.sigstr(-Inf) == "-Inf"
    @test FluorescenceCorrelationFitting.sigstr(NaN) == "NaN"
    @test FluorescenceCorrelationFitting.sigstr(12.3456, 3) == "12.3"
    @test FluorescenceCorrelationFitting.sigstr(999.999, 4) == "1000"
    @test occursin("e-5", FluorescenceCorrelationFitting.sigstr(9.99e-5, 3))
    @test occursin("e+6", FluorescenceCorrelationFitting.sigstr(1.23456e6, 4))
    @test occursin("e-3", FluorescenceCorrelationFitting.sigstr(9.9999e-4, 4))
    @test FluorescenceCorrelationFitting.sigstr(1.23000, 5) == "1.23"
end
