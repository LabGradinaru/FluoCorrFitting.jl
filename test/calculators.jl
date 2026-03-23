@testset "calculators" begin
    NA = FluorescenceCorrelationFitting.AVAGADROS
    kB = FluorescenceCorrelationFitting.BOLTZMANN

    @testset "Base functionality" begin
        D = 5e-11
        w0 = 250e-9
        κ = 10 * rand()
        g0 = rand()
        Ks = [0.1, 0.2]

        # τD and diffusivity
        τ = FluorescenceCorrelationFitting.τD(D, w0)
        @test τ ≈ (w0^2) / (4D)
        τ_scale = FluorescenceCorrelationFitting.τD(D, w0; scale="μ")
        @test τ_scale ≈ 1e6 * (w0^2) / (4D)
        D_back = FluorescenceCorrelationFitting.diffusivity(τ, w0)
        @test D_back ≈ D

        @test_throws ArgumentError FluorescenceCorrelationFitting.τD(-1.0, w0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.τD(D, -1.0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.diffusivity(-1.0, w0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.diffusivity(τ, -1.0)

        # confocal volume/area
        vol = FluorescenceCorrelationFitting.Veff(w0, κ)
        @test vol ≈ π^(3/2) * w0^3 * κ rtol=1e-12
        vol_scale = FluorescenceCorrelationFitting.Veff(w0, κ; scale="n")
        @test vol_scale ≈ 1e27 * π^(3/2) * w0^3 * κ
        ar = FluorescenceCorrelationFitting.Aeff(w0)
        @test ar ≈ π * w0^2 rtol=1e-12
        ar_scale = FluorescenceCorrelationFitting.Aeff(w0; scale="n")
        @test ar_scale ≈ 1e18 * π * w0^2

        @test_throws ArgumentError FluorescenceCorrelationFitting.Veff(-1.0, κ)
        @test_throws ArgumentError FluorescenceCorrelationFitting.Veff(w0, -1.0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.Aeff(-1.0)

        # concentration (blinkless)
        c = FluorescenceCorrelationFitting.concentration(g0, κ, w0)
        @test c ≈ (1/g0) / (NA * vol * 1000.0)
        c_scale = FluorescenceCorrelationFitting.concentration(g0, κ, w0; scale="m")
        @test c_scale ≈ (1/g0) / (NA * vol)

        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(-1.0, κ, w0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(g0, -1.0, w0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(g0, κ, -1.0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(g0, κ, w0; Ks=[-0.1])
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(g0, κ, w0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(g0, κ, w0; Ks=[], ics=[1])

        B0 = 1 + (0.1/0.9 + 0.2/0.8)
        Neff1 = B0 / g0
        c_expected = Neff1 / (NA * vol * 1000.0)
        @test FluorescenceCorrelationFitting.concentration(g0, κ, w0; Ks=Ks, ics=[2]) ≈ c_expected

        B0 = (1 + 0.1/0.9) * (1 + 0.2/0.8)
        Neff2 = B0 / g0
        c_expected = Neff2 / (NA * vol * 1000.0)
        @test FluorescenceCorrelationFitting.concentration(g0, κ, w0; Ks=Ks, ics=[1,1]) ≈ c_expected

        # surface density (2D analogue)
        sa = FluorescenceCorrelationFitting.surface_density(g0, w0)
        sa_expected = (1/g0) / (NA * ar)
        @test sa ≈ sa_expected rtol=1e-12

        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(-1.0, w0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(g0, -1.0)
        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(g0, w0; Ks=[-0.1])
        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(g0, w0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(g0, w0; Ks=[], ics=[1])

        sa_expected = Neff1 / (NA * ar)
        @test FluorescenceCorrelationFitting.surface_density(g0, w0; Ks=Ks, ics=[2]) ≈ sa_expected
        sa_expected = Neff2 / (NA * ar)
        @test FluorescenceCorrelationFitting.surface_density(g0, w0; Ks=Ks, ics=[1,1]) ≈ sa_expected

        # hydrodynamic radius
        T = 293.0;  η = 1.0016e-3
        Rh = FluorescenceCorrelationFitting.hydrodynamic(D; T=T, η=η)
        @test Rh ≈ kB * T / (6π * η * D)
        Rh_scale = FluorescenceCorrelationFitting.hydrodynamic(D; T=T, η=η, scale="A")
        @test Rh_scale ≈ 1e10 * kB * T / (6π * η * D)

        @test_throws ArgumentError FluorescenceCorrelationFitting.hydrodynamic(-1.0; T=T, η=η)
        @test_throws ArgumentError FluorescenceCorrelationFitting.hydrodynamic(D; T=-1.0, η=η)
        @test_throws ArgumentError FluorescenceCorrelationFitting.hydrodynamic(D; T=T, η=-1.0)
    end


    @testset "From FCSModelSpec + FCSFitResult" begin
        D = 1e-10 * rand()
        w0 = 500e-9 * rand()
        κ = 10*rand()
        g0 = rand()
        
        τ = 10 .^ range(-6, 0; length=300)

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, one diffuser → params = [g0, κ, w0]
        # ------------------------------------------------------------
        spec3 = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d3,
            anom = FluorescenceCorrelationFitting.none,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        
        p0_3 = [0.5, 5, 250e-9]
        model3 = FluorescenceCorrelationFitting.FCSModel(spec3, τ, p0_3)
        y3_true = model3(τ, [g0, κ, w0])

        fit3 = FluorescenceCorrelationFitting.fcs_fit(spec3, τ, y3_true, p0_3)

        τD_expected = FluorescenceCorrelationFitting.τD(D, w0)
        @test FluorescenceCorrelationFitting.τD(spec3, fit3) ≈ τD_expected rtol=1e-6
        @test FluorescenceCorrelationFitting.τD(spec3, fit3; scale="μ") ≈ 1e6 * τD_expected rtol=1e-6

        @test FluorescenceCorrelationFitting.diffusivity(spec3, fit3) ≈ D rtol=1e-12
        @test FluorescenceCorrelationFitting.diffusivity(spec3, fit3; scale="μ") ≈ 1e12 * D rtol=1e-12

        V_expected = FluorescenceCorrelationFitting.Veff(w0, κ)
        @test FluorescenceCorrelationFitting.Veff(spec3, fit3) ≈ V_expected rtol=1e-6
        @test FluorescenceCorrelationFitting.Veff(spec3, fit3; scale="n") ≈ 1e27 * V_expected rtol=1e-6

        c_expected = FluorescenceCorrelationFitting.concentration(g0, κ, w0)
        @test FluorescenceCorrelationFitting.concentration(spec3, fit3; scale="L") ≈ c_expected rtol=1e-6
        @test FluorescenceCorrelationFitting.concentration(spec3, fit3; scale="n") ≈ 1e9 * c_expected rtol=1e-6
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(spec3, fit3; nd=2) # more diffusers than allowed
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(spec3, fit3; nd=0) # fewer diffusers than allowed

        # ------------------------------------------------------------
        # 3D, free diffusivity, one diffuser → params = [g0, κ, τD]
        # ------------------------------------------------------------
        spec3_free = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d3,
            anom = FluorescenceCorrelationFitting.none,
            offset = 0.0,
            n_diff = 1,
        )
        
        p0_3free = [0.5, 5, 0.0003125]
        model3_free = FluorescenceCorrelationFitting.FCSModel(spec3_free, τ, p0_3free)
        y3_free_true = model3_free(τ, [g0, κ, τD_expected])

        lower = [0.0, 1.0, 1e-5]
        upper = [1.5, 10.0, 1e-3]
        fit3_free = FluorescenceCorrelationFitting.fcs_fit(spec3_free, τ, y3_free_true, p0_3free; lower, upper)

        @test_throws ArgumentError FluorescenceCorrelationFitting.Veff(spec3_free, fit3_free)
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(spec3_free, fit3_free)
        @test_throws ArgumentError FluorescenceCorrelationFitting.concentration(spec3_free, fit3_free; w0=w0, nd=0)

        @test FluorescenceCorrelationFitting.Veff(spec3_free, fit3_free; w0 = w0) ≈ V_expected rtol=1e-6
        @test FluorescenceCorrelationFitting.concentration(spec3_free, fit3_free; w0=w0) ≈ c_expected rtol=1e-6

        # ------------------------------------------------------------
        # 2D, fixed diffusivity, one diffuser → params = [g0, w0]
        # ------------------------------------------------------------
        spec2_fixed = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d2,
            anom = FluorescenceCorrelationFitting.none,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        
        p0_2f = [0.5, 250e-9]
        model2_fixed = FluorescenceCorrelationFitting.FCSModel(spec2_fixed, τ, p0_2f)
        y2_fixed_true = model2_fixed(τ, [g0, w0])
        
        fit2_fixed = FluorescenceCorrelationFitting.fcs_fit(spec2_fixed, τ, y2_fixed_true, p0_2f)

        ar = FluorescenceCorrelationFitting.Aeff(w0)
        @test FluorescenceCorrelationFitting.Aeff(spec2_fixed, fit2_fixed) ≈ ar rtol=1e-6

        sa_expected = (1/g0) / (NA * ar)
        sa_from_fit = FluorescenceCorrelationFitting.surface_density(spec2_fixed, fit2_fixed)
        @test sa_from_fit ≈ sa_expected rtol=1e-6

        # ------------------------------------------------------------
        # 2D, *free* diffusivity, one diffuser → params = [g0, τD]
        # ------------------------------------------------------------
        τD_true = 1e-3*rand()
        spec2_free = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d2,
            anom = FluorescenceCorrelationFitting.none,
            offset = 0.0,
            n_diff = 1,
        )
        
        p0_2free = [0.3, τD_true*0.8]
        model2_free = FluorescenceCorrelationFitting.FCSModel(spec2_free, τ, p0_2free)
        y2_free_true = model2_free(τ, [g0, τD_true])
 
        fit2_free = FluorescenceCorrelationFitting.fcs_fit(spec2_free, τ, y2_free_true, p0_2free)

        # calling without w0 should error
        @test_throws ArgumentError FluorescenceCorrelationFitting.surface_density(spec2_free, fit2_free)
        @test_throws ArgumentError FluorescenceCorrelationFitting.diffusivity(spec2_free, fit2_free)

        sa_from_free = FluorescenceCorrelationFitting.surface_density(spec2_free, fit2_free; w0 = w0)
        @test sa_from_free ≈ sa_expected rtol=1e-6
        
        diff_from_free = FluorescenceCorrelationFitting.diffusivity(spec2_free, fit2_free; w0 = w0)
        @test diff_from_free ≈ w0^2 / (4τD_true)

        τD_from_2free = FluorescenceCorrelationFitting.τD(spec2_free, fit2_free)
        @test τD_from_2free ≈ τD_true rtol=1e-10

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, 2 diffusers, GLOBAL anomalous exponent
        # params = [g0, κ, w01, w02, α, w1]
        # ------------------------------------------------------------
        w01 = 300e-9
        w02 = 700e-9
        α = rand()
        w1 = 0.5*rand()
        spec3_glob = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d3,
            anom = FluorescenceCorrelationFitting.globe,
            offset = 0.0,
            diffusivity = D,
            n_diff = 2,
        )
       
        p0_glob = [0.5, 5, w01, w02, 0.5, 0.25]
        model3_glob = FluorescenceCorrelationFitting.FCSModel(spec3_glob, τ, p0_glob)
        y3_glob_true = model3_glob(τ, [g0, κ, w01, w02, α, w1])
        
        lower_glob = [0, 0, 250e-9, 650e-9, 0, 0]
        upper_glob = [1, 10, 350e-9, 750e-9, 1, 0.5]
        fit3_glob = FluorescenceCorrelationFitting.fcs_fit(spec3_glob, τ, y3_glob_true, p0_glob; lower=lower_glob, upper=upper_glob)

        # τD for first and second diffuser should match their w0 slots
        τD1_exp = FluorescenceCorrelationFitting.τD(D, w01)
        τD2_exp = FluorescenceCorrelationFitting.τD(D, w02)
        @test FluorescenceCorrelationFitting.τD(spec3_glob, fit3_glob; nd=1) ≈ τD1_exp rtol=1e-6
        @test FluorescenceCorrelationFitting.τD(spec3_glob, fit3_glob; nd=2) ≈ τD2_exp rtol=1e-6
        @test_throws ArgumentError FluorescenceCorrelationFitting.τD(spec3_glob, fit3_glob; nd=3)

        # concentration for nd=2 should also work
        c2_exp = FluorescenceCorrelationFitting.concentration(g0, κ, w02)
        @test FluorescenceCorrelationFitting.concentration(spec3_glob, fit3_glob; nd=2) ≈ c2_exp rtol=1e-6

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, 2 diffusers, PER-POP anomalous exponents
        # params = [g0, κ, w01, w02, α1, α2, w1]
        # ------------------------------------------------------------
        α1 = rand()
        α2 = rand()+1
        spec3_perpop = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d3,
            anom = FluorescenceCorrelationFitting.perpop,
            offset = 0.0,
            diffusivity = D,
            n_diff = 2,
        )
        
        p0_perpop = [0.55, 5, w01, w02, 0.5, 1.5, 0.25]
        τ = 10 .^ range(-7, 0; length=400)
        model3_perpop = FluorescenceCorrelationFitting.FCSModel(spec3_perpop, τ, p0_perpop)
        y3_perpop_true = model3_perpop(τ, [g0, κ, w01, w02, α1, α2, w1])
        
        lower_perpop = [0, 0, 250e-9, 650e-9, 0, 1, 0]
        upper_perpop = [1, 10, 350e-9, 750e-9, 1, 2, 1]
        fit3_perpop = FluorescenceCorrelationFitting.fcs_fit(spec3_perpop, τ, y3_perpop_true, p0_perpop; lower=lower_perpop, upper=upper_perpop)
        
        @test FluorescenceCorrelationFitting.τD(spec3_perpop, fit3_perpop; nd=1) ≈ τD1_exp rtol=1e-6
        @test FluorescenceCorrelationFitting.τD(spec3_perpop, fit3_perpop; nd=2) ≈ τD2_exp rtol=1e-6

        # concentration still well-defined
        @test FluorescenceCorrelationFitting.concentration(spec3_perpop, fit3_perpop; nd=1) ≈ FluorescenceCorrelationFitting.concentration(g0, κ, w01) rtol=1e-6

        # ------------------------------------------------------------
        # 2D, fixed diffusivity, anomalous (global)
        # just make sure surface_density still works
        # ------------------------------------------------------------
        spec2_anom = FluorescenceCorrelationFitting.FCSModelSpec(;
            dim = FluorescenceCorrelationFitting.d2,
            anom = FluorescenceCorrelationFitting.globe,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        p0_2anom = [0.4, w0*1.1, 0.9]
        model2_anom = FluorescenceCorrelationFitting.FCSModel(spec2_anom, τ, p0_2anom)
        y2_anom_true = model2_anom(τ, [g0, w0, 0.85])  # [g0, w0, α]
        
        fit2_anom = FluorescenceCorrelationFitting.fcs_fit(spec2_anom, τ, y2_anom_true, p0_2anom)

        @test FluorescenceCorrelationFitting.surface_density(spec2_anom, fit2_anom) ≈ sa_expected rtol=1e-6
    end
end