@testset "modelling" begin
    @testset "Low-level helpers" begin
        # _ndyn_from_len
        @test FCSFitting._ndyn_from_len(0) == 0
        @test FCSFitting._ndyn_from_len(2) == 1
        @test FCSFitting._ndyn_from_len(10) == 5
        @test_throws ArgumentError FCSFitting._ndyn_from_len(-2)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(1)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(3)

        # dynamics_factor! (in-place)
        N = 50
        τ = 1e-3
        K = rand()
        τs = [1e-4, 1e-5]
        Ks = rand(2)
        t = 10.0 .^ (range(-6, -3, length=N))

        out  = similar(t, Float64)
        work = similar(t, Float64)

        # empty dynamics => ones
        FCSFitting.dynamics_factor!(out, work, t, Float64[], Float64[], Int[])
        @test out == ones(N)

        # single component
        expected_vec = @. 1 + K * (exp(-t/τ) - 1)
        FCSFitting.dynamics_factor!(out, work, t, [τ], [K], [1])
        @test out ≈ expected_vec rtol=1e-12

        # two independent blocks: [1,1] => product
        prod_vec = (@. 1 + Ks[1]*(exp(-t/τs[1]) - 1)) .* (@. 1 + Ks[2]*(exp(-t/τs[2]) - 1))
        FCSFitting.dynamics_factor!(out, work, t, τs, Ks, [1,1])
        @test out ≈ prod_vec rtol=1e-12

        # single block with 2 comps: [2] => (1 + sum)
        blk_vec = @. 1 + Ks[1]*(exp(-t/τs[1]) - 1) + Ks[2]*(exp(-t/τs[2]) - 1)
        FCSFitting.dynamics_factor!(out, work, t, τs, Ks, [2])
        @test out ≈ blk_vec rtol=1e-12

        # error paths
        @test_throws BoundsError FCSFitting.dynamics_factor!(out, work, t, [1e-4], [0.1,0.2], [2]) # too many weights
        @test_throws BoundsError FCSFitting.dynamics_factor!(out, work, t, [1e-4,1e-5], [0.1], [2]) # too many lifetimes
        @test_throws BoundsError FCSFitting.dynamics_factor!(out, work, t, [1e-4], [0.2], [2]) # too many ics

        # diff_factor! (in-place)
        outd = similar(t, Float64)

        # 2D Brownian, one component
        τDs1 = [1e-3]; wts1 = Float64[]
        spec2d = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)
        FCSFitting.diff_factor!(outd, t, nothing, τDs1, Float64[], wts1)
        @test outd == FCSFitting.udc_2d(t, τDs1[1])

        # 2D Brownian mixture
        τDs2 = [1e-4, 1e-3]; wts2 = [0.3]
        mix2d = @. 0.3*FCSFitting.udc_2d(t, τDs2[1]) + 0.7*FCSFitting.udc_2d(t, τDs2[2])
        FCSFitting.diff_factor!(outd, t, nothing, τDs2, Float64[], wts2)
        @test outd ≈ mix2d rtol=1e-12

        # 2D anomalous (per-pop α)
        αs3  = [0.7, 0.9, 1.0]
        τDs3 = [1e-5, 1e-4, 1e-3]; wts3 = [0.2, 0.3]
        spec2d_per = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.perpop, n_diff=3)
        mix2d_anom = 0.2 .* FCSFitting.udc_2d(t, τDs3[1], αs3[1]) .+
                     0.3 .* FCSFitting.udc_2d(t, τDs3[2], αs3[2]) .+
                     0.5 .* FCSFitting.udc_2d(t, τDs3[3], αs3[3])
        FCSFitting.diff_factor!(outd, t, nothing, τDs3, αs3, wts3)
        @test outd ≈ mix2d_anom rtol=1e-12

        # 3D Brownian mixture
        κ3d = 10*rand()
        spec3d = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none, n_diff=2)
        mix3d = @. 0.3*FCSFitting.udc_3d(t, τDs2[1], κ3d) + 0.7*FCSFitting.udc_3d(t, τDs2[2], κ3d)
        FCSFitting.diff_factor!(outd, t, κ3d, τDs2, Float64[], wts2)
        @test outd ≈ mix3d rtol=1e-12

        # 3D anomalous mixture
        spec3d_per = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.perpop, n_diff=3)
        mix3d_anom = 0.2 .* FCSFitting.udc_3d(t, τDs3[1], κ3d, αs3[1]) .+
                     0.3 .* FCSFitting.udc_3d(t, τDs3[2], κ3d, αs3[2]) .+
                     0.5 .* FCSFitting.udc_3d(t, τDs3[3], κ3d, αs3[3])
        FCSFitting.diff_factor!(outd, t, κ3d, τDs3, αs3, wts3)
        @test outd ≈ mix3d_anom rtol=1e-12

        # error paths
        @test_throws BoundsError FCSFitting.diff_factor!(outd, t, nothing, Float64[], Float64[], wts2)
        @test_throws BoundsError FCSFitting.diff_factor!(outd, t, nothing, τDs2, [0.8], [0.2,0.3]) # α length mismatch
        @test_throws BoundsError FCSFitting.diff_factor!(outd, t, nothing, τDs2, Float64[], Float64[]) # weights length mismatch
    end

    @testset "Diffusion kernels" begin
        τD = 1e-3
        α  = 0.5 + 1.5*rand()
        κ  = 10 * rand()
        t  = rand()
        ts = rand(100)

        # 2D Brownian
        @test FCSFitting.udc_2d(t, τD) ≈ inv(1 + t/τD)
        @test all(isreal, FCSFitting.udc_2d(ts, τD))

        # 2D anomalous
        @test FCSFitting.udc_2d(t, τD, α) ≈ inv(1 + (t/τD)^α)
        @test all(isreal, FCSFitting.udc_2d(ts, τD, α))

        # 3D Brownian
        @test FCSFitting.udc_3d(t, τD, κ) ≈ inv((1 + t/τD) * sqrt(1 + t/(κ^2 * τD)))
        @test all(isreal, FCSFitting.udc_3d(ts, τD, κ))

        # 3D anomalous
        @test FCSFitting.udc_3d(t, τD, κ, α) ≈ inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))
        @test all(isreal, FCSFitting.udc_3d(ts, τD, κ, α))
    end

    @testset "FCSModel end-to-end" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = 0.8 + 0.4*rand()
        off = 1e-2 * (randn() - 0.5)

        # 2D Brownian, free offset
        τD = 1e-3
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)
        p = [g0, off, τD]
        model = FCSFitting.FCSModel(spec, t, p)
        @test model(t, p) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD)

        # 2D Brownian, fixed offset in spec (omit offset from p)
        spec_fixoff = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=off)
        p_fixoff = [g0, τD]
        model_fixoff = FCSFitting.FCSModel(spec_fixoff, t, p_fixoff)
        @test model_fixoff(t, p_fixoff) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD)

        # 2D with dynamics (two independent components)
        τs = [1e-5, 1e-4]; Ks = [0.1, 0.2]; ics = [1,1]
        spec_dyn = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, ics=ics)
        p_dyn = vcat(g0, off, τD, τs, Ks)
        model_dyn = FCSFitting.FCSModel(spec_dyn, t, p_dyn)
        dynfac = FCSFitting.dynamics_factor(t, τs, Ks, ics)
        @test model_dyn(t, p_dyn) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD) * dynfac

        # 2D anomalous, global α
        α = 0.8
        spec_ag = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.globe, n_diff=1)
        p_ag = [g0, off, τD, α]
        model_ag = FCSFitting.FCSModel(spec_ag, t, p_ag)
        @test model_ag(t, p_ag) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD, α)

        # 2D anomalous, per-population α, mixture of 2 diffusers
        τDs2 = [1e-4, 2e-3]; αs2 = [0.7, 1.0]; wts = [0.3]
        spec_per = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.perpop, n_diff=2)
        p_per = vcat(g0, off, τDs2, αs2, wts)
        model_per = FCSFitting.FCSModel(spec_per, t, p_per)
        mix = @. 0.3*FCSFitting.udc_2d(t, τDs2[1], αs2[1]) + 0.7*FCSFitting.udc_2d(t, τDs2[2], αs2[2])
        @test model_per(t, p_per) ≈ @. off + g0 * mix

        # 3D Brownian, free offset
        κ = 10 * rand()
        τD3 = 1e-3
        spec3d = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none, n_diff=1)
        p3d = [g0, off, κ, τD3]
        model3d = FCSFitting.FCSModel(spec3d, t, p3d)
        @test model3d(t, p3d) ≈ @. off + g0 * FCSFitting.udc_3d(t, τD3, κ)

        # 3D anomalous, global α
        αg = 0.9
        spec3d_ag = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.globe, n_diff=1)
        p3d_ag = [g0, off, κ, τD3, αg]
        model3d_ag = FCSFitting.FCSModel(spec3d_ag, t, p3d_ag)
        @test model3d_ag(t, p3d_ag) ≈ @. off + g0 * FCSFitting.udc_3d(t, τD3, κ, αg)

        # 3D anomalous, per-population α, 2 diffusers + dynamics
        τDs3 = [5e-4, 2e-3]; αs3 = [0.8, 1.0]; wts3 = [0.25]
        τdyn = [2e-5]; Kdyn = [0.15]; ics = [1]
        spec3d_per = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.perpop, n_diff=2, ics=ics)
        p3d_per = vcat(g0, off, κ, τDs3, αs3, wts3, τdyn, Kdyn)
        model3d_per = FCSFitting.FCSModel(spec3d_per, t, p3d_per)
        dynfac3 = FCSFitting.dynamics_factor(t, τdyn, Kdyn, ics)
        mix3 = @. 0.25*FCSFitting.udc_3d(t, τDs3[1], κ, αs3[1]) + 0.75*FCSFitting.udc_3d(t, τDs3[2], κ, αs3[2])
        @test model3d_per(t, p3d_per) ≈ @. off + g0 * mix3 * dynfac3
    end
end
