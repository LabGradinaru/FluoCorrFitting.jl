@testset "extensions" begin
    # Error shims when optional extensions are not loaded
    τp = 1e-6:1e-6:1e-4
    Gp = 0.0 .+ 1.0 ./ (1 .+ τp ./ 1e-4)
    chp = FCSChannel("G[1]", collect(τp), Gp, nothing)
    spec_for_plot = FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)

    @test_throws ErrorException FCSFitting.fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting._fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting.resid_acf_plot([0.1, -0.1, 0.0])
    @test_throws ErrorException FCSFitting.fcs_table(spec_for_plot, nothing, nothing)
    @test_throws ErrorException FCSFitting.read_fcs("somefile.txt")
end