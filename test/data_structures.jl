@testset "data_structures" begin
    # FCSChannel / FCSData
    τ = collect(range(1e-6, 1e-3; length=10))
    G = 0.02 .+ 1.0 ./ (1 .+ τ ./ 1e-4)
    σ = fill(1e-3, length(G))

    ch = FCSChannel("G[1]", τ, G, σ)
    @test ch.name == "G[1]"
    @test ch.τ === τ
    @test ch.G === G
    @test ch.σ === σ

    data = FCSData([ch], Dict("sample" => "test", "note" => 123), "in-memory")
    @test length(data.channels) == 1
    @test data.metadata["sample"] == "test"
    @test data.source == "in-memory"
end