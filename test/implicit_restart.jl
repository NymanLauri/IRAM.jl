using Test

using IRAM: implicit_restart!, initialize, iterate_arnoldi!

@testset "Implicit restart" begin

    for T in (Float64, ComplexF64)

        n = 20
        A = sprand(T, n, n, 5 / n) + I
        min, max = 5, 8
        h = Vector{T}(undef, max)

        arnoldi = initialize(T, n, max)
        V, H = arnoldi.V, arnoldi.H
        iterate_arnoldi!(A, arnoldi, 1 : max, h)
        λs = sort!(eigvals(view(H, 1:max, 1:max)), by = abs, rev = true)

        m = implicit_restart!(arnoldi, λs, min, max)

        @test norm(V[:, 1 : m]' * V[:, 1 : m] - I) < 1e-13
        @test norm(V[:, 1 : m]' * A * V[:, 1 : m] - H[1 : m, 1 : m]) < 1e-13
        @test norm(A * V[:, 1 : m] - V[:, 1 : m + 1] * H[1 : m + 1, 1 : m]) < 1e-13
    end
end
