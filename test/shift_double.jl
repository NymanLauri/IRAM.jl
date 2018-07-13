using Test

using IRAM: double_shift!

function generate_real_H_with_imaginary_eigs(n, T::Type = Float64)
    while true
        H = triu(rand(T, n + 1, n), -1)
        λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs)

        for i = 1 : n
            μ = λs[i]
            if imag(μ) != 0
                # Remove conjugate pair
                deleteat!(λs, (i, i + 1))
                return H, λs, μ
            end
        end
    end
end

@testset "Double Shifted QR" begin
    n = 20

    is_hessenberg(H) = norm(tril(H, -2)) == 0

    # Test on a couple random matrices
    for i = 1 : 50
        H, λs, μ = generate_real_H_with_imaginary_eigs(n, Float64)
        Q = Matrix{Float64}(I, n+1, n+1)
        H_copy = copy(H)
        double_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-2, 1:n-2)), by = abs)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)

        # Test whether relation " H_prev * Q = Q * H_next " holds
        @test norm(H_copy * Q[1:n,1:n-2] - Q[1:n+1,1:n-1] * H[1:n-1,1:n-2]) < 1e-6
        @test norm(Q[1:n,1:n-2]' * H_copy[1:n,1:n] * Q[1:n,1:n-2] - H[1:n-2,1:n-2]) < 1e-6
    end
end
