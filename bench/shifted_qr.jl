using BenchmarkTools
using IRAM: shifted_qr_step!, ListOfRotations, qr!

function benchqr()
    n = 30
    H = triu(rand(ComplexF64, n+1,n), -1)
    λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs, rev = true)
    rotations = ListOfRotations(eltype(H), n-1)
    @benchmark shifted_qr_step!(A, μ, $rotations) setup = (A = copy($H); μ = $λs[$n])
end