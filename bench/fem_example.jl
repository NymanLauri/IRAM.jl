using BenchmarkTools
using SparseArrays

using IRAM: Givens, Hessenberg, ListOfRotations, qr!, restarted_arnoldi, SR, LR, SM

# mymatrix(n) = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.001, n-1))

function bencharnoldi()
    m = readdlm("C:\\Users\\Lauri\\Documents\\Aq.txt")
    for i = 1 : size(m,1)
        if typeof(m[i,3]) == SubString{String}
            m[i,3] = parse(ComplexF64, m[i,3])
        else
            m[i,3] = convert(ComplexF64, m[i,3])
        end
    #     if typeof(m[i,3]) == SubString{String}
    #         @show m[i,3]
    #         m[i,3] = parse(Complex128, m[i,3])
    #     else
    #         m[i,3] = convert(Complex128, m[i,3])
    #     end
    end
    # A = sparse(m[:,1], m[:,2], convert(Array{Complex128,1},m[:,3]))

    A = sparse(m[:,1], m[:,2], convert(Array{ComplexF64,1},m[:,3]))
    max_restarts = 50
    vecs = 10
    target = SM()
    # A = Matrix(sprand(100, 100, 5 / 100))
    # A = mymatrix(100)
    @benchmark restarted_arnoldi(B, nev+1, 2*nev+2, nev, 1e-10, 1000, tar) setup = (B = copy($A); nev = $vecs; tar = $target)
    # a = restarted_arnoldi(A, vecs+1, 2*vecs+2, vecs, 1e-10, 1000, target)
    # @show norm(A*a.Q*a.R[:,1] - a.R[1,1]*a.Q*a.R[:,1])

    # @benchmark eigs(B, nev=10, which=:SR) setup = (B = copy($A))
    # vals, vecs = eigs(A, nev=10, which=:SR)
    # @show norm(A*vecs[:,1] - vals[1]*vecs[:,1])
end