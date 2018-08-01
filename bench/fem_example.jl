# mymatrix(n) = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.001, n-1))
# A = mymatrix(100)

using BenchmarkTools
# using SparseArrays

# using IRAM: Givens, Hessenberg, ListOfRotations, qr!, restarted_arnoldi, SR, LR, SM

function bencharnoldi()
    real_mat = readdlm("C:\\Users\\Lauri\\Documents\\Aq_real.txt")
    imag_mat = readdlm("C:\\Users\\Lauri\\Documents\\Aq_imag.txt")

    A_real = sparse(real_mat[:,1], real_mat[:,2], real_mat[:,3])
    A_imag = sparse(imag_mat[:,1], imag_mat[:,2], imag_mat[:,3])

    A = A_real + im*A_imag

    max_restarts = 50
    vecs = 10
    # target = SM()
    # @benchmark restarted_arnoldi(B, nev+1, 2*nev+2, nev, 1e-10, 1000, tar) setup = (B = copy($A); nev = $vecs; tar = $target)
    # a = restarted_arnoldi(A, vecs+1, 2*vecs+2, vecs, 1e-10, 1000, target)
    # @show norm(A*a.Q*a.R[:,1] - a.R[1,1]*a.Q*a.R[:,1])

    # @benchmark eigs(B, nev=10, which=:SM) setup = (B = copy($A))
    # eigs(A, nev=10, which=:SM)
    vals, vecs = eigs(A, nev=10, which=:SM)
    @show norm(A*vecs[:,1] - vals[1]*vecs[:,1])
end