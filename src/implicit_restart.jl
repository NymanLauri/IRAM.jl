using LinearAlgebra: givensAlgorithm

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(A::AbstractMatrix, arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(undef, size(arnoldi.V,1),min)) where {T<:Real}
    # Real arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max+1, max+1)
    # Q_test = Matrix{Complex128}(I, max, max)
    V_test = copy(V)

    m = max

    while m > min
        μ = λs[m-active+1]
        if imag(μ) == 0
            single_shift!(V_test, H, active, m, max, real(μ), Q)
            m -= 1
        else
            # Dont double shift past min
            # m == min + 1 && break
            println("double shift")

            V_temp = copy(V)
            Vn = Matrix{Float64}(size(V,1), min-active)
            mul!(Vn, view(V_temp, :, active:max), view(Q, active:max, active:min-1))
            copyto!(view(V_temp, :, active:min-1), Vn)
            # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
            @show norm(V_temp[:, 1 : min-1]' * A * V_temp[:, 1 : min-1] - H[1 : min-1, 1 : min-1])

            double_shift!(V_test, A, V, H, active, m, max, min, μ, Q)

            V_temp = copy(V)
            Vn = Matrix{Float64}(size(V,1), min-active)
            mul!(Vn, view(V_temp, :, active:max), view(Q, active:max, active:min-1))
            copyto!(view(V_temp, :, active:min-1), Vn)
            @show norm(V_temp[:, 1 : min-1]' * A * V_temp[:, 1 : min-1] - H[1 : min-1, 1 : min-1])

            m -= 2 # incorrect
        end
        # V_temp = copy(V)
        # Vn = Matrix{Float64}(size(V,1), min-active)
        # mul!(Vn, view(V_temp, :, active:max), view(Q, active:max, active:min-1))
        # copyto!(view(V_temp, :, active:min-1), Vn)
        # # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
        # @show μ
        # @show norm(V_temp[:, 1 : min-1]' * A * V_temp[:, 1 : min-1] - H[1 : min-1, 1 : min-1])
        # a = sort!(eigvals(H[active:m,active:m]), by=abs)
        # # @show a[1:m-min+1]
        # @show norm(V' * V - I)
    end

    # Update & copy the Krylov basis
    mul!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copyto!(view(V, :, active:m), view(V_new, :, 1:m-active+1))
    copyto!(view(V, :, m+1), view(V, :, max+1))

    return m
end

function implicit_restart!(A::AbstractMatrix, arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(undef, size(arnoldi.V,1),min)) where {T}
    # Complex arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max+1, max+1)

    m = max

    while m > min
        μ = λs[m - active + 1]
        single_shift!(H, active, m, μ, Q)
        m -= 1
    end

    # Update & copy the Krylov basis
    mul!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copyto!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copyto!(view(V, :, m + 1), view(V, :, max + 1))

    return m
end

# """
# Assume a Hessenberg matrix of size (n + 1) × n.
# """
function single_shift!(V_test::AbstractMatrix, H_whole::AbstractMatrix{Tv}, min, max, realmax, μ::Tv, Q::AbstractMatrix) where {Tv}
    # println("Single:")
    H = view(H_whole, min : max + 1, min : max)
    n = size(H, 2)
    Q_test = Matrix{Complex128}(I, realmax, realmax)

    # Construct the first givens rotation that maps (H - μI)e₁ to a multiple of e₁
    @inbounds c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c, s, min)

    lmul!(givens, H_whole)
    rmul!(H_whole, givens)

    # Update Q
    rmul!(Q, givens)
    rmul!(Q_test, givens)

    # Chase the bulge!
    @inbounds for i = 2 : n - 1
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c, s, min + i - 1)
        lmul!(givens, H_whole)
        H_whole[i+1,i-1] = zero(Tv)
        rmul!(view(H_whole, 1 : min + i + 1, :), givens)
        
        # Update Q
        rmul!(Q, givens)
        rmul!(Q_test, givens)
    end

    # Do the last Given's rotation by hand (assuming exact shifts!)
    @inbounds H[n, n - 1] = H[n + 1, n - 1]
    @inbounds H[n + 1, n - 1] = zero(Tv)

    # Update Q with the last rotation
    Q[1:max+1, max] .= 0
    Q[max+1,max] = 1

    tmp = copy(view(V_test, :, max + 1))
    A_mul_B!(view(V_test, :, 1:max-1), copy(V_test[:, 1 : realmax]), Q_test[1:realmax, 1 : max-1])
    copyto!(view(V_test, :, max), tmp)

    return H
end

function double_shift!(V_test::AbstractMatrix, A::AbstractMatrix{Tv}, V::AbstractMatrix{Tv}, H_whole::AbstractMatrix{Tv}, min, max, realmax, minim, μ::Complex, Q::AbstractMatrix) where {Tv<:Real}
    H = view(H_whole, min : max + 1, min : max)
    H_copy = copy(H_whole)
    # Q_copy = copy(Q)
    Q_test = Matrix{Complex128}(I, realmax, realmax)
    n = size(H, 2)
    @show n
    # minim = realmax-10
    V_temp = copy(V)
    Vn = Matrix{Float64}(size(V,1), minim-min)
    mul!(Vn, view(V_temp, :, min:max), view(Q, min:max, min:minim-1))
    copyto!(view(V_temp, :, min:minim-1), Vn)
    # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
    # @show μ
    @show norm(V_temp[:, 1 : minim-1]' * A * V_temp[:, 1 : minim-1] - H_whole[1 : minim-1, 1 : minim-1])

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    @inbounds p₁ = abs2(μ) - 2 * real(μ) * H[1,1] + H[1,1] * H[1,1] + H[1,2] * H[2,1]
    @inbounds p₂ = -2.0 * real(μ) * H[2,1] + H[2,1] * H[1,1] + H[2,2] * H[2,1]
    @inbounds p₃ = H[3,2] * H[2,1]

    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, min+1)
    G₂ = Givens(c₂, s₂, min)

    lmul!(G₁, H_whole)
    lmul!(G₂, H_whole)
    rmul!(H_whole, G₁)
    rmul!(H_whole, G₂)

    # Update Q
    rmul!(Q, G₁)
    rmul!(Q, G₂)

    rmul!(Q_test, G₁)
    rmul!(Q_test, G₂)

    # minim = realmax-10
    V_temp = copy(V)
    Vn = Matrix{Float64}(size(V,1), minim-min)
    mul!(Vn, view(V_temp, :, min:max), view(Q, min:max, min:minim-1))
    copyto!(view(V_temp, :, min:minim-1), Vn)
    # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
    # @show μ
    @show norm(V_temp[:, 1 : minim-1]' * A * V_temp[:, 1 : minim-1] - H_whole[1 : minim-1, 1 : minim-1])

    # Bulge chasing!
    @inbounds for i = 2 : n - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, min+i)
        G₂ = Givens(c₂, s₂, min+i-1)

        # Restore to Hessenberg
        lmul!(G₁, view(H_whole, :, min+i-2:max))
        lmul!(G₂, view(H_whole, :, min+i-2:max))
        
        # Zero out off-diagonal values
        H_whole[min+i, i-1] = zero(Tv)
        H_whole[min+i+1, i-1] = zero(Tv)

        # Create a new bulge
        rmul!(view(H_whole, 1:min+i+2, :), G₁)
        rmul!(view(H_whole, 1:min+i+2, :), G₂)

        # Update Q
        rmul!(Q, G₁)
        rmul!(Q, G₂)

        rmul!(Q_test, G₁)
        rmul!(Q_test, G₂)

        if n == 33
        # realmax-10 happens to be the size after the implicit restart
        # minim = realmax-10
        V_temp = copy(V)
        Vn = Matrix{Float64}(size(V,1), minim-min)
        mul!(Vn, view(V_temp, :, min:max), view(Q, min:max, min:minim-1))
        copyto!(view(V_temp, :, min:minim-1), Vn)
        # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
        # @show μ
        # if norm(V_temp[:, 1 : minim-1]' * A * V_temp[:, 1 : minim-1] - H_whole[1 : minim-1, 1 : minim-1]) > 1e-6 && i < n-2
            @show norm(V_temp[:, 1 : minim-1]' * A * V_temp[:, 1 : minim-1] - H_whole[1 : minim-1, 1 : minim-1])
            # display()
        # end
        end
    end
    
    # Qv = Q_copy[1:max,1:max]' * Q[1:max,1:max]
    # @show norm(H_copy[1:max+1,1:max] * Q[1:max,1:max-2] - Q[1:max+1,1:max-1] * H_whole[1:max-1,1:max-2]) < 1e-6
    # @show norm(Qv[1:max,1:max-2]' * H_copy[1:max,1:max] * Qv[1:max,1:max-2] - H_whole[1:max-2,1:max-2])
    # @show norm(Q_test[1:max,1:max-2]' * H_copy[1:max,1:max] * Q_test[1:max,1:max-2] - H_whole[1:max-2,1:max-2])

    @inbounds if n > 2
        # Do the last Given's rotation by hand.
        H[n - 1, n - 2] = H[n + 1, n - 2]

        # Zero out the off-diagonal guys
        H[n    , n - 2] = zero(Tv)
        H[n + 1, n - 2] = zero(Tv)
        
        # Update Q with the last rotation
        Q[1:max+1, max-1:max] .= 0
        Q[max+1, max-1] = 1
    end

    @inbounds H[n + 1, n - 1] = zero(Tv)
    
    # V_temp = copy(V)
    # Vn = Matrix{Float64}(size(V,1), max-min+1)
    # mul!(Vn, view(V_temp, :, min:max), view(Q_test, min:max, min:max-1))
    # copyto!(view(V_temp, :, min:max-1), Vn)
    # @show norm(V_temp[:, 1 : max-2]' * A * V_temp[:, 1 : max-2] - H_whole[1 : max-2, 1 : max-2])


    # @show norm(V_temp[:, 1 : minim-2]' * A * V_temp[:, 1 : minim-2] - H_copy[1 : minim-2, 1 : minim-2])

    # tmp = copy(view(V_test, :, max + 1))
    # A_mul_B!(view(V_test, :, 1:max-2), copy(V_test[:, 1 : realmax]), Q_test[1:realmax, 1 : max-2])
    # copyto!(view(V_test, :, max-1), tmp)
    # @show norm(V_test[:, 1 : max-2]' * A * V_test[:, 1 : max-2] - H_copy[1 : max-2, 1 : max-2])
    # @show norm(Q_test[1:max,1:max-2]' * H_copy[1:max,1:max] * Q_test[1:max,1:max-2] - H_whole[1:max-2,1:max-2])

    # V_temp = copy(V)
    # Vn = Matrix{Float64}(size(V,1), minim-min)
    # mul!(Vn, view(V_temp, :, min:max), view(Q, min:max, min:minim-1))
    # copyto!(view(V_temp, :, min:minim-1), Vn)
    # # copyto!(view(V_temp, :, m+1), view(V_temp, :, max+1))
    # # @show μ
    # @show norm(V_temp[:, 1 : minim-1]' * A * V_temp[:, 1 : minim-1] - H_whole[1 : minim-1, 1 : minim-1])

    H
end
