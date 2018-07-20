using Printf
using LinearAlgebra: givensAlgorithm
import Base: @propagate_inbounds

@propagate_inbounds is_offdiagonal_small(H, i, tol) = abs(H[i+1,i]) < tol*(abs(H[i,i]) + abs(H[i+1,i+1]))

"""
Computes the eigenvalues of the matrix A. Assumes that A is in Schur form.
"""
function eigvalues(A::AbstractMatrix{T}; tol = eps(real(T))) where {T}
    n = size(A, 1)
    λs = Vector{complex(T)}(undef, n)
    i = 1

    @inbounds while i < n
        if is_offdiagonal_small(A, i, tol)
            λs[i] = A[i, i]
            i += 1
        else
            d = A[i,i] * A[i+1,i+1] - A[i,i+1] * A[i+1,i]
            x = 0.5*(A[i,i] + A[i+1,i+1])
            y = sqrt(complex(x*x - d))
            λs[i] = x + y
            λs[i + 1] = x - y
            i += 2
        end
    end

    if i == n 
        @inbounds λs[i] = A[n, n] 
    end

    return λs
end

local_schurfact!(A, Q; kwargs...) = local_schurfact!(A, Q, 1, size(A, 1); kwargs...)

function local_schurfact!(H::AbstractMatrix{T}, Q::AbstractMatrix{T}, start, stop; tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T<:Real}
    to = stop

    # iteration count
    iter = 0

    @inbounds while true
        iter += 1

        iter > maxiter && return false

        # Indexing
        # `to` points to the column where the off-diagonal value was last zero.
        # while `from` points to the smallest index such that there is no small off-diagonal
        # value in columns from:end-1. Sometimes `from` is just 1. Cartoon of a split 
        # with from != 1:
        # 
        #  + + | | | | + +
        #  + + | | | | + +
        #    o X X X X = =
        #      X X X X = =
        #      . X X X = =
        #      .   X X = =
        #      .     o + +
        #      .     . + +
        #      ^     ^
        #   from   to
        # The X's form the unreduced Hessenberg matrix we are applying QR iterations to,
        # the | and = values get updated by Given's rotations, while the + values remain
        # untouched! The o's are zeros -- or numerically considered zeros.

        # We keep `from` one column past the zero off-diagonal value, so we check whether
        # the `from - 1` column has a small off-diagonal value.
        from = to
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            to -= 1
        else
            # Now we are sure we can work with a 2x2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁, H₁₂ = H[to-1,to-1], H[to-1,to]
            H₂₁, H₂₂ = H[to  ,to-1], H[to  ,to]

            # Matrix determinant and trace
            d = H₁₁ * H₂₂ - H₂₁ * H₁₂
            t = H₁₁ + H₂₂
            discriminant = t * t - 4d

            if discriminant ≥ zero(T)
                # Real eigenvalues.
                # Note that if from + 1 == to in this case, then just one additional
                # iteration is necessary, since the Wilkinson shift will do an exact shift.

                # Determine the Wilkinson shift -- the closest eigenvalue of the 2x2 block
                # near H[to,to]
                sqr = sqrt(discriminant)
                λ₁ = (t + sqr) / 2
                λ₂ = (t - sqr) / 2
                λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
                # Run a bulge chase
                single_shift_schur!(H, Q, λ, from, to)
            else
                # Conjugate pair
                if from + 1 == to
                    # A conjugate pair has converged apparently!
                    to -= 2
                else
                    # Otherwise we do a double shift!
                    sqr = sqrt(complex(discriminant))
                    λ = (t + sqr) / 2
                    double_shift_schur!(H, from, to, λ, Q)
                end
            end
        end

        # Converged!
        to ≤ start && break
    end

    return true
end

function local_schurfact!(H::AbstractMatrix{T}, Q::AbstractMatrix{T}, start, stop; tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T}
    to = stop

    # iteration count
    iter = 0

    @inbounds while true
        iter += 1

        # Don't like that this throws :|
        # iter > maxiter && throw(ArgumentError("iteration limit $maxiter reached"))
        iter > maxiter && return false

        # Indexing
        # `to` points to the column where the off-diagonal value was last zero.
        # while `from` points to the smallest index such that there is no small off-diagonal
        # value in columns from:end-1. Sometimes `from` is just 1. Cartoon of a split 
        # with from != 1:
        # 
        #  + + | | | | + +
        #  + + | | | | + +
        #    o X X X X = =
        #      X X X X = =
        #      . X X X = =
        #      .   X X = =
        #      .     o + +
        #      .     . + +
        #      ^     ^
        #   from   to
        # The X's form the unreduced Hessenberg matrix we are applying QR iterations to,
        # the | and = values get updated by Given's rotations, while the + values remain
        # untouched! The o's are zeros -- or numerically considered zeros.

        # We keep `from` one column past the zero off-diagonal value, so we check whether
        # the `from - 1` column has a small off-diagonal value.
        from = to
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            to -= 1
        else
            # Now we are sure we can work with a 2x2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁, H₁₂ = H[to-1,to-1], H[to-1,to]
            H₂₁, H₂₂ = H[to  ,to-1], H[to  ,to]

            # Matrix determinant and trace
            d = H₁₁ * H₂₂ - H₂₁ * H₁₂
            t = H₁₁ + H₂₂
            discriminant = t * t - 4d

            # Note that if from + 1 == to in this case, then just one additional
            # iteration is necessary, since the Wilkinson shift will do an exact shift.

            # Determine the Wilkinson shift -- the closest eigenvalue of the 2x2 block
            # near H[to,to]
            sqr = sqrt(discriminant)
            λ₁ = (t + sqr) / 2
            λ₂ = (t - sqr) / 2
            λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
            # Run a bulge chase
            single_shift_schur!(H, Q, λ, from, to)
        end

        # Converged!
        to ≤ start && break
    end

    return true
end

function single_shift_schur!(HH::StridedMatrix, Q::AbstractMatrix, shift::Number, istart::Integer, iend::Integer)
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart + 1, istart]
    if m > istart + 1
        Htmp = HH[istart + 2, istart]
        HH[istart + 2, istart] = 0
    end
    c, s = givensAlgorithm(H11 - shift, H21)
    G = Givens(c, s, istart)
    lmul!(G, HH)
    rmul!(HH, G)
    rmul!(Q, G)
    for i = istart:iend - 2
        c, s = givensAlgorithm(HH[i + 1, i], HH[i + 2, i])
        G = Givens(c, s, i + 1)
        lmul!(G, HH)
        HH[i + 2, i] = Htmp
        if i < iend - 2
            Htmp = HH[i + 3, i + 1]
            HH[i + 3, i + 1] = 0
        end
        rmul!(HH, G)
        rmul!(Q, G)
    end
    return HH
end

function double_shift_schur!(H::AbstractMatrix{Tv}, min, max, μ::Complex, Q::AbstractMatrix) where {Tv<:Real}
    # Compute the three nonzero entries of (H - μ₂)(H - μ₁)e₁.
    @inbounds p₁ = abs2(μ) - 2 * real(μ) * H[min,min] + H[min,min] * H[min,min] + H[min,min+1] * H[min+1,min]
    @inbounds p₂ = -2.0 * real(μ) * H[min+1,min] + H[min+1,min] * H[min,min] + H[min+1,min+1] * H[min+1,min]
    @inbounds p₃ = H[min+2,min+1] * H[min+1,min]

    # Map that column to a mulitiple of e₁ via three Given's rotations
    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, min+1)
    G₂ = Givens(c₂, s₂, min)

    # Apply the Given's rotations
    lmul!(G₁, H)
    lmul!(G₂, H)
    rmul!(H, G₁)
    rmul!(H, G₂)

    # Update Q
    rmul!(Q, G₁)
    rmul!(Q, G₂)

    # Bulge chasing. First step of the for-loop below looks like:
    #   min           max
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x + + + x x x
    # i → x x x x x x x     + + + + + + +     x + + + x x x 
    #     x x x x x x x     o + + + + + +       + + + x x x
    #     x x x x x x x  ⇒  o + + + + + +  ⇒   + + + x x x
    #       |   x x x x           x x x x       + + + x x x
    #       |     x x x             x x x             x x x
    #       |       x x               x x               x x
    #       ↑
    #       i
    #
    # Last iterations looks like:
    #   min           max
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #       x x x x x x       x x x x x x       x x x + + +
    #         x x x x x  ⇒    x x x x x x  ⇒     x x + + +
    # i → ----- x x x x           + + + +           x + + +
    #           x x x x           o + + +             + + +
    #           x x x x           o + + +             + + +
    #             ↑
    #             i

    @inbounds for i = min + 1 : max - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, i+1)
        G₂ = Givens(c₂, s₂, i)

        # Restore to Hessenberg
        lmul!(G₁, H)
        lmul!(G₂, H)

        # Introduce zeros below Hessenberg part
        H[i+1,i-1] = zero(Tv)
        H[i+2,i-1] = zero(Tv)

        # Create a new bulge
        rmul!(H, G₁)
        rmul!(H, G₂)

        # Update Q
        rmul!(Q, G₁)
        rmul!(Q, G₂)
    end

    # Last bulge is just one Given's rotation
    #     min           max
    #       ↓           ↓
    # min → x x x x x x x    x x x x x x x    x x x x x + +  
    #       x x x x x x x    x x x x x x x    x x x x x + +  
    #         x x x x x x      x x x x x x      x x x x + +  
    #           x x x x x  ⇒     x x x x x  ⇒     x x x + +  
    #             x x x x          x x x x          x x + +  
    #               x x x            + + +            x + +  
    # max → ------- x x x            o + +              + +


    @inbounds c, s, = givensAlgorithm(H[max-1,max-2], H[max,max-2])
    G = Givens(c, s, max-1)
    lmul!(G, H)
    @inbounds H[max,max-2] = zero(Tv)
    rmul!(H, G)
    rmul!(Q, G)

    H
end

function schur_to_eigvec(R::AbstractMatrix{TR}) where {TR}

    n = size(R,1)
    X = Matrix{TR}(undef,n,n)
    # x = Vector{TR}(n)
    y = Vector{TR}(undef,n)
    for i = n : -1 : 1
        R_copy = copy(R)
        for j = 1:n
            R_copy[j,j] -= R[i,i]
        end
        y[i] = one(TR)
        y[1:i-1] .= -R_copy[1:i-1,i]
        y[i+1:n] .= zero(TR)
        # display(y)
        backward_subst!(view(R_copy,1:i,1:i), y)
        y ./= norm(y)
        X[:,i] .= y
    end
    X

end

function backward_subst!(R::AbstractMatrix, y::AbstractVector)

    # R[1:n-1,1:n-1]*x[1:n-1]=-R[1:n,n]
    n = size(R,1)

    @inbounds for i = n-1 : -1 : 1
        R[i,1:i] /= R[i,i]
        y[i] /= R[i,i]
        @inbounds for k = i-1 : -1 : 1
            R[k,1:i] .-= view(R, i, 1:i) * R[k,i]
            y[k] -= y[i]*R[k,i]
        end
    end
end


# function schur_to_eigen!(Schur::PartialSchur{TQ,TR}) where {TQ,TR}
#     T = Schur.R
#     VR = Schur.Q
#     LDT = size(T,1)
#     LDVR = size(VR, 1)
#     N = LDT # N is the "order of T"
#     WORK = Matrix{Float64}(3*N,3*N)
#     M = N

#     ZERO = zero(Float64)
#     ONE = one(Float64)

#     # Quick return if possible.
#     if N == 0
#         return false
#     end


#     # Set the constants to control overflow.

#     # UNFL = DLAMCH( 'Safe minimum' )
#     UNFL = eps(Float64) #Not sure what to use for safe minimum
#     OVFL = ONE / UNFL

#     # CALL DLABAD( UNFL, OVFL )
#     if log10(OVFL) > 2000 
#         UNFL = sqrt( UNFL )
#         OVFL = sqrt( OVFL )
#     end

#     # ULP = DLAMCH( 'Precision' )
#     ULP = 2*eps(Float64) # Base*eps
#     SMLNUM = UNFL*( N / ULP )
#     BIGNUM = ( ONE-ULP ) / SMLNUM

#     # Compute 1-norm of each column of strictly upper triangular
#     # part of T to control overflow in triangular solver.
#     WORK[1] = ZERO
#     for J = 2 : N
#         WORK[J] = ZERO
#         for I = 1 : J - 1
#             WORK[J] = WORK[J] + abs( T[I, J] )
#         end
#     end

#     # Right eigenvectors
#     IP = 0
#     IS = M
#     for KI = N:-1:1
#         if IP == 1 
#             if IP == 1
#                 IP = 0
#             end
#             if IP == -1
#                 IP = 1
#             end
#             continue
#         end
#         if KI != 1 && T[KI, KI-1] != zero(eltype(TR))
#             IP = -1
#         end

#         # Compute the KI-th eigenvalue (WR,WI).
#         WR = T[KI, KI]
#         WI = ZERO
#         if IP != 0 
#             WI = sqrt( abs( T[KI, KI-1] ) )*sqrt( abs( T[KI, KI-1] ) )
#         end
#         SMIN = max( ULP*( abs( WR )+abs( WI ) ), SMLNUM )

#         if IP == 0

#             # Real right eigenvector
#             WORK[KI+N] = ONE

#             # Form right-hand side
#             for K = 1 : KI - 1
#                 WORK[K+N] = -T[K, KI]
#             end

#             # Solve the upper quasi-triangular system:
#             # (T(1:KI-1,1:KI-1) - WR)*X = SCALE*WORK.
#             JNXT = KI - 1
#             for J = KI - 1: -1 : 1
#                 if J > JNXT
#                     # GO TO 60
#                     break
#                 end
#                 J1 = J
#                 J2 = J
#                 JNXT = J - 1
#                 if J > 1
#                     if !iszero(T[J, J-1])
#                         J1 = J - 1
#                         JNXT = J - 2
#                     end
#                 end

#                 if J1 == J2

#                     #DEFINE ALL VARIABLESs
#                     SCALE = Matrix{Float64}(1,1)
#                     X = Matrix{Float64}(1,1)
#                     XNORM = Matrix{Float64}(1,1)
#                     IERR = Matrix{Float64}(1,1)

#                     # SCALE = 1.0
#                     # X = 1.0
#                     # XNORM = 1.0
#                     # IERR = 1.0

#                     # 1-by-1 diagonal block
#                     DLALN2( false, 1, 1, SMIN, ONE, T[J, J], LDT, ONE, ONE, WORK[J+N], N, WR, ZERO, X, 2, SCALE, XNORM, IERR )

#                     # Scale X(1,1) to avoid overflow when updating
#                     # the right-hand side.
#                     if XNORM > ONE
#                     if WORK[J] > BIGNUM / XNORM
#                         X[1, 1] = X[1, 1] / XNORM
#                         SCALE = SCALE / XNORM
#                     end
#                     end

#                     # Scale if necessary
#                     if SCALE != ONE
#                         DSCAL( KI, SCALE, WORK[1+N], 1 )
#                     end
#                     WORK[J+N] = X[1, 1]

#                     # Update right-hand side
#                     DAXPY( J-1, -X[1, 1], T[1, J], 1, WORK[1+N], 1 )
#                 else

#                     # 2-by-2 diagonal block
#                     DLALN2( false, 2, 1, SMIN, ONE, T[J-1, J-1], LDT, ONE, ONE, WORK[J-1+N], N, WR, ZERO, X, 2, SCALE, XNORM, IERR )

#                     # Scale X(1,1) and X(2,1) to avoid overflow when
#                     # updating the right-hand side.              
#                     if XNORM > ONE
#                         BETA = max( WORK[J-1], WORK[j] )
#                         if BETA > BIGNUM / XNORM
#                             X[1, 1] = X[1, 1] / XNORM
#                             X[2, 1] = X[2, 1] / XNORM
#                             SCALE = SCALE / XNORM
#                         end
#                     end

#                     # Scale if necessary
#                     if SCALE != ONE
#                         DSCAL( KI, SCALE, WORK[1+N], 1 )
#                     end
#                     WORK[J-1+N] = X[1, 1]
#                     WORK[J+N] = X[2, 1]

#                     # Update right-hand side
#                     DAXPY( J-2, -X[1, 1], T[1, J-1], 1, WORK[1+N], 1 )
#                     DAXPY( J-2, -X[2, 1], T[1, J], 1, WORK[1+N], 1 )
#                 end
#             end

#             # Copy the vector x or Q*x to VR and normalize.
#             DCOPY( KI, WORK[1+N], 1, VR[1, IS], 1 )

#             II = IDAMAX( KI, VR[1, IS], 1 )
#             REMAX = ONE / abs( VR[II, IS] )
#             DSCAL( KI, REMAX, VR[1, IS], 1 )

#             for K = KI + 1 : N
#                 VR[K, IS] = ZERO
#             end

#         else

#             # Complex right eigenvector.

#             # Initial solve
#             # [ (T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I* WI)]*X = 0.
#             # [ (T(KI,KI-1)   T(KI,KI)   )               ]

#             if abs( T[KI-1, KI] ) >= abs( T[KI, KI-1] )
#                 WORK[KI-1+N] = ONE
#                 WORK[KI+N2] = WI / T[KI-1, KI]
#             else
#                 WORK[KI-1+N] = -WI / T[KI, KI-1]
#                 WORK[KI+N2] = ONE
#             end
#             WORK[KI+N] = ZERO
#             WORK[KI-1+N2] = ZERO

#             # Form right-hand side
#             for K = 1 : KI - 2
#                 WORK[K+N] = -WORK[KI-1+N]*T[K, KI-1]
#                 WORK[K+N2] = -WORK[KI+N2]*T[K, KI]
#             end

#             # Solve upper quasi-triangular system:
#             # (T(1:KI-2,1:KI-2) - (WR+i*WI))*X = SCALE*(WORK+i*WORK2)
#             JNXT = KI - 2
#             for J = KI - 2:-1:1
#                 if J > JNXT
#                     break
#                 end
#                 J1 = J
#                 J2 = J
#                 JNXT = J             
#                 if J > 1
#                     if T[J, J-1] != ZERO
#                         J1 = J - 1
#                         JNXT = J - 2
#                     end
#                 end

#                 if J1 == J2
#                     # 1-by-1 diagonal block
#                     DLALN2( false, 1, 2, SMIN, ONE, T[J, J], LDT, ONE, ONE, WORK[J+N], N, WR, WI, X, 2, SCALE, XNORM, IERR )

#                     # Scale X(1,1) and X(1,2) to avoid overflow when
#                     # updating the right-hand side
#                     if XNORM > ONE N
#                         if WORK[J] > BIGNUM / XNORM
#                             X[1, 1] = X[1, 1] / XNORM
#                             X[1, 2] = X[1, 2] / XNORM
#                             SCALE = SCALE / XNORM
#                         end
#                     end

#                     # Scale if necessary
#                     if SCALE != ONE
#                         DSCAL( KI, SCALE, WORK[1+N], 1 )
#                         DSCAL( KI, SCALE, WORK[1+N2], 1 )
#                     end
#                     WORK[J+N] = X[1, 1]
#                     WORK[J+N2] = X[1, 2]

#                     # Update the right-hand side
#                     DAXPY( J-1, -X[1, 1], T[1, J], 1, WORK[1+N], 1 )
#                     DAXPY( J-1, -X[1, 2], T[1, J], 1, WORK[1+N2], 1 )
#                 else

#                     # 2-by-2 diagonal block
#                     DLALN2( false, 2, 2, SMIN, ONE, T[J-1, J-1], LDT, ONE, ONE, WORK[J-1+N], N, WR, WI, X, 2, SCALE, XNORM, IERR )

#                     # Scale X to avoid overflow when updating
#                     # the right-hand side               
#                     if XNORM > ONE
#                         BETA = max( WORK[J-1], WORK[j] )
#                         if BETA > BIGNUM / XNORM
#                             REC = ONE / XNORM
#                             X[1, 1] = X[1, 1]*REC
#                             X[1, 2] = X[1, 2]*REC
#                             X[2, 1] = X[2, 1]*REC
#                             X[2, 2] = X[2, 2]*REC
#                             SCALE = SCALE*REC
#                         end
#                     end

#                     # Scale if necessary
#                     if SCALE != ONE
#                         DSCAL( KI, SCALE, WORK[1+N], 1 )
#                         DSCAL( KI, SCALE, WORK[1+N2], 1 )
#                     end
#                     WORK[J-1+N] = X[1, 1]
#                     WORK[J+N] = X[2, 1]
#                     WORK[J-1+N2] = X[1, 2]
#                     WORK[J+N2] = X[2, 2]

#                     # Update the right-hand side
#                     DAXPY( J-2, -X[1, 1], T[1, J-1], 1, WORK[1+N], 1 )
#                     DAXPY( J-2, -X[2, 1], T[1, J], 1, WORK[1+N], 1 )
#                     DAXPY( J-2, -X[1, 2], T[1, J-1], 1, WORK[1+N2], 1 )
#                     DAXPY( J-2, -X[2, 2], T[1, J], 1, WORK[1+N2], 1 )
#                 end
#             end

#             # Copy the vector x or Q*x to VR and normalize.
#             DCOPY( KI, WORK[1+N], 1, VR[1, IS-1], 1 )
#             DCOPY( KI, WORK[1+N2], 1, VR[1, IS], 1 )

#             EMAX = ZERO
#             for K = 1 : KI
#                 EMAX = max( EMAX, abs( VR[K, IS-1] ) + abs( VR[K, IS] ) )
#             end

#             REMAX = ONE / EMAX
#             DSCAL( KI, REMAX, VR[1, IS-1], 1 )
#             DSCAL( KI, REMAX, VR[1, IS], 1 )

#             for K = KI + 1 : N
#                 VR[K, IS-1] = ZERO
#                 VR[K, IS] = ZERO
#             end
#         end
#         IS = IS - 1
#         if IP != 0
#             IS = IS - 1
#         end
#         if IP == 1
#             IP = 0
#         end
#         if IP == -1
#             IP = 1
#         end
#     end

#     return true
# end

# function DAXPY(n, da, dx, incx, dy, incy)

#     if n <= 0 return false end
#     if iszero(da) return false end
#     if incx == 1 && incy == 1

#         # code for both increments equal to 1
#         # clean-up loop
#         m = mod(n,4)
#         if m != 0
#             for i = 1:m
#                 dy[i] = dy[i] + da*dx[i]
#             end
#         end
#         if n < 4 return false end
#         mp1 = m + 1
#         for i = mp1:4:n
#             dy[i] = dy[i] + da*dx[i]
#             dy[i+1] = dy[i+1] + da*dx[i+1]
#             dy[i+2] = dy[i+2] + da*dx[i+2]
#             dy[i+3] = dy[i+3] + da*dx[i+3]
#         end
#     else

#         # code for unequal increments or equal increments
#         #   not equal to 1
#         ix = 1
#         iy = 1
#         if incx < 0 ix = -n+1*incx + 1 end
#         if incy < 0 iy = -n+1*incy + 1 end
#         for i = 1:n
#             dy[iy] = dy[iy] + da*dx[ix]
#             ix = ix + incx
#             iy = iy + incy
#         end
#     end

# end

# function DSCAL(n, da, dx, incx)

#     if n <= 0 || incx <= 0 return false end
#     if incx == 1

#         # code for increment equal to 1
#         # clean-up loop
#         m = mod(n,5)
#         if m != 0
#             for i = 1:m
#                 dx[i] = da*dx[i]
#             end
#             if n < 5 return false end
#         end
#         mp1 = m + 1
#         for i = mp1:5:n
#             dx[i] = da*dx[i]
#             dx[i+1] = da*dx[i+1]
#             dx[i+2] = da*dx[i+2]
#             dx[i+3] = da*dx[i+3]
#             dx[i+4] = da*dx[i+4]
#         end
#     else

#         # code for increment not equal to 1
#         nincx = n*incx
#         for i = 1:incx:nincx
#             dx[i] = da*dx[i]
#         end
#     end
# end

# function DCOPY(n, dx, incx, dy, incy)

#     if n <= 0 return false end
#     if incx == 1 && incy == 1
#     # code for both increments equal to 1 
#     # clean-up loop
#         m = mod(n,7)
#         if m != 0
#             for i = 1:m
#                 dy[i] = dx[i]
#             end
#             if (n < 7) return false end
#         end
#         mp1 = m + 1
#         for i = mp1:7:n
#             dy[i] = dx[i]
#             dy[i+1] = dx[i+1]
#             dy[i+2] = dx[i+2]
#             dy[i+3] = dx[i+3]
#             dy[i+4] = dx[i+4]
#             dy[i+5] = dx[i+5]
#             dy[i+6] = dx[i+6]
#         end
#     else

#     #  code for unequal increments or equal increments
#     #    not equal to 1
#         ix = 1
#         iy = 1
#         if incx < 0 ix = (-n+1)*incx + 1 end
#         if incy < 0 iy = (-n+1)*incy + 1 end
#         for i = 1:n
#             dy[iy] = dx[ix]
#             ix = ix + incx
#             iy = iy + incy
#         end
#     end

#     return true
# end

# function IDAMAX(n, dx, incx)

#     idamax = 0
#     if n < 1 || incx <= 0 return false end
#     idamax = 1
#     if n == 1 return false end
#     if incx == 1
#     #    code for increment equal to 1
#         dmax = abs(dx[1])
#         for i = 2:n
#             if abs(dx[i]) > dmax
#                 idamax = i
#                 dmax = abs(dx[i])
#             end
#         end
#     else
#         # code for increment not equal to 1
#         ix = 1
#         dmax = abs(dx[1])
#         ix = ix + incx
#         for i = 2:n
#             if abs(dx[ix]) > dmax
#                 idamax = i
#                 dmax = abs(dx[ix])
#             end
#             ix = ix + incx
#         end
#     end

#     return true
# end

# function DLALN2(LTRANS, NA, NW, SMIN, CA, A, LDA, D1, D2, B, LDB, WR, WI, X, LDX, SCALE, XNORM, INFO)

#     ZERO = zero(Float64)
#     ONE = one(Float64)
#     TWO = 2*ONE

#     ZSWAP = [false, false, true, true]
#     RSWAP = [false, true, false, true]
#     IPIVOT = [1 2 3 4; 2 1 4 3; 3 4 1 2; 4 3 2 1]

#     CI = Matrix{Float64}(2,2)
#     CIV = CI

#     CR = Matrix{Float64}(2,2)
#     CRV = CR

# #  Compute BIGNUM
#     # SMLNUM = TWO*DLAMCH( 'Safe minimum' )
#     SMLNUM = TWO*eps(Float64)
#     BIGNUM = ONE / SMLNUM
#     SMINI = max( SMIN, SMLNUM )

# #  Don't check for input errors

#     INFO = 0

# #  Standard Initializations

#     SCALE = ONE

#     if NA == 1

#         # 1 x 1  (i.e., scalar) system   C X = B
#         if NW == 1

#             #    Real 1x1 system.

#             #    C = ca A - w D

#             CSR = CA*A[1, 1] - WR*D1
#             CNORM = abs( CSR )

#             #    If | C | < SMINI, use C = SMINI

#             if CNORM < SMINI
#                 CSR = SMINI
#                 CNORM = SMINI
#                 INFO = 1
#             end

#         #    Check scaling for  X = B / C

#             BNORM = abs( B[1, 1] )
#             if CNORM < ONE && BNORM > ONE
#                 if BNORM > BIGNUM*CNORM
#                     SCALE = ONE / BNORM
#                 end
#             end

#         #   Compute X
#             X[1, 1] = B[1, 1]*SCALE / CSR
#             XNORM = abs( X[1, 1] )

#         else

#         #   Complex 1x1 system (w is complex)
#         #   C = ca A - w D
#             CSR = CA*A[1, 1] - WR*D1
#             CSI = -WI*D1
#             CNORM = abs( CSR ) + abs( CSI )

#         #   If | C | < SMINI, use C = SMINI
#             if CNORM < SMINI
#                 CSR = SMINI
#                 CSI = ZERO
#                 CNORM = SMINI
#                 INFO = 1
#             end

#         #   Check scaling for  X = B / C
#             BNORM = abs( B[1, 1] ) + abs( B[1, 2] )
#             if CNORM < ONE && BNORM > ONE
#                 if BNORM > BIGNUM*CNORM
#                     SCALE = ONE / BNORM
#                 end
#             end

#         #   Compute X
#             DLADIV( SCALE*B[1, 1], SCALE*B[1, 2], CSR, CSI, X[1, 1], X[1, 2] )
#             XNORM = abs( X[1, 1] ) + abs( X[1, 2] )
#         end

#     else

# #    2x2 System

# #    Compute the real part of  C = ca A - w D  (or  ca A' - w D )

#         CR[1, 1] = CA*A[1, 1] - WR*D1
#         CR[2, 2] = CA*A[2, 2] - WR*D2
#         if LTRANS
#             CR[1, 2] = CA*A[2, 1]
#             CR[2, 1] = CA*A[1, 2]
#         else
#             CR[2, 1] = CA*A[2, 1]
#             CR[1, 2] = CA*A[1, 2]
#         end

#         if NW == 1

#     #   Real 2x2 system  (w is real)

#     #   Find the largest element in C

#             CMAX = ZERO
#             ICMAX = 0

#             for J = 1:4
#                 if abs( CRV[J] ) > CMAX
#                     CMAX = abs( CRV[J] )
#                     ICMAX = J
#                 end
#             end
#     #    10       CONTINUE

#         #   If norm(C) < SMINI, use SMINI*identity.

#             if CMAX < SMINI
#                 BNORM = max( abs( B[1, 1] ), abs( B[2, 1] ) )
#                 if SMINI < ONE && BNORM > ONE
#                     if BNORM > BIGNUM*SMINI
#                         SCALE = ONE / BNORM
#                     end
#                 end
#                 TEMP = SCALE / SMINI
#                 X[1, 1] = TEMP*B[1, 1]
#                 X[2, 1] = TEMP*B[2, 1]
#                 XNORM = TEMP*BNORM
#                 INFO = 1
#                 return false
#             end

#         #   Gaussian elimination with complete pivoting.

#             UR11 = CRV( ICMAX )
#             CR21 = CRV( IPIVOT[2, ICMAX] )
#             UR12 = CRV( IPIVOT[3, ICMAX] )
#             CR22 = CRV( IPIVOT[4, ICMAX] )
#             UR11R = ONE / UR11
#             LR21 = UR11R*CR21
#             UR22 = CR22 - UR12*LR21

#         #   If smaller pivot < SMINI, use SMINI

#             if abs( UR22 ) < SMINI
#                 UR22 = SMINI
#                 INFO = 1
#             end
#             if RSWAP( ICMAX )
#                 BR1 = B[2, 1]
#                 BR2 = B[1, 1]
#             else
#                 BR1 = B[1, 1]
#                 BR2 = B[2, 1]
#             end
#             BR2 = BR2 - LR21*BR1
#             BBND = max( abs( BR1*( UR22*UR11R ) ), abs( BR2 ) )
#             if BBND > ONE && abs( UR22 ) < ONE
#                 if BBND.GE.BIGNUM*abs( UR22 )
#                     SCALE = ONE / BBND
#                 end
#             end

#             XR2 = ( BR2*SCALE ) / UR22
#             XR1 = ( SCALE*BR1 )*UR11R - XR2*( UR11R*UR12 )
#             if ZSWAP( ICMAX )
#                 X[1, 1] = XR2
#                 X[2, 1] = XR1
#             else
#                 X[1, 1] = XR1
#                 X[2, 1] = XR2
#             end
#             XNORM = max( abs( XR1 ), abs( XR2 ) )

#         #   Further scaling if  norm(A) norm(X) > overflow

#             if XNORM > ONE && CMAX > ONE
#                 if XNORM > BIGNUM / CMAX
#                     TEMP = CMAX / BIGNUM
#                     X[1, 1] = TEMP*X[1, 1]
#                     X[2, 1] = TEMP*X[2, 1]
#                     XNORM = TEMP*XNORM
#                     SCALE = TEMP*SCALE
#                 end
#             end
#         else

#             # Complex 2x2 system  (w is complex)

#             # Find the largest element in C
#             CI[1, 1] = -WI*D1
#             CI[2, 1] = ZERO
#             CI[1, 2] = ZERO
#             CI[2, 2] = -WI*D2
#             CMAX = ZERO
#             ICMAX = 0

#             for J = 1:4
#                 if abs( CRV[J] )+abs( CIV[J] ) > CMAX
#                     CMAX = abs( CRV[J] ) + abs( CIV[J] )
#                     ICMAX = J
#                 end
#             end
#     #    20       CONTINUE

#         #   If norm(C) < SMINI, use SMINI*identity.

#             if CMAX < SMINI
#                 BNORM = max( abs( B[1, 1] )+abs( B[1, 2] ), abs( B[2, 1] )+abs( B[2, 2] ) )
#                 if SMINI < ONE && BNORM > ONE
#                     if BNORM > BIGNUM*SMINI
#                        SCALE = ONE / BNORM
#                     end
#                 end
#                 TEMP = SCALE / SMINI
#                 X[1, 1] = TEMP*B[1, 1]
#                 X[2, 1] = TEMP*B[2, 1]
#                 X[1, 2] = TEMP*B[1, 2]
#                 X[2, 2] = TEMP*B[2, 2]
#                 XNORM = TEMP*BNORM
#                 INFO = 1
#                 return false
#             end

#         #    Gaussian elimination with complete pivoting.

#             UR11 = CRV[ICMAX]
#             UI11 = CIV[ICMAX]
#             CR21 = CRV[IPIVOT[2, ICMAX]]
#             CI21 = CIV[IPIVOT[2, ICMAX]]
#             UR12 = CRV[IPIVOT[3, ICMAX]]
#             UI12 = CIV[IPIVOT[3, ICMAX]]
#             CR22 = CRV[IPIVOT[4, ICMAX]]
#             CI22 = CIV[IPIVOT[4, ICMAX]]

#             if ICMAX == 1 || ICMAX == 4

#             #   Code when off-diagonals of pivoted C are real

#                 if abs( UR11 ) > abs( UI11 )
#                     TEMP = UI11 / UR11
#                     UR11R = ONE / ( UR11*( ONE+TEMP^2 ) )
#                     UI11R = -TEMP*UR11R
#                 else
#                     TEMP = UR11 / UI11
#                     UI11R = -ONE / ( UI11*( ONE+TEMP^2 ) )
#                     UR11R = -TEMP*UI11R
#                 end
#                 LR21 = CR21*UR11R
#                 LI21 = CR21*UI11R
#                 UR12S = UR12*UR11R
#                 UI12S = UR12*UI11R
#                 UR22 = CR22 - UR12*LR21
#                 UI22 = CI22 - UR12*LI21
#             else

#             #  Code when diagonals of pivoted C are real

#                 UR11R = ONE / UR11
#                 UI11R = ZERO
#                 LR21 = CR21*UR11R
#                 LI21 = CI21*UR11R
#                 UR12S = UR12*UR11R
#                 UI12S = UI12*UR11R
#                 UR22 = CR22 - UR12*LR21 + UI12*LI21
#                 UI22 = -UR12*LI21 - UI12*LR21
#             end
#             U22ABS = abs( UR22 ) + abs( UI22 )

#         #   If smaller pivot < SMINI, use SMINI

#             if U22ABS < SMINI
#                 UR22 = SMINI
#                 UI22 = ZERO
#                 INFO = 1
#             end
#             if RSWAP( ICMAX )
#                 BR2 = B[1, 1]
#                 BR1 = B[2, 1]
#                 BI2 = B[1, 2]
#                 BI1 = B[2, 2]
#             else
#                 BR1 = B[1, 1]
#                 BR2 = B[2, 1]
#                 BI1 = B[1, 2]
#                 BI2 = B[2, 2]
#             end
#             BR2 = BR2 - LR21*BR1 + LI21*BI1
#             BI2 = BI2 - LI21*BR1 - LR21*BI1
#             BBND = max( ( abs( BR1 )+abs( BI1 ) )*( U22ABS*( abs( UR11R )+abs( UI11R ) ) ), abs( BR2 )+abs( BI2 ) )
#             if BBND > ONE && U22ABS < ONE
#                 if BBND.GE.BIGNUM*U22ABS
#                     SCALE = ONE / BBND
#                     BR1 = SCALE*BR1
#                     BI1 = SCALE*BI1
#                     BR2 = SCALE*BR2
#                     BI2 = SCALE*BI2
#                 end
#             end

#             DLADIV( BR2, BI2, UR22, UI22, XR2, XI2 )
#             XR1 = UR11R*BR1 - UI11R*BI1 - UR12S*XR2 + UI12S*XI2
#             XI1 = UI11R*BR1 + UR11R*BI1 - UI12S*XR2 - UR12S*XI2
#             if ZSWAP( ICMAX )
#                 X[1, 1] = XR2
#                 X[2, 1] = XR1
#                 X[1, 2] = XI2
#                 X[2, 2] = XI1
#             else
#                 X[1, 1] = XR1
#                 X[2, 1] = XR2
#                 X[1, 2] = XI1
#                 X[2, 2] = XI2
#             end
#             XNORM = max( abs( XR1 )+abs( XI1 ), abs( XR2 )+abs( XI2 ) )

#         #   Further scaling if  norm(A) norm(X) > overflow

#             if XNORM > ONE && CMAX > ONE
#                 if XNORM > BIGNUM / CMAX
#                     TEMP = CMAX / BIGNUM
#                     X[1, 1] = TEMP*X[1, 1]
#                     X[2, 1] = TEMP*X[2, 1]
#                     X[1, 2] = TEMP*X[1, 2]
#                     X[2, 2] = TEMP*X[1, 2]                     
#                     XNORM = TEMP*XNORM
#                     SCALE = TEMP*SCALE
#                 end
#             end
#         end
#     end

#     return true

# # End of DLALN2
# end

# function DLADIV(A, B, C, D, P, Q)

#     if abs( D ) < abs( C )
#         E = D / C
#         F = C + D*E
#         P = ( A+B*E ) / F
#         Q = ( B-A*E ) / F
#     else
#         E = C / D
#         F = D + C*E
#         P = ( B+A*E ) / F
#         Q = ( -A+B*E ) / F
#     end

# end