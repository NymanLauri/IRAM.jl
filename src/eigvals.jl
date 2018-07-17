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

function schur_to_eigen(Schur::PartialSchur{TQ,TR}, SIDE, HOWMNY, SELECT, N, LDT, VL, LDVL, 
    VR, LDVR, MM, M, WORK, INFO )) where {TQ,TR}
    T = Schur.R
    ZERO = zero(Float64)

    if SOMEV
    M = 0
    PAIR = false
    for J = 1 : N
       if PAIR
          PAIR = false
          SELECT[J] = false
       else
          if J < N
             if T[J+1, J] == ZERO
                if SELECT[J]
                     M = M + 1
                end
             else
                PAIR = .TRUE.
                if SELECT[J] || SELECT[J+1]
                   SELECT[J] = .TRUE.
                   M = M + 2
                end
             end
          else
             if SELECT[N]
                M = M + 1
             end
          end
       end
    end
 else
    M = N
 end

 if MM < M
    INFO = -11
 end
end
if INFO != 0
 CALL XERBLA( 'DTREVC', -INFO )
 return false
end
# *
# *     Quick return if possible.
# *
if N == 0
    return false
end
# *
# *     Set the constants to control overflow.
# *
UNFL = DLAMCH( 'Safe minimum' )
OVFL = ONE / UNFL
CALL DLABAD( UNFL, OVFL )
ULP = DLAMCH( 'Precision' )
SMLNUM = UNFL*( N / ULP )
BIGNUM = ( ONE-ULP ) / SMLNUM
# *
# *     Compute 1-norm of each column of strictly upper triangular
# *     part of T to control overflow in triangular solver.
# *
WORK[1] = ZERO
for J = 2 : N
 WORK[J] = ZERO
 for I = 1 : J - 1
    WORK[J] = WORK[J] + abs( T[I, J] )
 end
end



# Right eigenvectors
    IP = 0
    IS = M
    DO 140 KI = N, 1, -1

       if IP == 1 
         GO TO 130
       end
       if KI != 1 && T[KI, KI-1] != zero(TR)
        IP = -1 #(GO TO 40)
       end

# 40       CONTINUE
       if SOMEV != 0 
          if IP == 0
             if !select(KI)
               GO TO 130
             end
          else
             if !select(KI-1)
               GO TO 130
             end
        end
# *
# *           Compute the KI-th eigenvalue (WR,WI).
# *
       WR = T[KI, KI]
       WI = ZERO
       if IP != 0 
         WI = sqrt( abs( T[KI, KI-1] ) )*sqrt( abs( T[KI, KI-1] ) )
       end
       SMIN = max( ULP*( abs( WR )+abs( WI ) ), SMLNUM )
*
       if IP == 0
# *
# *              Real right eigenvector
# *
          WORK[KI+N] = one(eltype(WORK))
# *
# *              Form right-hand side
# *
          for K = 1 : KI - 1
             WORK[K+N] = -T[K, KI]
          end
# *
# *              Solve the upper quasi-triangular system:
# *                 (T(1:KI-1,1:KI-1) - WR)*X = SCALE*WORK.
# *
          JNXT = KI - 1
          for J = KI - 1: -1 : 1
             if J > JNXT
            #    GO TO 60
                break
             end
             J1 = J
             J2 = J
             JNXT = J - 1
             if J > 1
                if !iszero(T[J, J-1])
                   J1 = J - 1
                   JNXT = J - 2
                end
            end

             if J1 == J2
# *
# *                    1-by-1 diagonal block
# *
                CALL DLALN2( false, 1, 1, SMIN, ONE, T[J, J],
$                            LDT, ONE, ONE, WORK[J+N], N, WR,
$                            ZERO, X, 2, SCALE, XNORM, IERR )
# *
# *                    Scale X(1,1) to avoid overflow when updating
# *                    the right-hand side.
# *
                if XNORM > ONE
                   if WORK[J] > BIGNUM / XNORM
                      X[1, 1] = X[1, 1] / XNORM
                      SCALE = SCALE / XNORM
                   end
                end
# *
# *                    Scale if necessary
# *
                if SCALE != ONE
$                  CALL DSCAL( KI, SCALE, WORK[1+N], 1 )
                end
                WORK[J+N] = X[1, 1]
# *
# *                    Update right-hand side
# *
                CALL DAXPY( J-1, -X[1, 1], T[1, J], 1,
$                           WORK[1+N], 1 )
*
            else
# *
# *                    2-by-2 diagonal block
# *
                CALL DLALN2( false, 2, 1, SMIN, ONE,
$                            T[J-1, J-1 ), LDT, ONE, ONE,
$                            WORK[J-1+N], N, WR, ZERO, X, 2,
$                            SCALE, XNORM, IERR )
# *
# *                    Scale X(1,1) and X(2,1) to avoid overflow when
# *                    updating the right-hand side.              
                if XNORM > ONE
                    BETA = max( WORK( J-1 ), WORK )
                    if BETA > BIGNUM / XNORM
                      X[1, 1] = X[1, 1] / XNORM
                      X[2, 1] = X[2, 1] / XNORM
                      SCALE = SCALE / XNORM
                    end
                end
# *
# *                    Scale if necessary
# *
                if SCALE != ONE
$                  CALL DSCAL( KI, SCALE, WORK[1+N], 1 )
                end
                WORK[J-1+N] = X[1, 1]
                WORK[J+N] = X[2, 1]
# *
# *                    Update right-hand side
# *
                CALL DAXPY( J-2, -X[1, 1], T[1, J-1 ), 1,
$                           WORK[1+N], 1 )
                CALL DAXPY( J-2, -X[2, 1], T[1, J], 1,
$                           WORK[1+N], 1 )
                end
            end
# 60          CONTINUE
# *
# *              Copy the vector x or Q*x to VR and normalize.
# *
        if !OVER
            CALL DCOPY( KI, WORK[1+N], 1, VR[1, IS], 1 )

            II = IDAMAX( KI, VR[1, IS], 1 )
            REMAX = ONE / abs( VR[II, IS] )
            CALL DSCAL( KI, REMAX, VR[1, IS], 1 )

            for K = KI + 1 : N
                VR[K, IS] = ZERO
            end
# 70             CONTINUE
        else
            if KI > 1
    $         CALL DGEMV( 'N', N, KI-1, ONE, VR, LDVR,
    $                           WORK[1+N], 1, WORK( KI+N ),
    $                           VR[1, KI], 1 )
            end
                II = IDAMAX( N, VR[1, KI], 1 )
                REMAX = ONE / abs( VR( II, KI ) )
                CALL DSCAL( N, REMAX, VR[1, KI], 1 )
        end
    *
        else
# *
# *              Complex right eigenvector.
# *
# *              Initial solve
# *                [ (T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I* WI)]*X = 0.
# *                [ (T(KI,KI-1)   T(KI,KI)   )               ]
# *
          if abs( T[KI-1, KI ) > abs( T[KI, KI-1 )
             WORK[KI-1+N] = ONE
             WORK[KI+N2] = WI / T[KI-1, KI )
          else
             WORK[KI-1+N] = -WI / T[KI, KI-1 )
             WORK[KI+N2] = ONE
          end
          WORK( KI+N ) = ZERO
          WORK( KI-1+N2 ) = ZERO
# *
# *              Form right-hand side
# *
          for K = 1 : KI - 2
             WORK( K+N ) = -WORK[KI-1+N]*T[K, KI-1 )
             WORK( K+N2 ) = -WORK[KI+N2]*T[K, KI )
          end
# 80          CONTINUE
# *
# *              Solve upper quasi-triangular system:
# *              (T(1:KI-2,1:KI-2) - (WR+i*WI))*X = SCALE*(WORK+i*WORK2)
# *
          JNXT = KI - 2
          for J = KI - 2:-1:1
            if J > JNXT
    # $        GO TO 90
                break
            end
             J1 = J
             J2 = J
             JNXT = J             
             if J > 1
                if T[J, J-1] != ZERO
                   J1 = J - 1
                   JNXT = J - 2
                end
             end
# *
             if J1 == J2
# *
# *                    1-by-1 diagonal block
# *
                CALL DLALN2( false, 1, 2, SMIN, ONE, T[J, J ),
$                            LDT, ONE, ONE, WORK[J+N], N, WR, WI,
$                            X, 2, SCALE, XNORM, IERR )
# *
# *                    Scale X(1,1) and X(1,2) to avoid overflow when
# *                    updating the right-hand sid                
                if XNORM > ONE N
                    if WORK[J] > BIGNUM / XNORM
                      X[1, 1] = X[1, 1] / XNORM
                      X[1, 2] = X[1, 2] / XNORM
                      SCALE = SCALE / XNORM
                   end
                end
# *
# *                    Scale if necessary
# *
                if SCALE != ONE
                   CALL DSCAL( KI, SCALE, WORK[1+N], 1 )
                   CALL DSCAL( KI, SCALE, WORK[1+N2], 1 )
                end
                WORK[J+N] = X[1, 1]
                WORK[J+N2] = X[1, 2]
# *
# *                    Update the right-hand side
# *
                CALL DAXPY( J-1, -X[1, 1], T[1, J], 1,
$                           WORK[1+N], 1 )
                CALL DAXPY( J-1, -X[1, 2], T[1, J], 1,
$                           WORK[1+N2], 1 )
# *
             else
# *
# *                    2-by-2 diagonal block
# *
                CALL DLALN2( false, 2, 2, SMIN, ONE,
$                            T[J-1, J-1 ), LDT, ONE, ONE,
$                            WORK[J-1+N], N, WR, WI, X, 2, SCALE,
$                            XNORM, IERR )
# *
# *                    Scale X to avoid overflow when updating
# *                    the right-hand sid                
                if XNORM > ONE
                    BETA = max( WORK( J-1 ), WORK )
                    if BETA > BIGNUM / XNORM
                      REC = ONE / XNORM
                      X[1, 1] = X[1, 1]*REC
                      X[1, 2] = X[1, 2]*REC
                      X[2, 1] = X[2, 1]*REC
                      X[2, 2] = X[2, 2]*REC
                      SCALE = SCALE*REC
                    end
                end
# *
# *                    Scale if necessary
# *
                if SCALE != ONE
                   CALL DSCAL( KI, SCALE, WORK[1+N], 1 )
                   CALL DSCAL( KI, SCALE, WORK[1+N2], 1 )
                end
                WORK[J-1+N] = X[1, 1]
                WORK[J+N] = X[2, 1]
                WORK[J-1+N2] = X[1, 2]
                WORK[J+N2] = X[2, 2]
# *
# *                    Update the right-hand side
# *
                CALL DAXPY( J-2, -X[1, 1], T[1, J-1 ), 1,
$                           WORK[1+N], 1 )
                CALL DAXPY( J-2, -X[2, 1], T[1, J], 1,
$                           WORK[1+N], 1 )
                CALL DAXPY( J-2, -X[1, 2], T[1, J-1 ), 1,
$                           WORK[1+N2], 1 )
                CALL DAXPY( J-2, -X[2, 2], T[1, J], 1,
$                           WORK[1+N2], 1 )
            end
        end
# 90          CONTINUE
# *
# *              Copy the vector x or Q*x to VR and normalize.
# *
          if !OVER THEN
             CALL DCOPY( KI, WORK[1+N], 1, VR[1, IS-1], 1 )
             CALL DCOPY( KI, WORK[1+N2], 1, VR[1, IS], 1 )

             EMAX = ZERO
             for K = 1 : KI
                EMAX = max( EMAX, abs( VR[K, IS-1] ) + abs( VR[K, IS] ) )
             end
# 100             CONTINUE

             REMAX = ONE / EMAX
             CALL DSCAL( KI, REMAX, VR[1, IS-1], 1 )
             CALL DSCAL( KI, REMAX, VR[1, IS], 1 )

             for K = KI + 1 : N
                VR[K, IS-1] = ZERO
                VR[K, IS] = ZERO
             end
# 110             CONTINUE

            else
                if KI > 2
                    CALL DGEMV( 'N', N, KI-2, ONE, VR, LDVR,
$                           WORK[1+N], 1, WORK[KI-1+N],
$                           VR[1, KI-1], 1 )
                    CALL DGEMV( 'N', N, KI-2, ONE, VR, LDVR,
$                           WORK[1+N2], 1, WORK[KI+N2],
$                           VR[1, KI], 1 )
                else
                    CALL DSCAL( N, WORK[KI-1+N], VR[1, KI-1], 1 )
                    CALL DSCAL( N, WORK[KI+N2], VR[1, KI], 1 )
                end

             EMAX = ZERO
             for K = 1 : N
                EMAX = max( EMAX, abs( VR( K, KI-1 ) ) + abs( VR( K, KI ) ) )
             end
# 120             CONTINUE
             REMAX = ONE / EMAX
             CALL DSCAL( N, REMAX, VR[1, KI-1], 1 )
             CALL DSCAL( N, REMAX, VR[1, KI], 1 )
          end
       end

       IS = IS - 1
       if IP != 0
         IS = IS - 1
       end
# 130       CONTINUE
       if IP == 1
         IP = 0
       end
       if IP == -1
         IP = 1
       end
# 140    CONTINUE
end



end
