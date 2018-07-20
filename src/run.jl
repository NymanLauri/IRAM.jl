"""
Run IRAM until the eigenvectors are approximated to the prescribed tolerance or until 
`max_restarts` has been reached.
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, nev = min, ε = eps(T), max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    h = Vector{T}(undef, max)
    iterate_arnoldi!(A, arnoldi, 1 : min, h)

    # min′ is the effective starting point -- may be min - 1 when the last two removed guys
    # are a complex conjugate
    min′ = min

    active = 1
    V_prealloc = Matrix{T}(undef, n, min)
    for restarts = 1 : max_restarts
        n = max - active + 1

        iterate_arnoldi!(A, arnoldi, min′ + 1 : max, h)
        
        # Compute the eigenvalues of the active part
        Q = Matrix{T}(I, n, n)
        H_copy = copy(view(arnoldi.H, active:max, active:max))
        local_schurfact!(H_copy, Q)
        # λs = sort!(eigvalues(H_copy), by = abs, rev = true)
        λs = eigvalues(H_copy)

        y = Vector{T}(undef,n)
        res = Vector{Float64}(undef,n)
        @inbounds for i = n : -1 : 1
            y[i] = one(T)
            y[1:i-1] .= - view(H_copy, 1:i-1, i)
            y[i+1:n] .= zero(T)
            backward_subst!(view(H_copy,1:i-1,1:i-1), y, H_copy[i,i])
            y ./= norm(y)

            res[i] = abs(dot(view(Q, n, :), y) * arnoldi.H[max + 1, max])
        end

        perm = sortperm(res, by=abs)
        λs .= λs[perm]

        min′ = implicit_restart!(arnoldi, λs, min, max, active, V_prealloc)
        new_active = detect_convergence!(view(arnoldi.H, active:min′+1, active:min′), ε)
        new_active += active - 1 
        if new_active > active + 1
            # Bring the new locked part oF H into upper triangular form
            transform_converged(arnoldi, active, new_active, min′, V_prealloc)
        end

        active = new_active

        active > nev && break
    end

    return PartialSchur(arnoldi.V, arnoldi.H, active - 1)
end

"""
Transfrom the converged block into an upper triangular form.
"""
function transform_converged(arnoldi, active, new_active, min′, V_prealloc)
    
    # H = Q R Q'

    # A V = V H
    # A V = V Q R Q'
    # A (V Q) = (V Q) R
    
    # V <- V Q
    # H_right <- Q' H_right
    # H_lock <- Q' H_lock Q
    # H_above <- H_above Q

    Q_large = Matrix{eltype(arnoldi.H)}(I, new_active - 1, new_active - 1)

    H_locked = view(arnoldi.H, active : new_active - 1, active : new_active - 1)
    H_copy = copy(H_locked)

    H_copy_full = copy(arnoldi.H)

    local_schurfact!(arnoldi.H, Q_large, active, new_active - 1)
    Q_small = view(Q_large, active : new_active - 1, active : new_active - 1)

    V_locked = view(arnoldi.V, :, active : new_active - 1)
    mul!(view(V_prealloc, :, active : new_active - 1), V_locked, Q_small)
    V_locked .= view(V_prealloc, :, active : new_active - 1)

end
