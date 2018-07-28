using Test

using IRAM: eigvalues, local_schurfact!, backward_subst!
using LinearAlgebra

@testset "Schur factorization" begin
    let
        # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
        H = [1.0 2.0; 3.0 4.0]
        H_copy = copy(H)
        Q = Matrix{Float64}(I, 2, 2)

        @test local_schurfact!(H, Q, 1, 2, maxiter = 2)
        @test norm(Q' * H_copy * Q - H) < 10eps()
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H), by=abs, rev=true)
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H_copy), by=abs, rev=true)
        @test abs(H[2,1]) < 10eps()
    end

    let
        # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
        H = [1.0 2.0; 0.0 4.0]
        H_copy = copy(H)
        Q = Matrix{Float64}(I, 2, 2)
        
        @test local_schurfact!(H, Q, 1, 2, maxiter = 2)
        @test norm(Q' * H_copy * Q - H) < 10eps()
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H), by=abs, rev=true)
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H_copy), by=abs, rev=true)
        @test abs(H[2,1]) < 10eps()
    end

    let
        # 2-by-2 matrix with conjugate eigenvalues
        H = [1.0 4.0; -5.0 3.0]
        H_copy = copy(H)
        Q = Matrix{Float64}(I, 2, 2)
        
        @test local_schurfact!(H, Q, 1, 2, maxiter = 2)
        @test norm(Q' * H_copy * Q - H) < 10eps()
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H), by=abs, rev=true)
        @test sort!(eigvalues(H), by=abs, rev=true) ≈ sort!(eigvals(H_copy), by=abs, rev=true)
    end
    
    # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    for i = 0 : 4
        H = triu(rand(10,10), -1)

        # Add zeros to the subdiagonals assuming convergence
        if i != 0 
            # The previous block has converged, hence H[i+1,i] = 0
            H[i+1,i] = 0

            # Current block has converged, hence H[11-i,10-i] = 0
            H[11-i,10-i] = 0
        end
        λs = sort!(eigvals(H), by=abs, rev=true)
        H_copy = copy(H)
        Q = Matrix{Float64}(I, 10, 10)

        # Test that the procedure has converged
        @test local_schurfact!(H, Q, 1+i, 10-i)

        for j = 1 : 9 - 2*i
            t = H[i+j,i+j] + H[i+j+1,i+j+1]
            d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

            # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
            @test abs(H[i+j+1,i+j]) < 10eps() || t*t/4 - d < 0
        end

        # Test that the elements below the subdiagonal are 0
        for j = 1:10, i = j+2:10
            @test abs(H[i,j]) < 10eps()
        end

        # Test that the partial Schur decomposition relation holds
        @test norm(Q*H*Q' - H_copy) < 100eps()
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
    end


    # COMPLEX ARITHMETIC

    # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    for i = 0 : 4
        H = triu(rand(ComplexF64, 10,10), -1)

        # Add zeros to the subdiagonals assuming convergence
        if i != 0 
            # The previous block has converged, hence H[i+1,i] = 0
            H[i+1,i] = zero(ComplexF64)

            # Current block has converged, hence H[11-i,10-i] = 0
            H[11-i,10-i] = zero(ComplexF64)
        end

        λs = sort!(eigvals(H), by=abs, rev=true)
        H_copy = copy(H)
        Q = Matrix{ComplexF64}(I, 10, 10)

        # Test that the procedure has converged
        @test local_schurfact!(H, Q, 1+i, 10-i)

        for j = 1 : 9 - 2*i  
            # Test if subdiagonal is small. 
            @test abs(H[i+j+1,i+j]) < 10eps()
        end

        # Test that the elements below the subdiagonal are 0
        for j = 1:10
            for i = j+2:10
                @test abs(H[i,j]) < 10eps()
            end
        end

        # Test that the partial Schur decomposition relation holds
        @test norm(Q*H*Q' - H_copy) < 1000eps()
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
    end
end

@testset "Backward subsitution" begin

    # # Test whether backward substitution works
    # let  
    #     for i = 10:15
    #         for T in (Float64, ComplexF64)       
    #             R = triu(rand(T, i,i))
    #             y = rand(T, i)
    #             x = R\y
    #             backward_subst!(R, y)
    #             # @test R*x ≈ y
    #             # R should be identity
    #             @test x ≈ y
    #         end
    #     end
    # end

    # Test whether the eigenvector comes out properly
    for T in (Float64,ComplexF64)
        R = triu(rand(T, 10,10))
        for i = 2:10
            R_small = copy(R[1:i-1,1:i-1])
            λs, vs = eigen(R)
            y = -R[1:i-1,i]

            x = (R_small-I*R[i,i]) \ y

            backward_subst!(R_small, R[i,i], y)
            eigvec = [y; 1.0; zeros(Float64, 10-i)] / norm([y; 1.0; zeros(Float64, 10-i)])
            
            @test x ≈ y
            @test abs.(vs[:,i]) ≈ abs.(eigvec)
        end
    end

    #Real arithmetic with conjugate eigvals
    for i = 6:10
        R = triu(rand(10,10))
        R[i-4,i-5] = rand()
        R[i-2,i-3] = rand()
        R_small = copy(R[1:i-1,1:i-1])
        λs, vs = eigen(R)
        y = -R[1:i-1,i]
        x = (R_small-I*R[i,i]) \ y

        backward_subst!(R_small, R[i,i], y)
        eigvec = [y; 1.0; zeros(Float64, 10-i)] / norm([y; 1.0; zeros(Float64, 10-i)])
            
        @test x ≈ y
        @test vs[:,i] ≈ eigvec
    end

    #Repeated eigvals at different positions.
    for T in (Float64,ComplexF64)
        for i = 6:10
            R = triu(rand(T, 10,10))
            lambda = rand(T)
            R[i-2,i-2] = lambda
            R[i-3,i-3] = lambda
            R[i,i] = lambda
            λs, vs = eigen(R)
            R_small = R[1:i-1,1:i-1]
            y = [-R[1:i-1,i]; 1.0]

            backward_subst!(R_small, R[i,i], y)
            eigvec = [y; zeros(Float64, 10-i)] / norm([y; zeros(Float64, 10-i)])
                
            @test abs.(vs[:,i]) ≈ abs.(eigvec)
            # @show eigvec
            # @show vs[:,i]
        end
    end

    # Repeated eigvals. Test that all eigenvectors are correct.
    for T in (Float64,ComplexF64)
        R = triu(rand(T, 10,10))
        lambda = rand(T)
        R[9,9] = lambda
        R[8,8] = lambda
        R[7,7] = lambda
        λs, vs = eigen(R)

        for i = 10:-1:1
            R_small = R[1:i-1,1:i-1]
            y = [-R[1:i-1,i]; 1.0]

            backward_subst!(R_small, R[i,i], y)
            eigvec = [y; zeros(Float64, 10-i)] / norm([y; zeros(Float64, 10-i)])
                
            @test abs.(vs[:,i]) ≈ abs.(eigvec)
        end
    end

    #Geometric multiplicity > 1
    for T in (Float64,ComplexF64)
        R = triu(rand(T, 10,10))
        lambda = rand(T)
        R[9,9] = lambda
        R[8,8] = lambda
        R[8,9] = zero(T)
        R[7,7] = lambda
        R[7,9] = zero(T)
        R[7,8] = zero(T)
        λs, vs = eigen(R)

        for i = 10:-1:1
            R_small = R[1:i-1,1:i-1]
            y = [-R[1:i-1,i]; 1.0]

            backward_subst!(R_small, R[i,i], y)
            eigvec = [y; zeros(Float64, 10-i)] / norm([y; zeros(Float64, 10-i)])
            # @show eigvec
            # @show vs[:,i]
            @test abs.(vs[:,i]) ≈ abs.(eigvec)
        end
    end

    #Eigenvectors for conjugate eigenvalues
    let
        R = triu(rand(Float64, 10,10))
        a = rand(Float64)
        R[10,10] = a 
        R[9,9] = a
        R[9,10] = -3a
        R[10,9] = a
        R[5,4] = a
        λs, vs = eigen(R)

        y = Vector{ComplexF64}(undef, 10)
        backward_subst!(R, y)
        y ./= norm(y)
        @test abs.(vs[:,9]) ≈ abs.(y)
    end

    let
        R = triu(rand(Float64, 10,10))
        a = rand(Float64)
        R[6,6] = a 
        R[5,5] = a
        R[5,6] = -3a
        R[6,5] = a
        λs, vs = eigen(R)

        y = Vector{ComplexF64}(undef, 10)
        backward_subst!(R[1:6,1:6], y)
        y ./= norm(y)
        @test abs.(vs[:,5]) ≈ abs.(y)
    end
end