using Test
using BasicBlackHoleSim
using BasicBlackHoleSim.Constants
using BasicBlackHoleSim.Utils
using LinearAlgebra
using DifferentialEquations 

# --- Test Helper Functions ---

"""
Calculates the specific orbital energy (energy per unit mass) for a Newtonian orbit.
E/m = 1/2 v^2 - G M / r
"""
function calculate_newtonian_energy(u, M)
    q = u[1:3]
    v = u[4:6]
    
    r = norm(q)
    v_sq = dot(v, v)
    
    kinetic_energy = 0.5 * v_sq
    potential_energy = -(G * M) / r
    
    return kinetic_energy + potential_energy
end

"""
Calculates the conserved energy (E) and axial angular momentum (Lz) per unit mass
for a particle in a Kerr spacetime. These are constants of motion for a geodesic.
"""
function calculate_kerr_constants_of_motion(u, M, a)
    r, theta = u[2], u[3]
    ut, uphi = u[5], u[8]

    sin_theta_sq = sin(theta)^2
    cos_theta_sq = cos(theta)^2

    Σ = r^2 + a^2 * cos_theta_sq
    Δ = r^2 - 2*M*r + a^2

    # Covariant metric components
    g_tt = -(1 - 2 * M * r / Σ)
    g_tφ = -(2 * M * r * a * sin_theta_sq / Σ)
    g_φφ = ((r^2 + a^2)^2 - Δ * a^2 * sin_theta_sq) * sin_theta_sq / Σ

    # Conserved quantities per unit mass (from covariant momentum components)
    E = -(g_tt * ut + g_tφ * uphi)
    Lz = g_tφ * ut + g_φφ * uphi
    
    return E, Lz
end

"""
Calculates the squared norm of the 4-velocity, g_μν u^μ u^ν.
For a massive particle, this should be conserved and equal to -1.
"""
function calculate_4_velocity_norm_sq(u, M, a)
    r, theta, ut, ur, utheta, uphi = u[2], u[3], u[5], u[6], u[7], u[8]
    sin_theta_sq = sin(theta)^2
    Σ = r^2 + a^2 * cos(theta)^2
    Δ = r^2 - 2 * M * r + a^2

    # Re-calculate metric components needed for the dot product
    g_tt = -(1 - 2 * M * r / Σ); g_tφ = -(2 * M * r * a * sin_theta_sq / Σ)
    g_rr = Σ / Δ; g_θθ = Σ
    g_φφ = ((r^2 + a^2)^2 - Δ * a^2 * sin_theta_sq) * sin_theta_sq / Σ
    
    return g_tt*ut^2 + g_rr*ur^2 + g_θθ*utheta^2 + g_φφ*uphi^2 + 2*g_tφ*ut*uphi
end

@testset "BasicBlackHoleSim.jl" begin

    @testset "Physics Helpers" begin
        M = 1.0 * M_sun
        
        @testset "get_black_hole_parameters" begin
            expected_Rs = (2 * G * M) / c^2
            params = get_black_hole_parameters(M)
            @test params.rs ≈ expected_Rs
        end
        
        @testset "circular_velocity" begin
            r0 = 5.0e7
            expected_v = sqrt(G * M / r0)
            @test circular_velocity(M, r0) ≈ expected_v
        end
    end

    @testset "Post-Newtonian Solvers" begin
        M = 1.0 * M_sun
        r0 = 5.0e7 
        v0 = circular_velocity(M, r0)
        tspan = (0.0, 10.0) # Short simulation time for testing

        u0 = [r0, 0.0, 0.0, 0.0, v0, 0.0] 

        @testset "Newtonian Solver" begin
            sol = simulate_orbit(:newtonian, u0, tspan, M)
            @test Symbol(sol.retcode) == :Success
            
            # Test for energy conservation
            E_initial = calculate_newtonian_energy(sol.u[1], M)
            E_final = calculate_newtonian_energy(sol.u[end], M)
            @test E_final ≈ E_initial rtol=1e-6 # Check conservation to a reasonable tolerance
        end

        @testset "Schwarzschild Solver" begin
            sol = simulate_orbit(:schwarzschild, u0, tspan, M)
            @test Symbol(sol.retcode) == :Success
            @test sol.u[end] != u0
        end

        @testset "Kerr Solver" begin
            a_star = 0.98
            p = (M, a_star)
            sol = simulate_orbit(:kerr, u0, tspan, p)
            @test Symbol(sol.retcode) == :Success
            @test sol.u[end] != u0
        end
    end

    @testset "Geodesic Kerr Solver" begin
        M_kg = 1.0 * M_sun
        a_star = 0.98

        M_geom = (G * M_kg) / c^2 
        a_geom = a_star * M_geom
        
        # Stable circular orbit initial conditions
        r0 = 6.0 * M_geom
        theta0 = π/2
        ur0 = 0.0
        utheta0 = 0.0

        # Calculate initial 4-velocities for a circular orbit
        ut, uphi = calculate_circular_geodesic_velocity(r0, M_geom, a_geom)
        ut0 = normalize_velocity(r0, theta0, ur0, utheta0, uphi, M_geom, a_geom)
        
        @test !isnothing(ut0) 

        # State: t, r, θ, φ, ut, ur, uθ, uφ
        u0 = [0.0, r0, theta0, 0.0, ut0, ur0, utheta0, uphi] 
        tspan = (0.0, 50.0 * M_geom) # Simulate for a short proper time
        p = (M_geom, a_geom)

        # Use a stiff solver for the geodesic equations, which are numerically stiff.
        sol = simulate_orbit(:kerr_geodesic, u0, tspan, p, alg=Rodas5(), maxiters=1e7)
        @test Symbol(sol.retcode) == :Success

        # Test for conservation of Energy (E) and Angular Momentum (Lz)
        E_initial, Lz_initial = calculate_kerr_constants_of_motion(sol.u[1], M_geom, a_geom)
        E_final, Lz_final = calculate_kerr_constants_of_motion(sol.u[end], M_geom, a_geom)
        
        @test E_final ≈ E_initial rtol=1e-7
        @test Lz_final ≈ Lz_initial rtol=1e-7

        # Test that 4-velocity remains normalized to -1
        norm_sq_initial = calculate_4_velocity_norm_sq(sol.u[1], M_geom, a_geom)
        norm_sq_final = calculate_4_velocity_norm_sq(sol.u[end], M_geom, a_geom)

        @test norm_sq_initial ≈ -1.0 rtol=1e-9
        @test norm_sq_final ≈ -1.0 rtol=1e-7 # Allow slightly larger tolerance at the end
    end
    
end