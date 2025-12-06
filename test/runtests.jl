using Test
using BasicBlackHoleSim
using LinearAlgebra

@testset "Black Hole Physics Tests" begin
    
    # --- Test 1: Check Schwarzschild Radius Calculation ---
    M = 2.0e30
    G = BasicBlackHoleSim.Constants.G
    c = BasicBlackHoleSim.Constants.c
    expected_Rs = (2*G*M) / c^2
    @test G ≈ 6.67430e-11
    
    # --- Test 2: Newtonian Physics ---
    u = [100.0, 0.0, 0.0, 0.0] # x=100, vx=0
    du = similar(u)
    p = M
    BasicBlackHoleSim.Physics.newtonian_2D_model!(du, u, p, 0.0)
    @test du[3] < 0 
    @test du[1] == 0

end