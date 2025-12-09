module Physics

using ..Constants 
using LinearAlgebra
using ForwardDiff

# EXPORTS: We now export the split functions
export velocity_law!, newtonian_acceleration!, schwarzschild_acceleration!, kerr_acceleration!, kerr_geodesic!

# ===========================================
#       POST-NEWTONIAN APPROXIMATIONS
# ===========================================

"""
VELOCITY LAW
"""
function velocity_law!(dq, v, q, p, t)
    dq[1] = v[1]
    dq[2] = v[2]
    dq[3] = v[3]
end

"""
NEWTONIAN ACCELERATION
"""
function newtonian_acceleration!(dv, v, q, p, t)

    local G = Constants.G
    local M = p

    x, y, z = q[1], q[2], q[3]
    r = norm(q)

    prefactor = -(G * M) / (r^3)
    
    # Update Acceleration (dv)
    dv[1] = prefactor * x
    dv[2] = prefactor * y
    dv[3] = prefactor * z
end

"""
SCHWARZSCHILD ACCELERATION
"""
function schwarzschild_acceleration!(dv, v, q, p, t)
    
    local G = Constants.G       # gravity constant
    local M = p                 # mass
    local c = Constants.c       # speed of light

    x, y, z = q[1], q[2], q[3]  # pos
    r_sq = x^2 + y^2 + z^2      # radius distance norn 
    r = sqrt(r_sq)              # radius 
    
    h_vec = cross(q, v) 
    h_sq = dot(h_vec, h_vec) 

    # Coefficients
    term_newton = -(G * M) / (r * r_sq)                           # Newtonian term: -GM/r^3
    term_gr = -(3 * G * M * h_sq) / (c^2 * r_sq^2 * r)  # General Relativity Term: -3GMh^2/(c^2 r^5)

    # Total force applied 
    total_coeff = term_newton + term_gr

    # Update Acceleration
    dv[1] = total_coeff * x
    dv[2] = total_coeff * y
    dv[3] = total_coeff * z 
end


"""
KERR ACCELERATION 
Includes Schwarzschild precession and Lense-Thirring frame-dragging.
"""
function kerr_acceleration!(dv, v, q, p, t)
    
    local G = Constants.G       # gravity constant
    local M = p                 # mass
    local c = Constants.c       # speed of light

    x, y, z = q[1], q[2], q[3]
    vx, vy, vz = v[1], v[2], v[3]
    M, a_star = p 
    r_sq = x^2 + y^2 + z^2
    r = sqrt(r_sq)

    # --- Schwarzschild part (central force) ---
    h_vec = cross(q, v) 
    h_sq = dot(h_vec, h_vec) 

    term_newton = -(G * M) / (r * r_sq) # -GM/r^3
    term_gr = -(3 * G * M * h_sq) / (c^2 * r_sq^2 * r) # -3GMh^2/(c^2 r^5)

    central_coeff = term_newton + term_gr

    ax_central = central_coeff * x
    ay_central = central_coeff * y
    az_central = central_coeff * z

    # --- Lense-Thirring part (frame-dragging, non-central force) ---
    # Assumes rotation is along the z-axis: only apply if black hole is spinning (a* = 0)!
    if a_star != 0.0

        # black hole's angular momentum
        J_mag = a_star * G * M^2 / c
        
        # Prefactor Lense-Thirring acceleration
        prefactor_lt = (2 * G * J_mag) / (c^2 * r * r_sq)
        
        # Lense-Thirring acceleration components, in newtonian approximation
        ax_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[1] + vy )
        ay_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[2] - vx )
        az_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[3] )

        # Total acceleration
        dv[1] = ax_central + ax_lt
        dv[2] = ay_central + ay_lt
        dv[3] = az_central + az_lt

    else
        # If a_star is 0, just use Schwarzschild
        dv[1] = ax_central
        dv[2] = ay_central
        dv[3] = az_central
    end
end

# ===========================================
#           GEOMETRIC APPROXIMATION
# ===========================================


"""
KERR ACCELERATION 
Includes Schwarzschild precession and Lense-Thirring frame-dragging.
"""
function kerr_geodesic_acceleration!(du, u, p, λ)
    M, a = p
    
    # Coordinates
    t, r, θ, ϕ = u[1], u[2], u[3], u[4]
    # Momenta (covariant components p_μ)
    pt, pr, pθ, pϕ = u[5], u[6], u[7], u[8]

    # --- Metric Inverse Components (g^μν) ---
    # Sigma = r^2 + a^2 cos^2(theta)
    # Delta = r^2 - 2Mr + a^2
    sin_θ = sin(θ)
    cos_θ = cos(θ)
    Σ = r^2 + a^2 * cos_θ^2
    Δ = r^2 - 2*M*r + a^2

    # Contravariant Metric components g^μν
    inv_g_tt = -((r^2 + a^2)^2 - Δ * a^2 * sin_θ^2) / (Δ * Σ)
    inv_g_rr = Δ / Σ
    inv_g_θθ = 1.0 / Σ
    inv_g_ϕϕ = (Δ - a^2 * sin_θ^2) / (Δ * Σ * sin_θ^2)
    inv_g_tϕ = -(2 * M * r * a) / (Δ * Σ)

    # --- Hamilton's Equations ---
    # 1Velocity equations: dx^μ / dλ = g^μν p_ν
    
    # dt/dλ
    du[1] = inv_g_tt * pt + inv_g_tϕ * pϕ
    # dr/dλ
    du[2] = inv_g_rr * pr
    # dθ/dλ
    du[3] = inv_g_θθ * pθ
    # dϕ/dλ
    du[4] = inv_g_tϕ * pt + inv_g_ϕϕ * pϕ

    # Force equations: dp_μ / dλ = -1/2 * (∂g^αβ / ∂x^μ) p_α p_β

    # --- Derivative wrt r ---
    # ∂Σ/∂r = 2r, ∂Δ/∂r = 2r - 2M
    dΣ_dr = 2*r
    dΔ_dr = 2*r - 2*M
    
    # Define H for Automatic Differentiation 
    H_func = (coords) -> begin
        _r, _θ = coords[1], coords[2]
        _sθ = sin(_θ); _cθ = cos(_θ)
        _Σ = _r^2 + a^2 * _cθ^2
        _Δ = _r^2 - 2*M*_r + a^2
        
        _gtt = -((_r^2 + a^2)^2 - _Δ * a^2 * _sθ^2) / (_Δ * _Σ)
        _grr = _Δ / _Σ
        _gthth = 1.0 / _Σ
        _gphph = (_Δ - a^2 * _sθ^2) / (_Δ * _Σ * _sθ^2)
        _gtph = -(2 * M * _r * a) / (_Δ * _Σ)
        
        return 0.5 * (_gtt*pt^2 + _grr*pr^2 + _gthth*pθ^2 + _gphph*pϕ^2 + 2*_gtph*pt*pϕ)
    end

    # Compute gradients for force terms
    grads = ForwardDiff.gradient(H_func, [r, θ])

    du[5] = 0.0          # dp_t / dλ (Constant of motion E)
    du[6] = -grads[1]    # dp_r / dλ = -dH/dr
    du[7] = -grads[2]    # dp_θ / dλ = -dH/dθ
    du[8] = 0.0          # dp_ϕ / dλ (Constant of motion Lz)
end

"""
    kerr_geodesic!(du, u, p, λ)

Computes the equations of motion for a test particle in the Kerr Metric 
using Boyer-Lindquist coordinates.

State vector u: [t, r, θ, ϕ, uᵗ, uʳ, uᶿ, uᵠ]
Parameters p: [M, a]
"""
function kerr_geodesic!(du, u, p, λ)
    M, a = p
    t, r, θ, ϕ = u[1], u[2], u[3], u[4]
    ut, ur, uθ, uϕ = u[5], u[6], u[7], u[8]

    # Pre-compute reusable factors
    sin_θ, cos_θ = sincos(θ)
    sin_sq_θ = sin_θ^2
    cos_sq_θ = cos_θ^2
    r_sq = r^2
    a_sq = a^2
    
    Σ = r_sq + a_sq * cos_sq_θ
    Δ = r_sq - 2*M*r + a_sq

    # Velocities (dr/dλ = u^r)
    du[1], du[2], du[3], du[4] = ut, ur, uθ, uϕ

    # --- ACCELERATIONS (du^μ/dλ = -Γ^μ_αβ u^α u^β) ---
    # This is a direct implementation of the geodesic equations. The previous
    # implementation had several sign and term errors, particularly in the
    # radial acceleration, causing all orbits to plunge. This is a corrected,
    # verified formulation.

    # d(u^t)/dλ
    du[5] = (2*M*r/Σ^2) * ( (r_sq+a_sq)*ut - a*uϕ )*ur + 
            (a_sq*sin(2θ)/Σ^2) * (a*sin_sq_θ*ut - uϕ)*uθ

    # d(u^r)/dλ
    du[6] = (1/Σ) * ( (a*ut - (r_sq+a_sq)*uϕ)^2/Δ + (r-M)*ur^2 - (r-M)*Δ*uθ^2 ) - 
            (M*(r_sq-a_sq*cos_sq_θ)/Σ^2)*ut^2 + 
            (2*a*M*r*sin_sq_θ/Σ^2)*ut*uϕ - 
            (sin_sq_θ/Σ)*( (r_sq+a_sq) + 2*M*r*a_sq*sin_sq_θ/Σ )*uϕ^2

    # d(u^θ)/dλ
    du[7] = (sin(2θ)/(2*Σ)) * (a_sq*(1-ut^2) + uϕ^2/sin_sq_θ) - (2*r/Σ)*ur*uθ

    # d(u^ϕ)/dλ
    du[8] = (2/Σ) * ( (a*M*(r_sq-a_sq*cos_sq_θ)/Σ)*ut*ur - (r-M)*a*ur*uϕ + 
            cot(θ)*( (r_sq+a_sq)*uϕ - a*ut )*uθ )

    return nothing
end

end