module Physics

using ..Constants 
using LinearAlgebra 

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
Full geodesic equations for the Kerr metric.
Solves the 8-component state vector u = [t, r, theta, phi, ut, ur, utheta, uphi]
"""
function kerr_geodesic!(du, u, p, λ)

    M, a = p
    t, r, theta, phi, ut, ur, utheta, uphi = u

    # Helper variables
    a2 = a^2
    r2 = r^2
    sintheta = sin(theta)
    costheta = cos(theta)
    sin2theta = sintheta^2
    cos2theta = costheta^2

    sigma = r2 + a2*cos2theta
    delta = r2 - 2*M*r + a2

    # --- First 4 derivatives are just the 4-velocities ---
    du[1] = ut
    du[2] = ur
    du[3] = utheta
    du[4] = uphi
    
    # --- 4-accelerations d(u^μ)/dλ ---
    # Based on a standard implementation, e.g., Gama, et al. (2019), arXiv:1901.07577, Appendix A
    
    # d(u^r)/dλ
    du[6] = ( (r-M)/sigma ) * ( -delta*utheta^2 + ur^2 ) + (delta/sigma) * ( uphi^2 - a2*sin2theta*ut^2 ) - r*utheta^2

    # d(u^theta)/dλ
    du[7] = ( sin(2*theta)/sigma ) * ( a2*ut^2 - uphi^2/sin2theta ) - (2*r/sigma) * ur * utheta

    # d(u^t)/dλ
    du[5] = (2*M*r/sigma) * ( (r2+a2)*ut - a*uphi ) * ur + (a2*sin(2*theta)/sigma) * ( uphi - a*sin2theta*ut ) * utheta

    # d(u^phi)/dλ
    du[8] = (2*M*r/sigma) * ( a*ut - uphi ) * ur + (cot(theta)/sigma) * ( (r2+a2)^2*uphi - a*(r2+a2)*ut - a2*delta*sin2theta*uphi + a^3*sin2theta*ut ) * utheta

    return nothing
end

end