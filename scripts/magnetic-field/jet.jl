using DifferentialEquations
using LinearAlgebra
using ForwardDiff
using Plots
using Random
using Printf

# Use GR backend for speed
gr()

# ==========================================
# 1. PHYSICS KERNEL (Unchanged)
# ==========================================

function get_wald_potential(r, theta, M, a, B0)
    sin_sq = sin(theta)^2
    Sigma = r^2 + a^2 * cos(theta)^2
    Delta = r^2 - 2*M*r + a^2

    g_tt = -(1.0 - 2.0 * M * r / Sigma)
    g_tphi = -(2.0 * M * r * a * sin_sq) / Sigma
    g_phiphi = sin_sq * ((r^2 + a^2)^2 - Delta * a^2 * sin_sq) / Sigma

    At = (B0 / 2.0) * (g_tphi + 2.0 * a * g_tt)
    Aphi = (B0 / 2.0) * (g_phiphi + 2.0 * a * g_tphi)
    
    return At, Aphi
end

function charged_hamiltonian(u, p_params)
    M, a, B0, q = p_params
    t, r, theta, phi = u[1], u[2], u[3], u[4]
    Pt, Pr, Ptheta, Pphi = u[5], u[6], u[7], u[8]

    sθ = sin(theta); cθ = cos(theta)
    sθ2 = sθ^2
    Σ = r^2 + a^2 * cθ^2
    Δ = r^2 - 2*M*r + a^2

    inv_g_tt = -((r^2 + a^2)^2 - Δ * a^2 * sθ2) / (Δ * Σ)
    inv_g_rr = Δ / Σ
    inv_g_thth = 1.0 / Σ
    inv_g_phph = (Δ - a^2 * sθ2) / (Δ * Σ * sθ2)
    inv_g_tph = -(2 * M * r * a) / (Δ * Σ)

    _g_tt = -(1 - 2*M*r / Σ)
    _g_tphi = -(2 * M * r * a * sθ2) / Σ
    _g_phiphi = sθ2 * ((r^2 + a^2)^2 - Δ * a^2 * sθ2) / Σ

    At = (B0 / 2) * (_g_tphi + 2*a*_g_tt)
    Aphi = (B0 / 2) * (_g_phiphi + 2*a*_g_tphi)

    pi_t = Pt - q * At
    pi_phi = Pphi - q * Aphi
    pi_r = Pr
    pi_th = Ptheta

    return 0.5 * (inv_g_tt*pi_t^2 + inv_g_rr*pi_r^2 + inv_g_thth*pi_th^2 + 
                  inv_g_phph*pi_phi^2 + 2*inv_g_tph*pi_t*pi_phi)
end

function equations_of_motion!(du, u, p, λ)
    grad_H = ForwardDiff.gradient(x -> charged_hamiltonian(x, p), u)
    du[1:4] = grad_H[5:8]
    du[5:8] = -grad_H[1:4]
end

# ==========================================
# 2. INITIALIZATION (Unchanged)
# ==========================================

function get_initial_state_general(M, a, B0, q, r0, theta0, phi0, ur, uphi)
    utheta = 0.0
    sθ = sin(theta0); cθ = cos(theta0)
    Σ = r0^2 + a^2 * cθ^2
    Δ = r0^2 - 2*M*r0 + a^2
    
    g_tt = -(1 - 2*M*r0 / Σ)
    g_tphi = -(2 * M * r0 * a * sθ^2) / Σ
    g_rr = Σ / Δ
    g_thth = Σ
    g_phiphi = sθ^2 * ((r0^2 + a^2)^2 - Δ * a^2 * sθ^2) / Σ

    A = g_tt
    B = 2 * g_tphi * uphi
    C = g_rr*ur^2 + g_thth*utheta^2 + g_phiphi*uphi^2 + 1.0
    
    discriminant = B^2 - 4*A*C
    if discriminant < 0 
        return zeros(8)
    end
    ut = (-B - sqrt(discriminant)) / (2*A)

    pt_kin = g_tt*ut + g_tphi*uphi
    pphi_kin = g_tphi*ut + g_phiphi*uphi
    pr_kin = g_rr*ur
    pth_kin = g_thth*utheta

    At, Aphi = get_wald_potential(r0, theta0, M, a, B0)
    Pt = pt_kin + q * At
    Pphi = pphi_kin + q * Aphi
    
    return [0.0, r0, theta0, phi0, Pt, pr_kin, pth_kin, Pphi]
end

# ==========================================
# 3. FILAMENT SIMULATION (Pre-computation)
# ==========================================

function simulate_filaments()
    M, a, B0, q = 1.0, 0.99, 0.5, -20.0
    params = (M, a, B0, q)

    n_strands = 24 
    base_r = 4.5
    base_theta_north = 0.3 
    
    ensemble_sols = []
    println("Simulating $n_strands Jet Filaments (Pre-calculation)...")

    for i in 1:n_strands
        is_north = i <= (n_strands / 2)
        phi_start = (i * (2π / 12)) + (is_north ? 0.0 : 0.5) 

        if is_north
            theta_start = base_theta_north + 0.02 * randn()
        else
            theta_start = (π - base_theta_north) + 0.02 * randn()
        end
        r_start = base_r + 0.2 * randn()
        
        uphi_kick = 0.6 
        ur_kick = 0.05 

        try
            u0 = get_initial_state_general(M, a, B0, q, r_start, theta_start, phi_start, ur_kick, uphi_kick)
            
            # Note: We simulate a long time, but we will animate based on proper time steps
            tspan = (0.0, 150.0) 
            prob = ODEProblem(equations_of_motion!, u0, tspan, params)
            
            rh = M + sqrt(M^2 - a^2)
            condition(u, t, int) = u[2] < rh * 1.05 || u[2] > 25.0
            cb = DiscreteCallback(condition, terminate!)
            
            # Dense output is enabled by default, allowing us to interpolate sol(t) later
            sol = solve(prob, Tsit5(), callback=cb, reltol=1e-6, abstol=1e-6)
            push!(ensemble_sols, sol)
        catch; end
    end
    return ensemble_sols, params
end

# ==========================================
# 4. ANIMATION GENERATOR (Fixed: Static Camera)
# ==========================================

# 1. Run Physics (Same as before)
sols, params = simulate_filaments()
M, a, B0, q = params
rh = M + sqrt(M^2 - a^2)

# 2. Pre-calculate Static Elements (Same as before)
println("Generating static geometry...")
u_full = range(0, 2π, length=30)
v_full = range(0, π, length=15)
hx_wire, hy_wire, hz_wire = Float64[], Float64[], Float64[]

function add_line!(lx, ly, lz, x_arr, y_arr, z_arr)
    append!(lx, x_arr); push!(lx, NaN)
    append!(ly, y_arr); push!(ly, NaN)
    append!(lz, z_arr); push!(lz, NaN)
end

for phi in range(0, 2π, length=12)
    cx = [sqrt(rh^2+a^2)*sin(v)*cos(phi) for v in v_full]
    cy = [sqrt(rh^2+a^2)*sin(v)*sin(phi) for v in v_full]
    cz = [rh*cos(v) for v in v_full]
    add_line!(hx_wire, hy_wire, hz_wire, cx, cy, cz)
end
for v in range(0, π, length=12)
    cx = [sqrt(rh^2+a^2)*sin(v)*cos(u) for u in u_full]
    cy = [sqrt(rh^2+a^2)*sin(v)*sin(u) for u in u_full]
    cz = [rh*cos(v) for u in u_full]
    add_line!(hx_wire, hy_wire, hz_wire, cx, cy, cz)
end

dx, dy, dz = Float64[], Float64[], Float64[]
r_range = range(rh*2.5, 12.0, length=50)
for _ in 1:1500
    r = rand(r_range)
    phi = rand() * 2π
    push!(dx, r * cos(phi))
    push!(dy, r * sin(phi))
    push!(dz, 0.0 + 0.3*randn()) 
end

# 3. Create Animation
println("Rendering Animation Frames...")

frames = 150
max_sim_time = 100.0
dt = max_sim_time / frames
tail_length = 15.0

anim = @animate for i in 1:frames
    current_time = i * dt
    
    # --- FIX IS HERE ---
    # We fix the camera at 45 degrees azimuth.
    # This prevents the camera from "chasing" the particles.
    static_cam_angle = 45 
    
    p = plot(legend=false, bg=:black, axis=false, grid=false,
             xlims=(-15,15), ylims=(-15,15), zlims=(-15, 15),
             camera=(static_cam_angle, 30), size=(800, 600)) # Elevated view (30)
    
    # Draw Static Elements
    scatter!(p, dx, dy, dz, markersize=1.2, color=:orange, alpha=0.4, markerstrokewidth=0)
    plot!(p, hx_wire, hy_wire, hz_wire, color=:white, alpha=0.2, lw=0.5)

    # Draw Dynamic Filaments
    for sol in sols
        if current_time > sol.t[1]
            t_end = min(current_time, sol.t[end])
            t_start = max(sol.t[1], t_end - tail_length)

            if t_end > t_start
                times = range(t_start, t_end, length=20)
                traj = [sol(t) for t in times]
                
                rs = [u[2] for u in traj]
                ths = [u[3] for u in traj]
                phs = [u[4] for u in traj]
                
                xs = @. sqrt(rs^2 + a^2) * sin(ths) * cos(phs)
                ys = @. sqrt(rs^2 + a^2) * sin(ths) * sin(phs)
                zs = @. rs * cos(ths)
                
                plot!(p, xs, ys, zs, lw=1.5, alpha=0.8, color=:cyan)
                scatter!(p, [xs[end]], [ys[end]], [zs[end]], 
                         marker=:circle, markersize=3, color=:white, markerstrokewidth=0)
            end
        end
    end
    
    if i % 10 == 0
        @printf("Frame %d/%d\n", i, frames)
    end

end every 1

println("Saving GIF...")
gif(anim, "black_hole_jet_fixed.gif", fps=30)