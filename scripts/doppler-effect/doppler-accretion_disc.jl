using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_celestial
using DifferentialEquations: DiscreteCallback, ContinuousCallback, terminate!, CallbackSet
using Base.Threads
using Printf

# ==========================================
#       PHYSICS & REDSHIFT MATH
# ==========================================

"""
Returns the Keplerian angular velocity (Omega) and time-component of 4-velocity (u^t)
for a photon in circular equatorial orbit at radius r around a Kerr black hole.
"""
function disk_properties(r, M, a)
    # Keplerian Angular Velocity
    Omega = 1.0 / (r^1.5 + a)
    
    g_tt = -(1 - 2*M/r)
    g_tphi = -2*M*a/r
    g_phiphi = (r^2 + a^2 + 2*M*a^2/r)
    
    val = - (g_tt + 2*Omega*g_tphi + Omega^2*g_phiphi)
    
    if val <= 0
        return NaN, NaN # Unstable or invalid orbit
    end
    
    u_t = 1.0 / sqrt(val)
    return Omega, u_t
end

"""
Calculates redshift factor g = E_obs / E_emit.
g > 1 : Blueshift (Brighter/Blue). High energy photons.
g < 1 : Redshift (Dimmer/Red). Low energy photons.
"""
function calculate_redshift(r, lambda, M, a)
    Omega, u_t = disk_properties(r, M, a)
    if isnan(u_t) return 0.0 end
    
    E_emit = u_t * (1.0 - Omega * lambda)
    return 1.0 / E_emit
end

# ==========================================
#                CONFIG  
# ==========================================

M = 1.0
a_star = 0.99
solver_params = (M, a_star) 
a = a_star * M
bh_params = get_black_hole_parameters(solver_params)

# Observer
r_obs = 1000.0          # Far observer
theta_obs = 85 * π/180  # Edge-on view (not directly side on)

# Accretion Disk Dimensions
r_isco = 1.0 + sqrt(1.0 - a_star^2) # Approximate ISCO
r_disk_min = 0.0      # Horizon
r_disk_max = 20.0     # Outer edge of disk

# Resolution
screen_width = 25.0
screen_height = 12.0
res_x = 300
res_y = 150

alpha_range = range(-screen_width/2, screen_width/2, length=res_x)
beta_range = range(-screen_height/2, screen_height/2, length=res_y)

# Store results as Redshift, Intensity
image_data = zeros(res_y, res_x) 

# ==========================================
#               CALLBACKS  
# ==========================================

# Horizon Collision Callback
condition_horizon(u, t, integrator) = u[2] - (bh_params.rh * 1.01)
affect_horizon!(integrator) = terminate!(integrator)
cb_horizon = DiscreteCallback((u,t,int)->u[2] < bh_params.rh * 1.01, affect_horizon!)

# Stop if we cross the equator z=0 (accretion disc builds up)
function condition_disk(u, t, integrator)
    return u[3] - π/2
end

function affect_disk!(integrator)

    # Check if within the radial bounds of the disk
    r = integrator.u[2]
    if r > r_disk_min && r < r_disk_max && r > bh_params.rh
        terminate!(integrator)

    end
end
cb_disk = ContinuousCallback(condition_disk, affect_disk!)

callbacks = CallbackSet(cb_horizon, cb_disk)


# ==========================================
#        REDSHIFT RAY TRACING LOOP
# ==========================================

println("Rendering Redshift Disk (a*=$(a_star))...")

counter = Atomic{Int}(0)

# loop for each ray (pixel) on a grid of an observer's screen. Fire photon at black hole and measure energy shift. Low energy -> redshift, high energy -> blueshift.
@threads for i in 1:res_x
    for j in 1:res_y
        alpha = alpha_range[i]
        beta = beta_range[j]

        # place photon ray away from black hole
        u0 = get_initial_photon_state_celestial(alpha, beta, r_obs, theta_obs, M, a)
        
        # redshift impact parameter lambda is approximately -alpha * sin(theta_obs) (for an observer at infinity).
        lambda = -alpha * sin(theta_obs)

        # problem set up & solve
        prob = setup_problem(:kerr_geodesic_acceleration, u0, (0.0, 2000.0), solver_params)
        sol = solve_orbit(prob, callback=callbacks, reltol=1e-6, abstol=1e-6, save_everystep=false)

        final_u = sol.u[end]
        r_final = final_u[2]
        theta_final = final_u[3]

        # --- COLOURING LOGIC ---
        # CASE A: Hit the Black Hole (Horizon)
        if r_final < bh_params.rh * 1.1
            image_data[j, i] = NaN # shadow 
            
        # CASE B: Hit the Disk (Theta is approx π/2)
        elseif abs(theta_final - π/2) < 0.05 && r_final < r_disk_max
            g = calculate_redshift(r_final, lambda, M, a) # colour by red/blue shift
            image_data[j, i] = g
            
        # CASE C: Hits the 'sky'
        else
            # Assign black for sky
            image_data[j, i] = 0.0
        end
    end
    atomic_add!(counter, 1)
    if i % 10 == 0 print("\rProgress: $(counter[]) / $res_x") end
end

println("\nDone.")

# ==========================================
#               PLOTTING
# ==========================================

# Low Energy (g < 1)  -> Red
# High Energy (g > 1) -> Blue
heatmap(alpha_range, beta_range, image_data,
        c = reverse(cgrad(:seismic)),  
        clims = (0.4, 1.6),
        aspect_ratio = :equal,
        bg = :black,
        title = "Accretion Disk Doppler Shift",
        xlabel = "α", ylabel = "β")

output_path = projectdir("scripts/doppler-effect", "kerr_doppler.png")
savefig(output_path)
println("Saved to $output_path")