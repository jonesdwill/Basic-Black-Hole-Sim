using DifferentialEquations
using LinearAlgebra
using Plots
using Base.Threads 
using BasicBlackHoleSim
using BasicBlackHoleSim.Physics
using BasicBlackHoleSim.Utils


"""
Renders a high-quality image of the accretion disk using Reverse Ray Tracing.
"""
function render_accretion_disk(M, a_star, resolution=100; fov=15.0)
    
    # cameragrid (with an alpha-beta screen)
    println("Setting up render grid ($resolution x $resolution)...")
    alpha_vals = range(-fov, fov, length=resolution)
    beta_vals  = range(-fov, fov, length=resolution)
    image = zeros(RGB, resolution, resolution)

    # parameters for physics
    r_isco = 1.0 + sqrt(1.0 - a_star^2) # ~ ISCO 
    r_outer = 12.0
    
    # callback as accretion disc 
    function disk_condition(u, t, integrator)
        return u[3] - π/2 # (this is the disc) 
    end

    # stop ray 
    function disk_affect!(integrator)
        terminate!(integrator)
    end
    
    # create callback 
    disk_cb = ContinuousCallback(disk_condition, disk_affect!)
    
    # Main render loop (threaded for CPU cores)
    @threads for i in 1:resolution
        for j in 1:resolution
            alpha = alpha_vals[j] # Col
            beta = beta_vals[i]   # Row

            # places observer at r=1000, theta=85 degrees (nearly edge on). initialise photon for current pixel. 
            u0 = get_initial_photon_state_celestial(alpha, beta, 1000.0, deg2rad(85), M, a_star * M)
            
            if any(isnan, u0)
                image[i, j] = RGB(0.0, 0.0, 0.0)
                continue
            end

            # bypassing Solvers.jl as using custom callback here
            tspan = (0.0, 2500.0)
            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, tspan, (M, a_star * M))
            
            # Solve with high precision
            sol = solve(prob, Vern7(), reltol=1e-6, abstol=1e-6, callback=disk_cb, save_everystep=false)

            # colour
            final_u = sol[end]
            r_hit = final_u[2]
            theta_hit = final_u[3]
            
            # Horizon check
            if r_hit < (1.0 + sqrt(1.0 - a_star^2)) * M * 1.05
                image[i, j] = RGB(0.0, 0.0, 0.0) # Shadow
                
            # Check if we hit the Disk Plane (within tolerances) to colour 
            elseif abs(theta_hit - π/2) < 0.05 && r_hit > r_isco && r_hit < r_outer
                
                # Calculate Redshift 
                ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, M, a_star * M)
                
                # Photon Momentum (k_μ)
                E_obs = -final_u[5]                                         # Energy at camera (infinity)
                E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)   # Energy at disc frame
                
                g = E_obs / E_emit
                
                # Intensity scaling for better plotting 
                intensity = g^4
                
                # ---- TEXTURING ----
                # playing around with some texture options here to see what looks good

                # # # block plasma colour 
                # base_color = RGB(1.0, 0.9, 0.7)

                # # ring texture 
                # base_color = sin(20 * r_hit) > 0 ? RGB(1.0, 0.8, 0.6) : RGB(0.8, 0.4, 0.2)

                # # temperature proxy 
                # t_proxy = 1.0 / sqrt(r_hit) 
                # base_color = RGB(t_proxy, t_proxy * 0.6, t_proxy * 0.2)

                # # soft rings 
                # factor = 0.5 + 0.5 * sin(20 * r_hit) 
                # color_bright = RGB(1.0, 0.8, 0.6)
                # color_dark = RGB(0.8, 0.4, 0.2)
                # base_color = color_bright * factor + color_dark * (1.0 - factor)

                # temperature gradient with soft rings
                r_inner = r_isco 
                T = (r_inner / r_hit)^1.5
                T = clamp(T, 0.0, 1.0) 
                col_hot = RGB(1.0, 0.85, 0.5) 
                col_cold = RGB(0.5, 0.1, 0.0) 
                base_gradient = T * col_hot + (1.0 - T) * col_cold
                r_ring_pos = r_inner + 0.25  
                ring_width = 0.1             
                ring_boost = 3.0             
                ring_shape = exp(-((r_hit - r_ring_pos)^2) / (2 * ring_width^2)) # gaussian 
                col_ring = RGB(1.0, 0.95, 0.9) 
                added_ring_light = ring_shape * ring_boost * col_ring
                base_color = base_gradient + added_ring_light
                
                image[i, j] = base_color * intensity
            else

                # Ray escaped to infinity 
                image[i, j] = RGB(0.0, 0.0, 0.0) # colour black 
            end
        end
    end
    
    return image
end

# ==========================================
#                Main Call
# ==========================================

# M=1.0, a*=0.99, resolution=200
img = render_accretion_disk(1.0, 0.99, 200)

# black background, no axes or borders
plot(img, axis=nothing, border=:none, background_color=:black, margin=0Plots.mm)

# output to render folder
mkpath("scripts/render") 
savefig("scripts/render/accretion_disk_render_gaussian.png")