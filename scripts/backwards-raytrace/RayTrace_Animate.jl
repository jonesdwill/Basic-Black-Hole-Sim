projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using Plots
using Serialization
using Printf

# ==========================================
# 1. ARTISTIC CONTROLS
# ==========================================
fps = 15
base_exposure = 3.0

# Swirl Settings
rotation_speed = 4.0   # How fast the gas spins
swirl_density = 12.0   # How many "arms" the spiral has

# Color Palette Logic
function get_pixel_color(r, redshift, phi, time_val)
    if r < 0 
        return RGB(0.0, 0.0, 0.0) 
    end

    # 1. Doppler Beaming
    beaming = clamp(redshift^4, 0.05, 50.0) 
    
    # 2. PROCEDURAL TEXTURE
    # noise = sin(Angle * Density - Time * Speed + Radius)
    noise = sin(phi * swirl_density - time_val * rotation_speed + r)
    
    # Map noise (-1 to 1) to contrast (0.5 to 1.5)
    texture_variation = 1.0 + 0.5 * noise
    
    intensity = base_exposure * beaming * texture_variation

    # 3. Fire Palette
    if redshift > 1.1
        return RGB(min(1.0, intensity*0.8), min(1.0, intensity*0.9), min(1.0, intensity))
    elseif redshift > 0.8
        return RGB(min(1.0, intensity), min(1.0, intensity*0.6), 0.0)
    else
        return RGB(min(1.0, intensity*0.5), 0.0, 0.0)
    end
end

# ==========================================
# 2. RENDER LOOP
# ==========================================
data_dir = projectdir("scripts/backwards-raytrace", "swirl_data_frames")
files = readdir(data_dir)
# Filter for .jls files and sort them to ensure order
frame_files = sort(filter(x -> endswith(x, ".jls"), files))

if isempty(frame_files)
    println("No data found! Run 'RayTrace_Swirl_Compute.jl' first.")
    exit()
end

println("Found $(length(frame_files)) frames. Rendering GIF...")

anim = @animate for (idx, file) in enumerate(frame_files)
    path = joinpath(data_dir, file)
    raw_data = deserialize(path)
    height, width = size(raw_data)
    
    image_grid = fill(RGB(0.0, 0.0, 0.0), height, width)
    
    # Calculate Time for this frame (0.0 to 1.0)
    progress = (idx - 1) / length(frame_files)
    current_time = progress * 2.0 # 2.0 = two full rotations of texture roughly

    for j in 1:height
        for i in 1:width
            r_val, g_val, phi_val = raw_data[j, i]
            image_grid[j, i] = get_pixel_color(r_val, g_val, phi_val, current_time)
        end
    end
    
    plot(image_grid, axis=nothing, border=:none, size=(width*2, height*2), aspect_ratio=:equal)
    print("\rCompositing Frame $idx / $(length(frame_files))")
end

output_path = projectdir("scripts/backwards-raytrace", "blackhole_swirl_final.gif")
gif(anim, output_path, fps=fps)
println("\nGIF saved to: $output_path")