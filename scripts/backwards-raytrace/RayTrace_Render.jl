projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using Plots
using Serialization # To load the data

# ==========================================
# 1. ARTISTIC CONTROLS (TWEAK THESE!)
# ==========================================
# How bright is the disk overall?
base_exposure = 2.5 

# How much does the redshift affect brightness? 
# (Higher = One side becomes invisible, Lower = flatter look)
beaming_strength = 4.0 

# Color Palette Logic
function get_pixel_color(r, redshift)
    # If r is -1, it's empty space
    if r < 0 
        return RGB(0.0, 0.0, 0.0) 
    end

    # Calculate Intensity
    beaming = clamp(redshift^beaming_strength, 0.05, 100.0) 
    intensity = base_exposure * beaming

    # --- COLOR GRADING ---
    
    # 1. High Blue-shift (Moving towards camera) -> White/Blue hot
    if redshift > 1.1
        return RGB(
            min(1.0, intensity * 0.8), 
            min(1.0, intensity * 0.9), 
            min(1.0, intensity)
        )
    
    # 2. Middle Range -> Orange/Gold
    elseif redshift > 0.8
        return RGB(
            min(1.0, intensity), 
            min(1.0, intensity * 0.6), 
            0.0
        )
        
    # 3. High Red-shift (Moving away) -> Deep Red
    else
        return RGB(
            min(1.0, intensity * 0.5), 
            0.0, 
            0.0
        )
    end
end

# ==========================================
# 2. LOAD & RENDER
# ==========================================
data_file = projectdir("scripts/backwards-raytrace", "blackhole_data.jls")

if !isfile(data_file)
    println("Error: Data file not found. Run 'RayTrace_Compute.jl' first!")
    exit()
end

println("Loading Data...")
raw_data = deserialize(data_file)
height, width = size(raw_data)

println("Rendering Image ($width x $height)...")
image_grid = fill(RGB(0.0, 0.0, 0.0), height, width)

for j in 1:height
    for i in 1:width
        r_val, g_val = raw_data[j, i]
        image_grid[j, i] = get_pixel_color(r_val, g_val)
    end
end

# ==========================================
# 3. SAVE IMAGE
# ==========================================
output_path = projectdir("scripts/backwards-raytrace", "final_render.png")

plot(image_grid, 
     axis=nothing, 
     border=:none, 
     size=(width, height), 
     aspect_ratio=:equal
)

savefig(output_path)
println("Image saved to: $output_path")