module Solvers

using DifferentialEquations
using LinearAlgebra
using RecursiveArrayTools
using ..Constants
using ..Physics
using ..Utils: get_black_hole_parameters # Import the new helper

export setup_problem, solve_orbit

function setup_problem(model_type::Symbol, u0, tspan, model_params)
    q0 = u0[1:3] 
    v0 = u0[4:6] 

    if model_type == :schwarzschild

        return DynamicalODEProblem(Physics.schwarzschild_acceleration!, Physics.velocity_law!, v0, q0, tspan, model_params)

    elseif model_type == :newtonian

        return DynamicalODEProblem(Physics.newtonian_acceleration!, Physics.velocity_law!, v0, q0, tspan, model_params)

    elseif model_type == :kerr_acceleration

        return DynamicalODEProblem(Physics.kerr_acceleration!, Physics.velocity_law!, v0, q0, tspan, model_params)

    elseif model_type == :kerr_geodesic_acceleration

        return ODEProblem(Physics.kerr_geodesic_acceleration!, u0, tspan, model_params)
    else
        error("Unknown model type.")
    end
end

function solve_orbit(prob; solver=Tsit5(), reltol=1e-8, abstol=1e-8, kwargs...)

    # Use the helper function to get model parameters
    params = get_black_hole_parameters(prob.p)

    function condition(u, t, integrator)
        local r
        if u isa Vector
            r = u[2] 
        else 
            pos = u.x[2]      
            r = norm(pos)
        end
        return r - params.rh 
    end

    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    return solve(prob, solver; reltol=reltol, abstol=abstol, callback=cb, kwargs...)
end

end