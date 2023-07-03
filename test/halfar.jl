using Revise
using ODINN 
using Infiltrator
using Test 
using CairoMakie

# Include utils for halfar 
include("./utils_test.jl")

# Default parameters in ODINN
F = parameters.simulation.float_type
I = parameters.simulation.int_type
ρ = parameters.physical.ρ   
g = parameters.physical.g
n = parameters.physical.n

# Bed (it has to be flat for the Halfar solution)
B = zeros((nx,ny))

function unit_halfar_test(; A, t₀, t₁, Δx, Δy, nx, ny, h₀, r₀, rtol=0.2, atol=1.0, distance_to_border=3)

    # Get parameters for a simulation 
    parameters = Parameters(simulation=SimulationParameters(tspan=(t₀, t₁),
                                                            use_MB=false,
                                                            use_iceflow=true,
                                                            multiprocessing=true,
                                                            workers=1),
                                                            physical=PhysicalParameters(A=A),
                                                            solver=SolverParameters(reltol=1e-12))

    model = Model(iceflow = SIA2Dmodel(parameters),
                  mass_balance = TImodel1(parameters; DDF=8.0/1000.0, acc_factor=1.0/1000.0),
                  machine_learning = NN(parameters))

    # Initial condition of the glacier
    R₀ = [sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2) for i in 1:nx, j in 1:ny]
    ν = (A, h₀, r₀)
    H₀ = halfar_solution.(R₀, Ref(t₀), Ref(ν))
    S = B + H₀

    # Define synthetic glacier shaper following Halfar solutions
    glacier = ODINN.Glacier{F,I}(rgi_id = "toy", H₀ = H₀, S = S, B = B, Δx=Δx, Δy=Δy, nx=nx, ny=ny)

    glaciers = Vector{ODINN.Glacier}([glacier])

    prediction = Prediction(model, glaciers, parameters)

    run!(prediction) 

    # Final solution
    H₁ = halfar_solution.(R₀, Ref(t₁), Ref(ν))
    H₁_pred = prediction.results[1].H[end]
    H_diff = H₁_pred .- H₁

    absolute_error = maximum(abs.(H_diff[is_border(H₁, distance_to_border)])) 
    percentage_error = maximum(abs.((H_diff./H₁)[is_border(H₁, distance_to_border)])) 
    maximum_flow = maximum(abs.(((H₁ .- H₀))[is_border(H₁, distance_to_border)])) 
    
    @show percentage_error, absolute_error, maximum_flow
    @test all([percentage_error < rtol, absolute_error < atol])
end

function halfar_test(; rtol, atol)
    unit_halfar_test(A=4e-17, t₀=5.0, t₁=10.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=1000, rtol=rtol, atol=atol)
    unit_halfar_test(A=8e-17, t₀=5.0, t₁=10.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=1000, rtol=rtol, atol=atol)
    unit_halfar_test(A=4e-17, t₀=5.0, t₁=40.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=600, rtol=rtol, atol=atol)
    unit_halfar_test(A=8e-17, t₀=5.0, t₁=40.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=600, rtol=rtol, atol=atol)
    unit_halfar_test(A=4e-17, t₀=5.0, t₁=100.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=600, rtol=rtol, atol=atol)
    unit_halfar_test(A=8e-17, t₀=5.0, t₁=100.0, Δx=50, Δy=50, nx=100, ny=100, h₀=500, r₀=600, rtol=rtol, atol=atol)
    unit_halfar_test(A=4e-17, t₀=5.0, t₁=40.0, Δx=80, Δy=80, nx=100, ny=100, h₀=300, r₀=1000, rtol=rtol, atol=atol)
    unit_halfar_test(A=8e-17, t₀=5.0, t₁=40.0, Δx=80, Δy=80, nx=100, ny=100, h₀=300, r₀=1000, rtol=rtol, atol=atol)
end


# f = Figure(resolution = (800, 800))

# Axis(f[1, 1], title = "Initial Condition")
# heatmap!(H₀, colormap=:viridis, colorrange=(0, maximum(H₀)))

# Axis(f[1, 2], title = "Final State")
# heatmap!(H₁, colormap=:viridis, colorrange=(0, maximum(H₀)))

# Axis(f[2,1], title="Prediction")
# heatmap!(H₁_pred, colormap=:viridis, colorrange=(0, maximum(H₀)))

# Axis(f[2,2], title="Difference")
# heatmap!(H_diff, colormap=Reverse(:balance), colorrange=(-10, 10))

# save("test/halfar_test.png", f)

