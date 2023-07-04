using Revise
using ODINN 
using Infiltrator
using Test 
using CairoMakie

# Include utils for halfar 
include("./utils_test.jl")

function unit_halfar_test(; A, t₀, t₁, Δx, Δy, nx, ny, h₀, r₀, rtol=0.02, atol=1.0, distance_to_border=3, save_plot=false)

    # Get parameters for a simulation 
    parameters = Parameters(simulation=SimulationParameters(tspan=(t₀, t₁),
                                                            use_MB=false,
                                                            use_iceflow=true,
                                                            multiprocessing=true,
                                                            workers=1),
                            physical=PhysicalParameters(A=A),
                            solver=SolverParameters(reltol=1e-12))

    # Bed (it has to be flat for the Halfar solution)
    B = zeros((nx,ny))

    model = Model(iceflow = SIA2Dmodel(parameters),
                  mass_balance = TImodel1(parameters; DDF=8.0/1000.0, acc_factor=1.0/1000.0),
                  machine_learning = NN(parameters))

    # Initial condition of the glacier
    R₀ = [sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2) for i in 1:nx, j in 1:ny]
    H₀ = halfar_solution(R₀, t₀, h₀, r₀, parameters.physical)
    S = B + H₀
    # Final expected solution
    H₁ = halfar_solution(R₀, t₁, h₀, r₀, parameters.physical)

    # Define glacier object
    glacier = ODINN.Glacier{parameters.simulation.float_type, 
                            parameters.simulation.int_type}(
                                rgi_id = "toy", H₀ = H₀, S = S, B = B, 
                                Δx=Δx, Δy=Δy, nx=nx, ny=ny)
    glaciers = Vector{ODINN.Glacier}([glacier])

    prediction = Prediction(model, glaciers, parameters)
    run!(prediction) 

    # Final solution
    H₁_pred = prediction.results[1].H[end]
    H_diff = H₁_pred .- H₁

    # Error calculation
    absolute_error = maximum(abs.(H_diff[is_border(H₁, distance_to_border)])) 
    percentage_error = maximum(abs.((H_diff./H₁)[is_border(H₁, distance_to_border)])) 
    maximum_flow = maximum(abs.(((H₁ .- H₀))[is_border(H₁, distance_to_border)])) 
    
    # Optional plot
    if save_plot
        fig = Figure(resolution = (800, 800))

        Axis(fig[1, 1], title = "Initial Condition")
        heatmap!(H₀, colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[1, 2], title = "Final State")
        heatmap!(H₁, colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[2,1], title="Prediction")
        heatmap!(H₁_pred, colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[2,2], title="Difference")
        heatmap!(H_diff, colormap=Reverse(:balance), colorrange=(-10, 10))

        save("test/halfar_test.png", fig)
    end

    # @show percentage_error, absolute_error, maximum_flow
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
    unit_halfar_test(A=4e-17, t₀=5.0, t₁=10.0, Δx=10, Δy=10, nx=500, ny=500, h₀=300, r₀=1000, rtol=0.02, atol=1.0)
end