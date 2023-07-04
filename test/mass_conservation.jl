using Revise
using ODINN 
using Infiltrator
using Test 


function unit_mass_test(; H₀, B, A, t_sim, Δx, Δy, rtol=0.02, atol=1.0, distance_to_border=3, save_plot=false)

    # Get parameters for a simulation 
    parameters = Parameters(simulation=SimulationParameters(tspan=(0.0, t_sim),
                                                            use_MB=false,
                                                            use_iceflow=true,
                                                            multiprocessing=true,
                                                            workers=1),
                            physical=PhysicalParameters(A=A),
                            solver=SolverParameters(reltol=1e-12))

    model = Model(iceflow = SIA2Dmodel(parameters),
                  mass_balance = TImodel1(parameters; DDF=8.0/1000.0, acc_factor=1.0/1000.0),
                  machine_learning = NN(parameters))

    # Surface
    S = B + H₀
    nx, ny = size(H₀)

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

    # Optional plot
    if save_plot
        fig = Figure(resolution = (800, 800))

        Axis(fig[1, 1], title = "Initial Condition")
        heatmap!(H₀, colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[1, 2], title = "Final State")
        heatmap!(H₁, colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[2,1], title="Difference")
        heatmap!(H₁_pred .- H₀, colormap=Reverse(:balance), colorrange=(-10, 10))

        save("test/mass_conservation_test.png", fig)
    end

    # Initial total mass 
    mass₀ = sum(H₀)
    mass₁ = sum(H₁_pred)
    Δmass = mass₁ - mass₀
    # @show Δmass, Δmass / mass₀

    # No mass in the borders of the domain
    @test maximum(maximum([H₁_pred[2,:], H₁_pred[:,2]])) < 1.0e-7
    @test Δmass / mass₀ < rtol
end

function unit_mass_flatbed_test(; rtol, atol)
    for nx in 80:20:140
        ny = nx
        for shape in ["parabolic", "square"]
            for A in [4e-17, 8e-17]
                B = zeros((nx, ny))
                if shape == "parabolic"
                    H₀ = [ 0.5 * ( (nx/4)^2 - (i - nx/2)^2 - (j - ny/2)^2 ) for i in 1:nx, j in 1:ny]
                    H₀[H₀ .< 0.0] .= 0.0
                elseif shape == "square"
                    H₀ = zeros((nx,ny))
                    @views H₀[floor(Int,nx/3):floor(Int,2nx/3), floor(Int,ny/3):floor(Int,2ny/3)] .= 400
                end
                unit_mass_test(; H₀=H₀, B=B, A=A, t_sim=10.0, Δx=50, Δy=50, rtol=rtol, atol=atol, save_plot=false)
            end
        end
    end
end