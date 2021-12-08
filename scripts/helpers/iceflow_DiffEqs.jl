###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

using Zygote
using Flux
using Flux: @epochs
using Tullio
include("utils.jl")

# Patch suggested by Michael Abbott needed in order to correctly retrieve gradients
Flux.Optimise.update!(opt, x::AbstractMatrix, Δ::AbstractVector) = Flux.Optimise.update!(opt, x, reshape(Δ, size(x)))

"""
    generate_ref_dataset(temp_series, gref, H₀, t)

Training of reference dataset with multiple glaciers forced with 
different temperature series.
"""
function generate_ref_dataset(temp_series, H₀, t)
    
    # Compute reference dataset in parallel
    # TODO: Add @everything macros to run this in parallel
    iceflow_prob = map(temps -> ref_glacier(temps, H₀, t), temp_series)
    
    return iceflow_prob
    
end


"""
    ref_glacier(temps, gref, H₀, t)

Training reference dataset of a single glacier
"""
function ref_glacier(temps, H₀, t)
      
    tempn = mean(temps)
    println("Reference simulation with temp ≈ ", tempn)
    H = deepcopy(H₀)
    
    # Initialize all matrices for the solver
    S, dSdx, dSdy = zeros(Float32,nx,ny),zeros(Float32,nx-1,ny),zeros(Float32,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1),zeros(Float32,nx-1,ny-1)
    D, dH, Fx, Fy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-2,ny-2),zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1)
    V, Vx, Vy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1)
    
    # Gather simulation parameters
    current_year = 0
    p = [A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year]

    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    iceflow_prob = ODEProblem(iceflow!,H,(0.0,t₁),p)
    #@time solve(iceflow_prob, alg_hints=[:stiff], reltol=1e-14,abstol=1e-14, progress=true, progress_steps = 1)
    #@time iceflow_sol = solve(iceflow_prob, Vern7(), progress=true, progress_steps = 1)
    iceflow_sol = solve(iceflow_prob, BS3(), progress=true, saveat=1.0, progress_steps = 1)
    #@time iceflow_sol = solve(iceflow_prob, KenCarp4(autodiff=false), reltol=1e-8,abstol=1e-8, progress=true, progress_steps = 1)

    return iceflow_sol[end]
    
end

"""
    train_batch_iceflow_UDE!(H₀, UA, glacier_refs, temp_series, hyparams, idx, p)

Training of a batch for the iceflow UDE based on the SIA.
"""
function train_batch_iceflow_UDE(H₀, UA, H_refs, temp_series, hyparams, idxs)
    
    # Train UDE batch in parallel
    iceflow_trained = map(idx -> train_iceflow_UDE(H₀, UA, H_refs, temp_series, hyparams, idx), idxs) 
    
    return iceflow_trained
    
end


function train_iceflow_UDE(H₀, UA, H_refs, temp_series, hyparams, idx)
    
    # Gather simulation parameters
    H = deepcopy(H₀)
    temps = temp_series[idx]
    norm_temps = norm_temp_series[idx]
    H_ref = H_refs[idx]
    println("\nTemperature in training: ", temps[1])
    
    # Initialize all matrices for the solver
    S, dSdx, dSdy = zeros(Float32,nx,ny),zeros(Float32,nx-1,ny),zeros(Float32,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1),zeros(Float32,nx-1,ny-1)
    D, dH, Fx, Fy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-2,ny-2),zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1)
    V, Vx, Vy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1)
    
    # Gather simulation parameters
    current_year = 0
    θ = initial_params(UA)
    context = [A, B, S, dSdx, dSdy, D, norm_temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year, H_ref, H, UA, θ]

    loss(θ) = loss_iceflow(UA, θ, H, context) # closure

    iceflow_trained = DiffEqFlux.sciml_train(loss, θ, RMSProp(hyparams.η), cb=callback, maxiters = 10)
    
    predicted_A = predict_A̅(UA, θ, [mean(norm_temps)]')[1]
    fake_A = A_fake(mean(temps)) 
    A_error = predicted_A - fake_A
    println("Predicted A: ", predicted_A)
    println("Fake A: ", fake_A)
    println("A error: ", A_error)

    return iceflow_trained
    
end


"""
    update_UDE_batch!(UA, back_UAs)

Update neural network weights after having trained on a whole batch of glaciers. 
"""
function update_UDE_batch!(UA, loss_UAs, back_UAs)
    
    println("Backpropagation...")
    # We update the weights with the gradients of all tha glaciers in the batch
    # This is equivalent to taking the gradient with respect of the full loss function
    θ = Flux.params(UA)
    
    for back_UA in back_UAs
        ∇_UA = back_UA(1)
        println("#$i Updating NN weights")
        Flux.Optimise.update!(opt, θ, ∇_UA) # with UA
    end

    #∇_UA = back_UA(one(mean(trackers["losses_batch"]))) # with UA
    #println("Updating NN weights")
    #Flux.Optimise.update!(opt, θ, ∇_UA) # with UA

    # Keep track of the loss function per batch
    println("Loss batch: ", mean(loss_UAs))  

end


"""
    plot_training!(UA, old_trained, temp_values, norm_temp_values)

Plot evolution of the functions learnt by the UDE. 
"""
function plot_training!(old_trained, UA, loss_UAs, temp_values, norm_temp_values)
    
    # Plot progress of the loss function 
    # temp_values = LinRange(-25, 0, 20)'
    # plot(temp_values', A_fake.(temp_values)', label="Fake A")
    # pfunc = scatter!(temp_values', predict_A̅(UA, temp_values)', yaxis="A", xaxis="Air temperature (°C)", label="Trained NN", color="red")
    # ploss = plot(trackers["losses"], title="Loss", xlabel="Epoch", aspect=:equal)
    # display(plot(pfunc, ploss, layout=(2,1)))
    
    # Plot the evolution
    plot(temp_values', A_fake.(temp_values)', label="Fake A")
    scatter!(temp_values', predict_A̅(UA, θ, norm_temp_values)', yaxis="A", xaxis="Air temperature (°C)", label="Trained NN", color="red")#, ylims=(3e-17,8e-16)))
    pfunc = scatter!(temp_values', old_trained, label="Previous NN", color="grey", aspect=:equal, legend=:outertopright)#, ylims=(3e-17,8e-16)))
    ploss = plot(loss_UAs, xlabel="Epoch", ylabel="Loss", aspect=:equal, legend=:outertopright, label="")
    ptrain = plot(pfunc, ploss, layout=(2,1))

    savefig(ptrain,joinpath(root_dir,"plots/training","epoch$i.png"))
    #if x11 
    #    display(ptrain) 
    #end

    old_trained = predict_A̅(UA, θ, norm_temp_values)'
    
end


"""
    loss(H, glacier_ref, UA, p, t, t₁)

Computes the loss function for a specific batch
"""
# We determine the loss function
function loss_iceflow(UA, θ, H, context)
    
    H = predict_iceflow(UA, θ, H, context)
    
    H_ref = @view context[19][:,:]
    l_H = sqrt(Flux.Losses.mse(H[H .!= 0.0], H_ref[H.!= 0.0]; agg=sum))
    
    # l_V = sqrt(Flux.Losses.mse(V̂[V̂ .!= 0.0], mean(glacier_ref["V"])[V̂ .!= 0.0]; agg=sum))

    println("l_H: ", l_H)
    # println("l_V: ", l_V)

    # Zygote.ignore() do
    # #    hml = heatmap(mean(glacier_ref["V"]) .- V̂, title="Loss error - V")
    #    hml = heatmap(glacier_ref["H"][end] .- H, title="Loss error - H")
    #    display(hml)
    # end

    return l_H
end

function predict_iceflow(UA, θ, H, context)
        
    #H = @view u0[20][:,:]
    #θ = p[end]
    
    iceflow_UDE!(dH, H, θ, t) = iceflow_NN!(dH, H, θ, t, context) # closure
    tspan = (0.0,t₁)
    iceflow_prob = ODEProblem(iceflow_UDE!,H,tspan,θ)
    #@infiltrate
    H_pred = solve(iceflow_prob, BS3(), u0=H, p=θ, saveat=1.0, 
                   sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP()), 
                   progress=true, progress_steps = 1)
    
    #prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p)
    #Array(solve(prob_nn, Vern7(), u0 = X, p=θ,
    #            tspan = (T[1], T[end]), saveat = T,
    #            abstol=1e-6, reltol=1e-6,
    #            sensealg = ForwardDiffSensitivity()
    #            ))
    
    return H_pred[end]
end


"""
    iceflow!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow!(dH, H, p,t)

    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = @view p[18]
    A = @view p[1]
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year && year <= t₁

        #println("Year: ", year)
        #println("current_year before: ", current_year)
        #println("current_year before p: ", p[end])
        
        # Predict A with the fake A law
        #println("temps: ", temps)
        temp = @view p[7][year]
        A .= A_fake(temp)
        #println("A fake: ", p[1])

        # Unpack and repack tuple with updated A value
        current_year .= year
        #println("current_year after: ", current_year)

    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, p)

end  


predict_A̅(UA, θ, temp) = UA(temp, θ)[1] .* 1e-16


"""
    iceflow_UDE!(dH, UA, H, p,t)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow_NN!(dH, H, θ, t, context)
    
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, norm_temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year, H_ref, H, UA, θ
    current_year = context[18]
    A = Ref{Float32}(context[1])
    UA = context[21]
    dH = @view context[20][:,:]
    #θ = p[22]
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year && year <= t₁

        #println("Year: ", year)
        #println("current_year before: ", current_year)
        #println("current_year before p: ", p[end])
        
        # Predict A with the fake A law
        #println("temps: ", temps)
        
        temp = context[7][year]
        #temp = @view p[7][year]
 
        #@infiltrate
        context[1] = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters
        println("Predicted A: ", A[])
        println("context[1]: ", context[1])

        # Unpack and repack tuple with updated A value
        current_year = year
        #println("current_year after: ", current_year)

    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)

end  

"""
    SIA(H, p)

Compute a step of the Shallow Ice Approximation PDE in a forward model
"""

function SIA!(dH, H, context)
    
    # Retrieve parameters
    #A, B, S, dSdx, dSdy, D, norm_temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year, H_ref, H, UA, θ
    
    A = Ref{Float32}(context[1])
    B = @view context[2][:,:]
    S = @view context[3][:,:]
    dSdx = @view context[4][:,:]
    dSdy = @view context[5][:,:]
    D = @view context[6][:,:]
    dSdx_edges = @view context[8][:,:]
    dSdy_edges = @view context[9][:,:]
    ∇S = @view context[10][:,:]
    Fx = @view context[11][:,:]
    Fy = @view context[12][:,:]
    Vx = @view context[13][:,:]
    Vy = @view context[14][:,:]
    
    #@infiltrate
    
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    println("A in Γ: ", A[])
    Γ = 2 * A[] * (ρ * g)^n / (n+2) # 1 / m^3 s 
    
    D = Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
    dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 
    #@tullio dH[i,j] = -(diff(Fx, dims=1)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff(Fy, dims=2)[pad(i-1,1,1),pad(j-1,1,1)] / Δy)

    # Compute velocities    
    Vx = -D./(avg(H) .+ ϵ).*avg_y(dSdx)
    Vy = -D./(avg(H) .+ ϵ).*avg_x(dSdy)
    
    #@tullio dH[i,j] = H[i,j] + F[pad(i-1,1,1),pad(j-1,1,1)]
    #dH = H .+ F
    
end

"""
    C_fake(MB, ∇S)

Fake law to determine the sliding rate factor 
"""
# TODO: to be updated in order to make it depend on the surrounding
# ice surface velocity pattern
function C_fake(MB, ∇S)
    MB[MB .> 0] .= 0
    MB[MB .< -20] .= 0
    #println("∇S max: ", maximum(∇S))
    #println("((MB).^2)/4) max: ", maximum(((MB).^2)/4))

    return ((avg(MB).^2)./6) .* 3e-13
    #return ((avg(MB).^2)./6) 

end

"""
    A_fake(ELA)

Fake law to determine A in the SIA
"""
function A_fake(temp)
    # Matching point MB values to A values

    #temp_range = -25:0.01:1

    #A_step = (maxA-minA)/length(temp_range)
    #A_range = sigmoid.(Flux.normalise(minA:A_step:maxA).*1.5e14).*1.5e-18 # add nonlinear relationship

    #A = A_range[closest_index(temp_range, temp)]

    return @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    #return A
end

# function A_fake(MB_buffer, shape, var_format)
#     # Matching point MB values to A values
#     maxA = 3e-16
#     minA = 1e-17

#     if var_format == "matrix"
#         MB_range = reverse(-15:0.01:8)
#     elseif var_format == "scalar"
#         MB_range = reverse(-3:0.01:0)
#     end

#     A_step = (maxA-minA)/length(MB_range)
#     A_range = sigmoid.(Flux.normalise(minA:A_step:maxA).*2.5f11).*5f-16 # add nonlinear relationship

#     if var_format == "matrix"
#         A = []
#         for MB_i in MB_buffer
#             push!(A, A_range[closest_index(MB_range, MB_i)])
#         end
#         A = reshape(A, shape)
#     elseif var_format == "scalar"
#         A = A_range[closest_index(MB_range, nanmean(MB_buffer))]
#     end

#     return A
# end

"""
    create_NNs()

Generates the hyperaparameters and the neural networks needed for the training of UDEs
"""
function create_NNs()
    ######### Define the network  ############
    # We determine the hyperameters for the training
    hyparams = Hyperparameters()

    # Leaky ReLu as activation function
    leakyrelu(x, a=0.01) = max(a*x, x)

    # Constraints A within physically plausible values
    minA = 0.3
    maxA = 8
    rangeA = minA:1e-3:maxA
    stdA = std(rangeA)*2
    relu_A(x) = min(max(minA, x), maxA)
    #relu_A(x) = min(max(minA, 0.00001 * x), maxA)
    sigmoid_A(x) = minA + (maxA - minA) / ( 1 + exp(-x) )

    A_init(custom_std, dims...) = randn(Float32, dims...) .* custom_std
    A_init(custom_std) = (dims...) -> A_init(custom_std, dims...)

    #UA = Chain(
    #    Dense(1,10), 
    #    #Dense(10,10, x->tanh.(x), init = A_init(stdA)), 
    #    Dense(10,10, x->tanh.(x)), #init = A_init(stdA)), 
    #    #Dense(10,5, x->tanh.(x), init = A_init(stdA)), 
    #    Dense(10,5, x->tanh.(x)), #init = A_init(stdA)), 
    #    Dense(5,1, sigmoid_A)
    #)

    UA = FastChain(
        FastDense(1,3, x->tanh.(x)),
        FastDense(3,10, x->tanh.(x)),
        FastDense(10,3, x->tanh.(x)),
        FastDense(3,1, sigmoid_A)
    )

    return hyparams, UA
end

"""
    callback(l)

Callback to track evolution of the neural network's training. 
"""
# Callback to show the loss during training
callback(l) = begin
    # Container to track the losses
    losses = Float64[]
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end
