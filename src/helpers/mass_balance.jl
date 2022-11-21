export generate_random_MB

###############################################
############  FUNCTIONS   #####################
###############################################

function generate_random_MB(gdirs_climate, tspan; plot=true)
    random_MB = [] # tuple with RGI_ID, MB_max, MB_min
    gdirs = gdirs_climate[2]
    climates = gdirs_climate[3]
    for (gdir, climate) in zip(gdirs, climates)
        scaling = 1.0f0
        clim = scaling*(1.0f0/abs(mean(climate)))^(1.0f0/7.0f0) # climate driver to adapt random MB
        # clim = (clim <= 1.1)*clim # clip large values
        MB_max = Float32.((ref_max_MB .+ randn(MersenneTwister(1),floor(Int,tspan[2]-tspan[1]+1))).*clim)
        MB_min = Float32.((ref_min_MB .+ randn(MersenneTwister(2),floor(Int,tspan[2]-tspan[1]+1))).*clim)
        # MB_min = ifelse.(MB_min.>=-15.0f0, MB_min, -15.0f0)
        push!(random_MB, (gdir.rgi_id, MB_max, MB_min))
    end

    if plot
        MBmax_series, MBmin_series, labels = [],[],[]
        for glacier_MB in random_MB
            push!(MBmax_series, glacier_MB[2])
            push!(MBmin_series, glacier_MB[3])
            push!(labels, glacier_MB[1])
        end
        Plots.plot(MBmax_series, 
                xlabel="Years", ylabel="Max/min mass balance (m.w.e./yr)", label="",
                legend=:topright;
                palette=palette(:blues,15))
        MBplot = Plots.plot!(MBmin_series, label="";
                palette=palette(:reds,15))
        display(MBplot)
        Plots.savefig(MBplot,joinpath(root_plots,"MBseries.png"))
        Plots.savefig(MBplot,joinpath(root_plots,"MBseries.pdf"))
    end

    return random_MB
end

function compute_MB_matrix!(context, B, H_y, year)
    # MB array has tuples with (RGI_ID, MB_max, MB_min)
    MB_series = context.x[24]
    simulation_years = context.x[31]
    max_MB = MB_series[2][year .== simulation_years]
    min_MB = MB_series[3][year .== simulation_years]
    
    # Add mass balance based on gradient
    max_S = context.x[28]
    min_S = context.x[29] 
    max_S .= maximum(getindex(B, H_y .> 0.0f0) .+ getindex(H_y, H_y .> 0.0f0))
    min_S .= minimum(getindex(B, H_y .> 0.0f0) .+ getindex(H_y, H_y .> 0.0f0))

    # Define the mass balance as line between minimum and maximum surface
    MB = context.x[25]
    MB .= (min_MB .+ (B .+ H_y .- min_S) .* 
                ((max_MB .- min_MB) ./ (max_S .- min_S)) .* Matrix(H_y.>0.0f0)) ./ 12.0f0 # TODO: control MB timestepping
end

function compute_MB_matrix(context, S, H, year)
    simulation_years = context[12]
    max_MB = context[7][1][2][year .== simulation_years]
    min_MB = context[7][1][3][year .== simulation_years]
    max_S = maximum(getindex(S, H .> 0.0f0))
    min_S = minimum(getindex(S, H .> 0.0f0))
    MB = (min_MB .+ (S .- min_S) .* ((max_MB .- min_MB) ./ (max_S .- min_S))) .* Matrix{Float32}(H.>0.0f0) ./ 12.0f0 # TODO: control MB timestepping
    return MB
end

function compute_MB_matrix(random_MB::Tuple{String, Vector{Float32}, Vector{Float32}}, S, H, year)
    max_MB = random_MB[2][year]
    min_MB = random_MB[3][year]
    max_S = maximum(getindex(S, H .> 0.0f0))
    min_S = minimum(getindex(S, H .> 0.0f0))
    MB = (min_MB .+ (S .- min_S) .* ((max_MB - min_MB) / (max_S - min_S))) .* Matrix{Float32}(H.>0.0f0)
    return MB
end