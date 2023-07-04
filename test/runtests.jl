import Pkg
Pkg.activate(dirname(Base.current_project()))

using PyCall
# Update SSL certificate to avoid issue in GitHub Actions CI
certifi = pyimport("certifi")
ENV["SSL_CERT_FILE"] = certifi.where()
# println("Current SSL certificate: ", ENV["SSL_CERT_FILE"])

using Revise
using ODINN
using Test
using JLD2
using Plots
using Infiltrator

ODINN.enable_multiprocessing(1) # Force one single worker

include("PDE_UDE_solve.jl")
include("halfar.jl")
include("mass_conservation.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

atol = 0.01
# @testset "PDE and UDE SIA solvers without MB" pde_solve_test(atol; MB=false, fast=true)

atol = 2.0
# @testset "PDE and UDE SIA solvers with MB" pde_solve_test(atol; MB=true, fast=true)

@testset "Halfar Solution" halfar_test(rtol=0.02, atol=1.0)

@testset "Conservation of Mass - Flat Bed" unit_mass_flatbed_test(rtol=1.0e-7, atol=1000)