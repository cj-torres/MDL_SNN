using DifferentialEquations

mutable struct IzhNetwork
    
    # number of neurons
    const N::Int
    # time scale recovery parameter
    const a::Vector{Float16}
    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{Float16}
    # post-spike reset value of membrane potential v
    const c::Vector{Float16}
    # post-spike reset of recovery variable u
    const d::Vector{Float16}

    # membrane poential and recovery variable, used in Izhikevich system of equations
    v::Vector{Float16}
    u::Vector{Float16}

    # synaptic weights
    S::Matrix{Float16}

    # boolean of is-fired
    fired::Vector{Bool}


end


function step_network(in_voltage::Vector{Float16}, network::IzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage = in_voltage + (network.S * network.fired)

    # update voltages (twice for stability)
    network.v = network.v + 0.5*(0.04*network.v + network.v .^ 2 + 5*network.v .+ 140 - network.u + in_voltage)
    network.v = network.v + 0.5*(0.04*network.v + network.v .^ 2 + 5*network.v .+ 140 - network.u + in_voltage)

    # update recovery parameter u
    network.u = network.u + network.a * (network.b .* network.v - network.u)

end




Ne, Ni = 1600, 400

re, ri = rand(Ne), rand(Ni)

a = vcat([.02 for i in 1:Ne], (.02 .+ .08*ri))
b = vcat([.2 for i in 1:Ne] , (.25 .- .05*ri))
c = vcat((-65.0 .+ 15.0 * (re .^ 2)), ([-65.0 for i in 1:Ni]))
d = vcat((8.0 .- 6.0*(re .^ 2)), [2.0 for i in 1:Ni])
S = hcat(.5*rand(Ne+Ni, Ne), -rand(Ne+Ni, Ni))

v = [-65.0 for i in 1:(Ne+Ni)]
u = b .* v










































