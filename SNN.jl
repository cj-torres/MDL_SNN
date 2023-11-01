using Plots
using SparseArrays
using Profile
using Random

Random.seed!(1234)

cd("F:\\VSCode Projects\\MDLSNN\\MDL_SNN")

abstract type IzhNetwork end

# AboAmmar quick elementwise row multiplication
# Found here: https://stackoverflow.com/questions/48460875/vector-matrix-element-wise-multiplication-by-rows-in-julia-efficiently

function pre_trace_copier(pre_trace::Vector{<:AbstractFloat}, firings::Vector{Bool})
    side = length(pre_trace)
    M = zeros(side, side)

    @simd for i in 1:length(firings)
        @inbounds if firings[i]
            M[i, :] = pre_trace
        end
    end
    M
end

function post_trace_copier(post_trace::Vector{<:AbstractFloat}, firings::Vector{Bool})
    side = length(post_trace)
    M = zeros(side, side)

    @simd for j in 1:length(firings)
        @inbounds if firings[j]
            M[:, j] = post_trace
        end
    end
    M
end

# Implement SSP?

# reward functionality
# Thanks be to Quintana, Perez-Pena, and Galindo (2022) for the following algorithm

mutable struct Reward
    # i.e. "dopamine"

    # amount of "reward" present in the system
    const reward::AbstractFloat

    # constant decay parameter
    const decay::AbstractFloat
end

function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward - reward.reward/reward.decay + reward_injection, reward.decay)
end

mutable struct EligibilityTrace
    # for speed the inhibitory-ness of junctions must be stored here within the constants

    # vectors to keep track of traces, typically initialized at 0
    pre_trace::Vector{<:AbstractFloat}
    post_trace::Vector{<:AbstractFloat}
    e_trace::Matrix{<:AbstractFloat}

    # Parameters for pre/post incrementing and decay
    # 
    const pre_increment::AbstractFloat
    const post_increment::AbstractFloat

    # Constant to multiply junction traces by when updating the eligibility trace
    # Should typically be negative for inhibitory junctions
    const constants::Matrix{<:AbstractFloat}

    # Decay parameters
    const pre_decay::AbstractFloat
    const post_decay::AbstractFloat
    const e_decay::AbstractFloat
end


function step_trace!(trace::EligibilityTrace, firings::Vector{Bool})
    len_pre = length(trace.pre_trace)
    len_post = length(trace.post_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment


    # restructured logic
    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

    @inbounds for i in 1:len_post
        @inbounds @simd for j in 1:len_pre


            # i is row index, j is column index, so...
            #
            #
            # Pre-synaptic input (indexed with j)
            #     |
            #     v
            # . . . . .
            # . . . . . --> Post-synaptic output (indexed with i)
            # . . . . .
            
            # Check if presynaptic neuron is inhibitory

            # We add the *opposite* trace given a neural spike
            # So if post-synaptic neuron i spikes, we add the trace for the 
            # pre-synaptic neuron to the eligibility trace

            if firings[i]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
            end

            # And if pre-synaptic neuron j spikes, we add the trace for the 
            # post-synaptic neuron to the eligibility trace

            if firings[j]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
            end

            # each trace will decay according to the decay parameter
            

        end
    end

    trace.e_trace = trace.e_trace .- trace.e_trace/trace.e_decay
end

function step_trace!(trace::EligibilityTrace, firings::Vector{Bool}, mask::AbstractMatrix{Bool})

    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

    @inbounds for i in 1:len_post
        @inbounds @simd for j in 1:len_pre


            # i is row index, j is column index, so...
            #
            #
            # Pre-synaptic input (indexed with j)
            #     |
            #     v
            # . . . . .
            # . . . . . --> Post-synaptic output (indexed with i)
            # . . . . .
            
            # Check if presynaptic neuron is inhibitory

            # Check if the neurons have a synpatic connection j -> i
            # remember wonky row/column indexing makes things "backwards"
            if mask[i,j]

                # We add the *opposite* trace given a neural spike
                # So if post-synaptic neuron i spikes, we add the trace for the 
                # pre-synaptic neuron to the eligibility trace

                if firings[i]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
                end

                # And if pre-synaptic neuron j spikes, we add the trace for the 
                # post-synaptic neuron to the eligibility trace

                if firings[j]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
                end

                # each trace will decay according to the decay parameter
                
            end

        end
    end

    # each trace will decay according to the decay parameter
    trace.e_trace = trace.e_trace - trace.e_trace/trace.e_decay
end



function step_trace!(trace::EligibilityTrace, firings::Vector{Bool}, mask::SparseMatrixCSC{Bool, <:Integer})
    # Sparse masking not currently recommended, but kept because it was written

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

    # restructured logic
    #
    # Pre-synaptic input (second index j, or column index j, of matrix)
    #     |
    #     v
    # . . . . .
    # . . . . . --> Post-synaptic output (first index i or row index i)
    # . . . . .

    #firings_s = firings
    addable_pre_trace = pre_trace_copier(trace.pre_trace, firings) #trace.pre_trace * firings_s'
    addable_post_trace = post_trace_copier(trace.post_trace, firings) #firings_s * trace.post_trace'
    e_trace_delta = (addable_post_trace+addable_pre_trace) .* trace.constants 
    trace.e_trace = trace.e_trace + e_trace_delta .* mask - trace.e_trace/trace.e_decay


    #for j in 1:J
    #    is_inhibitory_cleft = is_inhibitory[j]
    #    if is_inhibitory_cleft
    #        # k variable used to loop over range in nzrange and retrieve true row index i
    #        for k in nzrange(mask, j)
    #            # use nzrange to get actual row index i
    #            i = rows[k]
    #
    #
    #            if firings[i]
    #                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.inhibitory_constant*trace.pre_trace[j]
    #            end
    #            if firings[j]
    #                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.inhibitory_constant*trace.post_trace[i]
    #            end
    #
    #        end
    #    else
    #        for k in nzrange(mask, j)
    #            i = rows[k]
    #            if firings[i]
    #                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.pre_trace[j]
    #            end
    #            if firings[j]
    #                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.post_trace[i]
    #            end
    #
    #        end
    #    end
    #end

    # each trace will decay according to the decay parameter
    #trace.e_trace = trace.e_trace - trace.e_trace/trace.e_decay
end

function weight_update(trace::EligibilityTrace, reward::Reward)
    return reward.reward * trace.e_trace 
end

# network structures, see Izhikevich simple model and STDP papers

mutable struct UnmaskedIzhNetwork <: IzhNetwork
    
    # number of neurons
    const N::Integer

    # time scale recovery parameter
    const a::Vector{<:AbstractFloat}

    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{<:AbstractFloat}

    # post-spike reset value of membrane potential v
    const c::Vector{<:AbstractFloat}

    # post-spike reset of recovery variable u
    const d::Vector{<:AbstractFloat}

    # membrane potential and recovery variable, used in Izhikevich system of equations
    v::Vector{<:AbstractFloat}
    u::Vector{<:AbstractFloat}

    # synaptic weights
    S::Matrix{<:AbstractFloat}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{<:AbstractFloat}
    S_lb::Matrix{<:AbstractFloat}


    # boolean of is-fired
    fired::Vector{Bool}

    function UnmaskedIzhNetwork(N::Integer, a::Vector{<:AbstractFloat}, b::Vector{<:AbstractFloat}, c::Vector{<:AbstractFloat}, d::Vector{<:AbstractFloat}, v::Vector{<:AbstractFloat}, u::Vector{<:AbstractFloat}, S::Matrix{<:AbstractFloat}, S_ub::Matrix{<:AbstractFloat}, S_lb::Matrix{<:AbstractFloat}, fired::AbstractVector{Bool})
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert size(S) == (N, N)
        @assert size(S_lb) == (N, N)
        @assert size(S_ub) == (N, N)
        @assert length(fired) == N

        return new(N, a, b, c, d, v, u, S, S_ub, S_lb, fired)
    end
end


mutable struct MaskedIzhNetwork <: IzhNetwork
    
    # number of neurons
    const N::Integer
    # time scale recovery parameter
    const a::Vector{<:AbstractFloat}
    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{<:AbstractFloat}
    # post-spike reset value of membrane potential v
    const c::Vector{<:AbstractFloat}
    # post-spike reset of recovery variable u
    const d::Vector{<:AbstractFloat}

    # membrane poential and recovery variable, used in Izhikevich system of equations
    v::Vector{<:AbstractFloat}
    u::Vector{<:AbstractFloat}

    # synaptic weights
    S::Union{Matrix{<:AbstractFloat}, SparseMatrixCSC{<:AbstractFloat, <:Integer}}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{<:AbstractFloat}
    S_lb::Matrix{<:AbstractFloat}

    # mask
    mask::Union{AbstractMatrix{Bool}, SparseMatrixCSC{Bool, <:Integer}}

    # boolean of is-fired
    fired::Vector{Bool}

    function MaskedIzhNetwork(N::Integer, a::Vector{<:AbstractFloat}, b::Vector{<:AbstractFloat}, c::Vector{<:AbstractFloat}, d::Vector{<:AbstractFloat}, v::Vector{<:AbstractFloat}, u::Vector{<:AbstractFloat}, S::Union{Matrix{<:AbstractFloat}, SparseMatrixCSC{<:AbstractFloat, <:Integer}}, S_ub::Matrix{<:AbstractFloat}, S_lb::Matrix{<:AbstractFloat}, mask::Union{AbstractMatrix{Bool}, SparseMatrixCSC{Bool, <:Integer}}, fired::AbstractVector{Bool})
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert size(S) == (N, N)
        @assert size(mask) == (N, N)
        @assert length(fired) == N

        return new(N, a, b, c, d, v, u, S, S_ub, S_lb, mask, fired)
    end


end


mutable struct BidirectionalConnection
    forward::Matrix{<:AbstractFloat}
    forward_mask::AbstractMatrix{Bool}

    backward::Matrix{<:AbstractFloat}
    backward_mask::AbstractMatrix{Bool}
end


mutable struct IzhSuperNetwork <: IzhNetwork
    nodes::Vector{IzhNetwork}
    connections::Dict{Tuple{Int, Int}, BidirectionalConnection}
end



function step_network!(in_voltage::Vector{<:AbstractFloat}, network::UnmaskedIzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage = in_voltage + (network.S * network.fired)

    # update voltages (twice for stability)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)

    # update recovery parameter u
    network.u = network.u + network.a .* (network.b .* network.v - network.u)

end


function step_network!(in_voltage::Vector{<:AbstractFloat}, network::MaskedIzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage = in_voltage + ((network.S .* network.mask) * network.fired)

    # update voltages (twice for stability)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)

    # update recovery parameter u
    network.u = network.u + network.a .* (network.b .* network.v - network.u)

end


# neural plotting functions

function raster_plot(spike_matrix::AbstractMatrix{Bool})
    # takes boolean matrices of shape N x T dpecifying whether or not neuron N fired at time T

    N, T = size(spike_matrix)
    
    # Prepare x and y arrays for scatter plot
    xs = []
    ys = []
    
    for n in 1:N
        for t in 1:T
            if spike_matrix[n, t]
                push!(xs, t)
                push!(ys, n)
            end
        end
    end

    plot(xs, ys, seriestype=:scatter, fmt=:png, ms=.5, legend=false, yflip=true, xlabel="Time", ylabel="Neuron Index", title="Raster Plot")
end


function eligibility_trace_plots(traces::Vector{<:Vector{<:AbstractFloat}})
    # takes trace plots with inputs of various matrices of shape T
    # value for each entry at t <: T indicates trace strength at time t
    plots = []

    for trace in traces
        time = [t for t in 1:length(trace)]
        push!(plots, plot(time, trace))
    end

    plot(plots..., layout = (length(plots), 1), fmt=:png)
end



### Two Neuron Test Strengthen ###

S_2 = [0.0 .01; .01 0.0]
S_lb_2 = [0.0 0.0; 0.0 0.0]
S_ub_2 = [20.0 20.0; 20.0 20.0]
mask_2 = [false true; true false]
a_2 = [.02, .02]
b_2 = [.2, .2]
c_2 = [-65.0, -65.0]
d_2 = [8.0, 8.0]

v_2 = [-65.0, -65.0]
u_2 = b_2 .* v_2
firings_2 = [false, false]

reward_2 = Reward(0.0, 200)

net_2 = MaskedIzhNetwork(2, a_2, b_2, c_2, d_2, v_2, u_2, S_2, S_ub_2, S_lb_2, mask_2, firings_2)


pre_synaptic_increment_2 = .0125
post_synaptic_increment_2 = -.0125

const_matrix_2 = [1.0 1.0; 1.0 1.0]
all_decay = 1000

eligibility_trace_2 = EligibilityTrace(zeros(2), zeros(2), zeros(2, 2), pre_synaptic_increment_2, post_synaptic_increment_2, const_matrix_2, all_decay, all_decay, all_decay)

print("Starting simulation Two Neuron+\n")
for T in 1:1
    global firings_2 = [false, false]
    pre_voltage = Float64[]
    post_voltage = Float64[]

    pre_u = Float64[]
    post_u = Float64[]

    pre_trace_plus = Float64[]
    post_trace_plus = Float64[]
    e_trace_plus = Float64[]
    reward_trace_plus = Float64[]
    weight_plus = Float64[]

    for t in 1:2000
        if t%100 == 25
            I = [25.0, 0.0]
        elseif t%200 == 30
            I = [0.0, 25.0]
        else
            I = [0.0, 0.0]
        end

        step_network!(I, net_2)

        step_trace!(eligibility_trace_2, net_2.fired, net_2.mask)

        # inject dopamine if Group 1 stimulated
        global reward_2  = t%100 == 30 ? step_reward(reward_2, .5) : step_reward(reward_2, 0.0)

        # find weight update increment
        dw = weight_update(eligibility_trace_2, reward_2)
        
        # update weights
        net_2.S = net_2.S + dw
        net_2.S .= clamp.(net_2.S, S_lb_2, S_ub_2)

        global firings_2 = hcat(firings_2, net_2.fired)
        push!(pre_trace_plus, eligibility_trace_2.pre_trace[1])
        push!(post_trace_plus, eligibility_trace_2.post_trace[2])
        push!(e_trace_plus, eligibility_trace_2.e_trace[2, 1])
        push!(reward_trace_plus, reward_2.reward)
        push!(weight_plus, net_2.S[2, 1])
        push!(pre_voltage, net_2.v[1])
        push!(post_voltage, net_2.v[2])
        push!(pre_u, net_2.u[1])
        push!(post_u, net_2.u[2])
    end

    print("$T seconds finished\n")
    if T % 5 == 1
        raster_plot(firings_2)
        savefig("two_neuron_strengthen_raster_plot_$T.png")
        eligibility_trace_plots([pre_trace_plus, post_trace_plus, e_trace_plus, reward_trace_plus, weight_plus])
        savefig("two_neuron_strengthen_trace_plot_$T.png")
        eligibility_trace_plots([pre_voltage, post_voltage])
        savefig("two_neuron_strengthen_voltage_plot_$T.png")
        eligibility_trace_plots([pre_u, post_u])
        savefig("two_neuron_strengthen_u_plot_$T.png")
    end
end


### Two Neuron Test Weaken ###



S_2 = [0.0 20.0; 20.0 0.0]
S_lb_2 = [0.0 0.0; 0.0 0.0]
S_ub_2 = [20.0 20.0; 20.0 20.0]
mask_2 = [false true; true false]
a_2 = [.02, .02]
b_2 = [.2, .2]
c_2 = [-65.0, -65.0]
d_2 = [8.0, 8.0]

v_2 = [-65.0, -65.0]
u_2 = b_2 .* v_2
firings_2 = [false, false]

reward_2 = Reward(0.0, 200)

net_2 = MaskedIzhNetwork(2, a_2, b_2, c_2, d_2, v_2, u_2, S_2, S_ub_2, S_lb_2, mask_2, firings_2)


pre_synaptic_increment_2 = .0125
post_synaptic_increment_2 = -.0125

const_matrix_2 = [1.0 1.0; 1.0 1.0]
all_decay = 1000

eligibility_trace_2 = EligibilityTrace(zeros(2), zeros(2), zeros(2, 2), pre_synaptic_increment_2, post_synaptic_increment_2, const_matrix_2, all_decay, all_decay, all_decay)

print("Starting simulation Two Neuron-\n")
for T in 1:1
    global firings_2 = [false, false]
    pre_voltage = Float64[]
    post_voltage = Float64[]

    pre_trace_minus = Float64[]
    post_trace_minus = Float64[]
    e_trace_minus = Float64[]
    reward_trace_minus = Float64[]
    weight_minus = Float64[]

    for t in 1:2000
        if t%200 == 25
            I = [0.0, 25.0]
        elseif t%100 == 30
            I = [25.0, 0.0]
        else
            I = [0.0, 0.0]
        end

        step_network!(I, net_2)

        step_trace!(eligibility_trace_2, net_2.fired, net_2.mask)

        # inject dopamine if Group 1 stimulated
        global reward_2 = t%100 == 30 ? step_reward(reward_2, .5) : step_reward(reward_2, 0.0)

        # find weight update increment
        dw = weight_update(eligibility_trace_2, reward_2)
        
        # update weights
        net_2.S = net_2.S + dw
        net_2.S .= clamp.(net_2.S, S_lb_2, S_ub_2)

        global firings_2 = hcat(firings_2, net_2.fired)
        push!(pre_trace_minus, eligibility_trace_2.pre_trace[1])
        push!(post_trace_minus, eligibility_trace_2.post_trace[2])
        push!(e_trace_minus, eligibility_trace_2.e_trace[2, 1])
        push!(reward_trace_minus, reward_2.reward)
        push!(weight_minus, net_2.S[2, 1])
        push!(pre_voltage, net_2.v[1])
        push!(post_voltage, net_2.v[2])
    end

    print("$T seconds finished\n")
    if T % 5 == 1
        raster_plot(firings_2)
        savefig("two_neuron_weaken_raster_plot_$T.png")
        eligibility_trace_plots([pre_trace_minus, post_trace_minus, e_trace_minus, reward_trace_minus, weight_minus])
        savefig("two_neuron_weaken_trace_plot_$T.png")
        eligibility_trace_plots([pre_voltage, post_voltage])
        savefig("two_neuron_weaken_voltage_plot_$T.png")
    end
end




### STDP NETWORK TEST ####
### POP+ LEARNING     ####

# excitatory and inhibitory nuerons
Ne, Ni = 800, 200
N = Ne + Ni
N = Int64(N)

# initialize network parameters and network
re, ri = rand(Ne), rand(Ni)

is_inhibitory = vcat([false for i in 1:Ne],[true for in in 1:Ni])


a = (vcat([.02 for i in 1:Ne], [.06 for i in 1:Ni]))
b = (vcat([.2 for i in 1:Ne] , [.225 for i in 1:Ni]))
c = [-65.0 for i in 1:N]
d = (vcat([8.0 for i in 1:Ne], [2.0 for i in 1:Ni]))
S = (hcat(.5*rand(Ne+Ni, Ne), -rand(Ne+Ni, Ni)))
mask = rand([true, false], 1000, 1000) .* rand([true, false], 1000, 1000)
S = S .* mask
S_ub = repeat(vcat(4.0*ones(Ne), zeros(Ni))', N, 1)
S_lb = repeat(vcat(zeros(Ne), -4*ones(Ni))', N, 1)

v = ([-65.0 for i in 1:(Ne+Ni)])
u = (b .* v)
firings = [false for i in 1:(Ne+Ni)]

net = MaskedIzhNetwork(Ne+Ni, a, b, c, d, v, u, S, S_ub, S_lb, mask, firings)

# initialize "dopamine" levels and decay parameter (.2s in ms)
reward = Reward(0.0, 200)

# initialize elgibility trace parameters
pre_synaptic_increment = .125
post_synaptic_increment = -.125

# used to initialize const_matrix
inhibitory_multiple = -1.5
excitatory_multiple = 1
const_vector = vcat([excitatory_multiple for i in 1:Ne], [inhibitory_multiple for i in 1:Ni])

# create a matrix of constants for the eligibility trace, this is used to keep inhibitory connections negative
const_matrix = repeat(const_vector', N, 1)

all_decay = 1000
eligibility_trace = EligibilityTrace(zeros(N), zeros(N), zeros(N, N), pre_synaptic_increment, post_synaptic_increment, const_matrix, all_decay, all_decay, all_decay)

output_groups = [randperm(Ne)[1:50], randperm(Ne)[1:50], randperm(Ne)[1:50]]
input = randperm(Ne)[1:500]

avg_tau = 10.0
energy = 5


print("Starting simulation simple sequence learner \n")
for T in 1:36001
    global firings = [false for i in 1:N]
    fire_group_1 = [false for i in 1:50]
    fire_group_2 = [false for i in 1:50]
    fire_group_3 = [false for i in 1:50]

    ema_1 = 0.0
    ema_2 = 0.0
    ema_3 = 0.0

    fire_rate_1 = Float64[]
    fire_rate_2 = Float64[]
    fire_rate_3 = Float64[]

    pre_trace_1 = Float64[]
    post_trace_1 = Float64[]
    reward_trace = Float64[]

    for t in 1:1000
        I = zeros(N)
        if T % 2 == 1 
            I[input] = energy*randn(500)
        end

        step_network!(I, net)

        step_trace!(eligibility_trace, net.fired, net.mask)

        # update exponential moving average
        ema_1 = sum(net.fired[output_groups[1]]) / avg_tau + ema_1 * (1 -  1 / avg_tau)
        ema_2 = sum(net.fired[output_groups[2]]) / avg_tau + ema_2 * (1 -  1 / avg_tau)
        ema_3 = sum(net.fired[output_groups[3]]) / avg_tau + ema_3 * (1 -  1 / avg_tau)

        # perform sequence learning on odd trials, otherwise allow network to return to baselines
        if T % 2 == 1
            # first half of second should output one rate
            if t < 500
                if ema_1 > .5 && ema_2 < .2 && ema_3 < .2
                    global reward = step_reward(reward, 0.2)
                else
                    global reward = step_reward(reward, 0.0)
                end
            # second half should output another rate
            else
                if ema_2 > .5 && ema_1 < .2 && ema_3 < .2
                    global reward = step_reward(reward, 0.2)
                else
                    global reward = step_reward(reward, 0.0)
                end
            end
        else
            global reward = step_reward(reward, 0.0)
        end

        # find weight update increment
        dw = weight_update(eligibility_trace, reward)
        
        # update weights
        net.S = net.S + dw
        net.S .= clamp.(net.S, S_lb, S_ub)

        global firings = hcat(firings, net.fired)
        push!(pre_trace_1, eligibility_trace.pre_trace[1])
        push!(post_trace_1, eligibility_trace.post_trace[1])
        push!(fire_rate_1, ema_1)
        push!(fire_rate_2, ema_2)
        push!(fire_rate_3, ema_3)
        push!(reward_trace, reward.reward)

        if t%100 == 0
            print("$t ms simulated... \n")
        end
    end

    print("$T seconds finished\n")
    if T % 2 == 1
        raster_plot(firings)
        savefig("raster_plot_$T.png")
        eligibility_trace_plots([pre_trace_1, post_trace_1, reward_trace])
        savefig("trace_plot_$T.png")
        eligibility_trace_plots([fire_rate_1, fire_rate_2, fire_rate_3])
        savefig("fire_rate_$T.png")
    end
end

