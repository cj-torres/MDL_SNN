using CUDA
using Plots
using Random

abstract type IzhNetwork end

struct Reward
    # i.e. "dopamine"

    # amount of "reward" present in the system
    reward::AbstractFloat

    # constant decay parameter
    decay::AbstractFloat
end

mutable struct EligibilityTrace
    pre_trace::CuArray{<:AbstractFloat, 1}
    post_trace::CuArray{<:AbstractFloat, 1}
    e_trace::CuArray{<:AbstractFloat, 2}
    const pre_increment::AbstractFloat
    const post_increment::AbstractFloat
    const constants::CuArray{<:AbstractFloat, 2}
    const pre_decay::AbstractFloat
    const post_decay::AbstractFloat
    const e_decay::AbstractFloat
end


function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward .- reward.reward ./ reward.decay .+ reward_injection, reward.decay)
end


function weight_update(trace::EligibilityTrace, reward::Reward)
    return reward.reward * trace.e_trace 
end

function step_trace!(trace::EligibilityTrace, firings::CuArray{Bool, 1}, mask::CuArray{Bool, 2})
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment

    # Vectorized update for e_trace
    # Broadcasting the firings array to match the dimensions of e_trace
    firings_row = reshape(firings, 1, :)
    firings_col = reshape(firings, :, 1)

    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.pre_trace') .* mask) .* firings_col
    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.post_trace) .* mask) .* firings_row

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
end


function step_trace!(trace::EligibilityTrace, firings::CuArray{Bool, 1})
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment

    # GPU-compatible way to update e_trace
    # You may need to adjust this logic based on the specifics of your model
    for i in 1:length(firings)
        if firings[i]
            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ trace.constants[i, :] .* trace.pre_trace
            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ trace.constants[:, i] .* trace.post_trace
        end
    end

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
end


#function step_trace!(trace::EligibilityTrace, firings::CuArray{Bool, 1}, mask::CuArray{Bool, 2})
#    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
#    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment#
#
#    # GPU-compatible way to update e_trace with mask
#    for i in 1:length(firings)
#        if firings[i]
#            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ (trace.constants[i, :] .* trace.pre_trace) .* mask[i, :]
#            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ (trace.constants[:, i] .* trace.post_trace) .* mask[:, i]
#        end
#    end
#
#    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
#end

mutable struct UnmaskedIzhNetwork <: IzhNetwork
    const N::Integer
    const a::CuArray{<:AbstractFloat, 1}
    const b::CuArray{<:AbstractFloat, 1}
    const c::CuArray{<:AbstractFloat, 1}
    const d::CuArray{<:AbstractFloat, 1}
    v::CuArray{<:AbstractFloat, 1}
    u::CuArray{<:AbstractFloat, 1}
    S::CuArray{<:AbstractFloat, 2}
    S_ub::CuArray{<:AbstractFloat, 2}
    S_lb::CuArray{<:AbstractFloat, 2}
    fired::CuArray{Bool, 1}

    function UnmaskedIzhNetwork(N::Integer, 
                                a::CuArray{<:AbstractFloat, 1}, 
                                b::CuArray{<:AbstractFloat, 1}, 
                                c::CuArray{<:AbstractFloat, 1}, 
                                d::CuArray{<:AbstractFloat, 1}, 
                                v::CuArray{<:AbstractFloat, 1}, 
                                u::CuArray{<:AbstractFloat, 1}, 
                                S::CuArray{<:AbstractFloat, 2}, 
                                S_ub::CuArray{<:AbstractFloat, 2}, 
                                S_lb::CuArray{<:AbstractFloat, 2}, 
                                fired::CuArray{Bool, 1})
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
    const N::Integer
    const a::CuArray{<:AbstractFloat, 1}
    const b::CuArray{<:AbstractFloat, 1}
    const c::CuArray{<:AbstractFloat, 1}
    const d::CuArray{<:AbstractFloat, 1}
    v::CuArray{<:AbstractFloat, 1}
    u::CuArray{<:AbstractFloat, 1}
    S::CuArray{<:AbstractFloat, 2}
    S_ub::CuArray{<:AbstractFloat, 2}
    S_lb::CuArray{<:AbstractFloat, 2}
    mask::CuArray{Bool, 2}
    fired::CuArray{Bool, 1}

    function MaskedIzhNetwork(N::Integer, 
                                a::CuArray{<:AbstractFloat, 1}, 
                                b::CuArray{<:AbstractFloat, 1}, 
                                c::CuArray{<:AbstractFloat, 1}, 
                                d::CuArray{<:AbstractFloat, 1}, 
                                v::CuArray{<:AbstractFloat, 1}, 
                                u::CuArray{<:AbstractFloat, 1}, 
                                S::CuArray{<:AbstractFloat, 2}, 
                                S_ub::CuArray{<:AbstractFloat, 2}, 
                                S_lb::CuArray{<:AbstractFloat, 2}, 
                                mask::CuArray{Bool, 2}, 
                                fired::CuArray{Bool, 1})
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


function step_network!(in_voltage::CuArray{<:AbstractFloat, 1}, network::UnmaskedIzhNetwork)
    network.fired .= network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] .+ network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage .= in_voltage .+ (network.S * network.fired)

    # update voltages (twice for stability)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)

    # update recovery parameter u
    network.u .= network.u .+ network.a .* (network.b .* network.v .- network.u)
end


function step_network!(in_voltage::CuArray{<:AbstractFloat, 1}, network::MaskedIzhNetwork)
    network.fired .= network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] .+ network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage .= in_voltage .+ ((network.S .* network.mask) * network.fired)

    # update voltages (twice for stability)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= min.(network.v, 30)

    # update recovery parameter u
    network.u .= network.u .+ network.a .* (network.b .* network.v .- network.u)
end


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

function two_voltage_plot(voltage_traces::Vector{<:Vector{<:AbstractFloat}})
    plots = []
    time = [t for t in 1:length(voltage_traces[1])]
    ylimits = [0,1200]
    push!(plots, plot(time, voltage_traces[1], legend = false, ylims = [-90,40]))
    push!(plots, plot(time, voltage_traces[2], xlabel="Time (ms)", legend = false, ylims = [-90,40]))

    plot(plots..., layout = (length(plots), 1), fmt=:png, ylabel = "Voltage (mV)")
end

# Two Neuron Test Strengthen Initialization
a_2 = CuArray([.02, .02])
b_2 = CuArray([.2, .2])
c_2 = CuArray([-65.0, -65.0])
d_2 = CuArray([8.0, 8.0])

v_2 = CuArray([-65.0, -65.0])
u_2 = b_2 .* v_2
firings_2 = CuArray([false, false])

S_2 = CuArray([0.0 0.01; 0.01 0.0])
S_lb_2 = CuArray([0.0 0.0; 0.0 0.0])
S_ub_2 = CuArray([20.0 20.0; 20.0 20.0])
mask_2 = CuArray([false true; true false])

net_2 = MaskedIzhNetwork(2, a_2, b_2, c_2, d_2, v_2, u_2, S_2, S_ub_2, S_lb_2, mask_2, firings_2)

pre_synaptic_increment_2 = 0.0125
post_synaptic_increment_2 = -0.0125
reward_2 = Reward(0.0, 200)

const_matrix_2 = CuArray([1.0 1.0; 1.0 1.0])
all_decay = 1000

eligibility_trace_2 = EligibilityTrace(CUDA.zeros(2), CUDA.zeros(2), CUDA.zeros(2, 2), pre_synaptic_increment_2, post_synaptic_increment_2, const_matrix_2, all_decay, all_decay, all_decay)

print("Starting simulation Two Neuron+\n")
for T in 1:1
    global firings_2 = CuArray([false, false])
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
        I = t % 100 == 25 ? CuArray([25.0, 0.0]) : t % 200 == 30 ? CuArray([0.0, 25.0]) : CuArray([0.0, 0.0])

        step_network!(I, net_2)
        step_trace!(eligibility_trace_2, net_2.fired, net_2.mask)
        global reward_2 = t % 100 == 30 ? step_reward(reward_2, .5) : step_reward(reward_2, 0.0)

        # Find weight update increment and update weights
        dw = weight_update(eligibility_trace_2, reward_2)
        net_2.S = net_2.S + dw
        net_2.S .= clamp.(net_2.S, S_lb_2, S_ub_2)

        # Collect data for plotting
        push!(pre_voltage, Array(net_2.v)[1])
        push!(post_voltage, Array(net_2.v)[2])
        push!(pre_u, Array(net_2.u)[1])
        push!(post_u, Array(net_2.u)[2])
        push!(pre_trace_plus, Array(eligibility_trace_2.pre_trace)[1])
        push!(post_trace_plus, Array(eligibility_trace_2.post_trace)[2])
        push!(e_trace_plus, Array(eligibility_trace_2.e_trace)[2, 1])
        push!(reward_trace_plus, reward_2.reward)
        push!(weight_plus, Array(net_2.S)[2, 1])

        global firings_2 = hcat(firings_2, Array(net_2.fired))
    end

    print("$T seconds finished\n")

    if T % 5 == 1
        raster_plot(Array(firings_2))
        savefig("two_neuron_strengthen_raster_plot_$T.png")
        eligibility_trace_plots([pre_trace_plus, post_trace_plus, e_trace_plus, reward_trace_plus, weight_plus])
        savefig("two_neuron_strengthen_trace_plot_$T.png")
        two_voltage_plot([pre_voltage, post_voltage])
        savefig("two_neuron_strengthen_voltage_plot_$T.png")
        eligibility_trace_plots([pre_u, post_u])
        savefig("two_neuron_strengthen_u_plot_$T.png")
    end
end

### STDP NETWORK TEST ####
### POP+ LEARNING     ####

# excitatory and inhibitory nuerons
Ne, Ni = 1600, 400
N = Ne + Ni
N = Int64(N)

# initialize network parameters and network
re, ri = rand(Ne), rand(Ni)

is_inhibitory = vcat([false for i in 1:Ne],[true for in in 1:Ni])


a = CuArray(vcat([.02 for i in 1:Ne], [.06 for i in 1:Ni]))
b = CuArray(vcat([.2 for i in 1:Ne] , [.225 for i in 1:Ni]))
c = CuArray([-65.0 for i in 1:N])
d = CuArray(vcat([8.0 for i in 1:Ne], [2.0 for i in 1:Ni]))
S = CuArray(hcat(.5*rand(Ne+Ni, Ne), -rand(Ne+Ni, Ni)))
mask = CuArray(rand([true, false], N, N) .* rand([true, false], N, N).* rand([true, false], N, N))
S = S .* mask
S_ub = CuArray(repeat(vcat(4.0*ones(Ne), zeros(Ni))', N, 1))
S_lb = CuArray(repeat(vcat(zeros(Ne), -4*ones(Ni))', N, 1))

v = CuArray([-65.0 for i in 1:(Ne+Ni)])
u = CuArray(b .* v)
firings = CuArray([false for i in 1:(Ne+Ni)])

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
const_matrix = CuArray(repeat(const_vector', N, 1))

all_decay = 1000
eligibility_trace = EligibilityTrace(CUDA.zeros(N), CUDA.zeros(N), CUDA.zeros(N, N), pre_synaptic_increment, post_synaptic_increment, const_matrix, all_decay, all_decay, all_decay)

output_groups = [randperm(Ne)[1:50], randperm(Ne)[1:50], randperm(Ne)[1:50]]
input = randperm(Ne)[1:500]

avg_tau = 10.0
energy = 5


print("Starting simulation simple sequence learner \n")
for T in 1:36001
    global firings = CuArray([false for i in 1:N])
    fire_group_1 = CuArray([false for i in 1:50])
    fire_group_2 = CuArray([false for i in 1:50])
    fire_group_3 = CuArray([false for i in 1:50])

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
        I = CuArray(zeros(N))
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
                if ema_1 > .3 && ema_2 < .1 && ema_3 < .1
                    global reward = step_reward(reward, 0.1)
                else
                    global reward = step_reward(reward, 0.0)
                end
            # second half should output another rate
            else
                if ema_2 > .3 && ema_1 < .1 && ema_3 < .1
                    global reward = step_reward(reward, 0.1)
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
