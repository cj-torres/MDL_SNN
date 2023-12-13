using CUDA

function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward .- reward.reward ./ reward.decay .+ reward_injection, reward.decay)
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

function step_trace!(trace::EligibilityTrace, firings::CuArray{Bool, 1}, mask::CuArray{Bool, 2})
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment

    # GPU-compatible way to update e_trace with mask
    for i in 1:length(firings)
        if firings[i]
            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ (trace.constants[i, :] .* trace.pre_trace) .* mask[i, :]
            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ (trace.constants[:, i] .* trace.post_trace) .* mask[:, i]
        end
    end

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
end

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