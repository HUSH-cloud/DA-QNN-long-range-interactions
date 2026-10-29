# Digital-analog QNN with a long-range transverse-field Ising analog block.
#
#   H = -sum_{i<j} |i-j|^(-alpha) Z_i Z_j - kac_norm(N, alpha) * h * sum_i X_i
#   H / kac_norm is Eq. (1) of the paper, so evolving H for t = tau / kac_norm equals evolving Eq. (1) for tau.
#   circuit = R_0 (U R_1) (U R_2) ... (U R_ell),  U = exp(-i H t),  R_l = Rx Rz Rx on every qubit
#
# Qubit labels follow Yao: qubit 1 is the least significant bit of the state index.

using Yao, LinearAlgebra, SparseArrays, Random, NPZ

const BATCH_SIZE = 100
const LEARNING_RATE = 0.1
const N_ITERATIONS = 70

"""Kac factor (1/(N-1)) sum_{i<j} |i-j|^(-alpha). It equals 1 for alpha = Inf."""
function kac_norm(nbit::Int, alpha::Float64)
    isinf(alpha) && return 1.0
    total = 0.0
    for i in 1:nbit, j in (i + 1):nbit
        total += 1.0 / (j - i)^alpha
    end
    return total / (nbit - 1)
end

"""Sparse Hamiltonian of size (2^nbit, 2^nbit). alpha = Inf keeps nearest-neighbor bonds only."""
function lrtfim_hamiltonian(nbit::Int, alpha::Float64, h::Float64)
    dim = 2^nbit
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    bonds = isinf(alpha) ? [(i, i + 1) for i in 1:(nbit - 1)] : [(i, j) for i in 1:nbit for j in (i + 1):nbit]
    for (i, j) in bonds
        J_ij = isinf(alpha) ? 1.0 : 1.0 / (j - i)^alpha
        for s in 0:(dim - 1)
            z_i = 1 - 2 * ((s >> (i - 1)) & 1)
            z_j = 1 - 2 * ((s >> (j - 1)) & 1)
            push!(rows, s + 1); push!(cols, s + 1); push!(vals, -J_ij * (z_i * z_j))
        end
    end
    field = kac_norm(nbit, alpha) * h
    for i in 1:nbit
        for s in 0:(dim - 1)
            push!(rows, s + 1); push!(cols, (s ⊻ (1 << (i - 1))) + 1); push!(vals, -field)
        end
    end
    return sparse(rows, cols, vals, dim, dim)
end

"""Analog block holding the exact propagator U = exp(-i H t), shape (2^nbit, 2^nbit)."""
struct AnalogBlock <: Yao.PrimitiveBlock{2}
    nbit::Int
    U::Matrix{ComplexF64}
end

Yao.nqudits(b::AnalogBlock) = b.nbit
Yao.print_block(io::IO, b::AnalogBlock) = print(io, "AnalogBlock(nbit=$(b.nbit))")
Yao.mat(::Type{ComplexF64}, b::AnalogBlock) = b.U

# Yao chains apply a block through unsafe_apply!; this method applies the exact propagator U to the state.
function Yao.YaoBlocks.YaoAPI.unsafe_apply!(reg::Yao.AbstractRegister, b::AnalogBlock)
    reg.state .= b.U * reg.state
    return reg
end

"""Rx Rz Rx on every qubit, 3 * nbit parameters."""
rotation_layer(nbit::Int) = chain(
    chain(put(nbit, i => Rx(0.0)) for i in 1:nbit),
    chain(put(nbit, i => Rz(0.0)) for i in 1:nbit),
    chain(put(nbit, i => Rx(0.0)) for i in 1:nbit),
)

"""R_0 followed by ell pairs (U, R_l), with (ell + 1) * 3 * nbit parameters."""
function da_circuit(nbit::Int, ell::Int, t::Float64, alpha::Float64, h::Float64)
    U = AnalogBlock(nbit, exp(-im * t * Matrix(lrtfim_hamiltonian(nbit, alpha, h))))
    layers = Any[rotation_layer(nbit)]
    for _ in 1:ell
        push!(layers, U, rotation_layer(nbit))
    end
    return chain(layers...)
end

"""
Product state with cos(theta_j/2)|0> + exp(i phi_j) sin(theta_j/2)|1> on encoded qubit j, length 2^N.
kron puts encoded qubit 1 in the most significant bit, which is Yao qubit N.
"""
function product_state(theta::AbstractVector, phi::AbstractVector)
    psi = [1.0 + 0.0im]
    for j in eachindex(theta)
        psi = kron(psi, [cos(theta[j] / 2); exp(im * phi[j]) * sin(theta[j] / 2)])
    end
    return psi
end

"""
Read x_{train,test}_n{nbit}.npy, shape (2 * nbit, samples) with theta in rows 1:nbit and phi below,
and the one-hot labels y_{train,test}.npy, shape (samples, 2).
States are returned with shape (2^nbit, samples).
"""
function load_states(dir::String, nbit::Int)
    encode(x) = hcat([product_state(x[1:nbit, i], x[(nbit + 1):(2 * nbit), i]) for i in 1:size(x, 2)]...)
    return (
        states_train = encode(npzread(joinpath(dir, "x_train_n$(nbit).npy"))),
        labels_train = npzread(joinpath(dir, "y_train.npy")),
        states_test = encode(npzread(joinpath(dir, "x_test_n$(nbit).npy"))),
        labels_test = npzread(joinpath(dir, "y_test.npy")),
    )
end

"""
Train the circuit parameters in place with AdaGrad on the cross-entropy of p0 = P(|0>) on Yao qubit `readout`.
Label column 1 is the class predicted by p0 > 1/2.
p0 and 1 - p0 are clipped to [prob_floor, 1 - prob_floor] and shifted by prob_offset in the gradient.
The rng MersenneTwister(seed) draws the initial parameters (randn) unless `initial_params` (length n_params) is
given, and the batch order of every iteration (randperm). If `loss_history` is a vector, the batch cross-entropy
of the clipped probabilities is appended to it before each update, n_iterations values in total.
"""
function train_circuit!(circuit, data, readout::Int; seed::Int, prob_floor::Float64 = 1e-7, prob_offset::Float64 = 0.0,
                        n_iterations::Int = N_ITERATIONS, initial_params = nothing, loss_history = nothing)
    rng = MersenneTwister(seed)
    nbit = nqudits(circuit)
    n_params = nparameters(circuit)
    n_train = size(data.states_train, 2)
    projector0 = put(nbit, readout => matblock(ComplexF64[1 0; 0 0]))

    if initial_params === nothing
        params = randn(rng, n_params)
    else
        @assert length(initial_params) == n_params "expected $n_params initial parameters, got $(length(initial_params))"
        params = copy(initial_params)
    end
    dispatch!(circuit, params)
    accumulated = zeros(n_params)

    for _ in 1:n_iterations
        batch = randperm(rng, n_train)[1:min(BATCH_SIZE, n_train)]
        n_batch = length(batch)
        regs = [ArrayReg(data.states_train[:, k:k]) for k in batch]
        labels = data.labels_train[batch, :]

        p0 = [real(expect(projector0, regs[b] => circuit)) for b in 1:n_batch]
        @assert all(-1e-9 .<= p0 .<= 1 + 1e-9) "readout probability outside [0, 1]: the state is not normalized"
        q0 = clamp.(p0, prob_floor, 1 - prob_floor)
        q1 = clamp.(1 .- p0, prob_floor, 1 - prob_floor)
        if loss_history !== nothing
            push!(loss_history, -sum(labels[b, 1] * log(q0[b]) + labels[b, 2] * log(q1[b]) for b in 1:n_batch) / n_batch)
        end

        dp0_dparams = zeros(n_batch, n_params)
        for b in 1:n_batch
            dp0_dparams[b, :] = expect'(projector0, regs[b] => circuit).second
        end
        dloss_dp0 = [labels[b, 2] / (q1[b] + prob_offset) - labels[b, 1] / (q0[b] + prob_offset) for b in 1:n_batch]
        grads = [sum(dloss_dp0[b] * dp0_dparams[b, j] for b in 1:n_batch) for j in 1:n_params]

        accumulated .+= grads .^ 2
        params = params .- LEARNING_RATE .* grads ./ (sqrt.(accumulated) .+ 1e-8)
        dispatch!(circuit, params)
    end
    return circuit
end

"""
Initial parameters for (seed, n_params), shape (n_params,), from a table written by data/initial_params_julia110.jl:
'#' lines are comments, every other line is "seed n_params v1 ... vn".
"""
function read_initial_params(path::String, seed::Int, n_params::Int)
    for line in eachline(path)
        if startswith(line, "#")
            continue
        end
        fields = split(line)
        if parse(Int, fields[1]) == seed && parse(Int, fields[2]) == n_params
            return parse.(Float64, fields[3:end])
        end
    end
    error("no initial parameters for seed $seed and $n_params parameters in $path")
end

"""Fraction of test samples whose argmax of P(Yao qubit `readout`) matches the one-hot label."""
function test_accuracy(circuit, data, readout::Int)
    n_test = size(data.states_test, 2)
    correct = 0
    for k in 1:n_test
        out = ArrayReg(data.states_test[:, k:k]) |> circuit
        correct += argmax(Yao.probs(density_matrix(out, (readout,)))) == argmax(data.labels_test[k, :])
    end
    return correct / n_test
end

"""Build, train and test one configuration, then save the accuracy as a one-element .npy unless it exists."""
function run_and_save(out_file::String, data_dir::String, nbit::Int, ell::Int, tau::Float64, alpha::Float64, h::Float64;
                      seed::Int, readout::Int, train_options...)
    if isfile(out_file)
        println("[SKIP] $out_file")
        return
    end
    start = time()
    data = load_states(data_dir, nbit)
    circuit = da_circuit(nbit, ell, tau / kac_norm(nbit, alpha), alpha, h)
    train_circuit!(circuit, data, readout; seed = seed, train_options...)
    accuracy = test_accuracy(circuit, data, readout)
    mkpath(dirname(out_file))
    npzwrite(out_file, [accuracy])
    println("acc=$(round(accuracy, digits = 4))  $(round(Int, time() - start))s  $out_file")
end
