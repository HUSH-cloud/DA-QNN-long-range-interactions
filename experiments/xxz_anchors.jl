# Fig. 7 and Fig. 16: unsupervised XXZ run, one circuit trained on one anchor ground state at one (alpha, seed).
# Loss L = (1/N) sum_i P_i(0), the probability of reading 0 averaged over the N qubits. AdaGrad (learning rate
# LEARNING_RATE = 0.1, epsilon 1e-8) for 50 steps on the single anchor state; L is recorded before every update and
# once at the end (51 values).
# The trained circuit then gives the loss L of every ground state on the 50 x 50 grid.
# N = 8 comes from the data file; initial parameters from INIT_FILE (data/initial_params_julia110.txt).
# ENV: ANCHOR (0 zAFM, 1 qzAFM, 2 XY-QLRO, 3 VBS), ELL, H_FIELD, ALPHA, TAU_PER_LAYER, SEED, DATA_FILE, INIT_FILE, OUT_DIR
# Output: OUT_DIR/a{ALPHA}_anchor{ANCHOR}_s{SEED}.npz with test_loss (2500,), loss_history (51,), final_params (n_params,)

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

const N_STEPS = 50

"""L = (1/N) sum_i P_i(0) of one state (2^N, 1) after the circuit, with one projector |0><0| per qubit."""
mean_p0(projectors, reg, circuit) = sum(real(expect(op, reg => circuit)) for op in projectors) / length(projectors)

"""Train in place from initial_params; returns the loss history, shape (N_STEPS + 1,)."""
function train_on_anchor!(circuit, anchor_state, initial_params)
    nbit = nqudits(circuit)
    projectors = [put(nbit, i => matblock(ComplexF64[1 0; 0 0])) for i in 1:nbit]
    reg = ArrayReg(reshape(anchor_state, :, 1))
    params = copy(initial_params)
    dispatch!(circuit, params)
    accumulated = zeros(length(params))
    loss_history = Float64[]
    for step in 0:N_STEPS
        push!(loss_history, mean_p0(projectors, reg, circuit))
        if step == N_STEPS
            break
        end
        grads = zeros(length(params))
        for op in projectors
            grads .+= expect'(op, reg => circuit).second
        end
        grads ./= nbit
        accumulated .+= grads .^ 2
        params = params .- LEARNING_RATE .* grads ./ (sqrt.(accumulated) .+ 1e-8)
        dispatch!(circuit, params)
    end
    return loss_history
end

"""
L of every column of states (2^N, samples), shape (samples,). The probability of reading 0 averaged over the
qubits equals sum_b w_b |(U psi)_b|^2 with U the circuit matrix and w_b = (number of 0 bits of b) / N.
"""
function test_losses(circuit, states)
    nbit = nqudits(circuit)
    weights = [(nbit - count_ones(b)) / nbit for b in 0:(2^nbit - 1)]
    amplitudes = Matrix(mat(circuit)) * states
    return vec(sum(weights .* abs2.(amplitudes); dims = 1))
end

function run_one_point()
    anchor = parse(Int, ENV["ANCHOR"])
    ell = parse(Int, ENV["ELL"])
    alpha = parse(Float64, ENV["ALPHA"])
    seed = parse(Int, ENV["SEED"])
    out_file = joinpath(ENV["OUT_DIR"], "a$(ENV["ALPHA"])_anchor$(anchor)_s$(seed).npz")
    if isfile(out_file)
        println("[SKIP] $out_file")
        return
    end
    start = time()

    # Rows of the stored arrays are states; the circuit takes them as columns.
    file = npzread(ENV["DATA_FILE"])
    anchor_state = file["anchor_states"][anchor + 1, :]
    test_states = permutedims(file["test_states"])
    nbit = round(Int, log2(length(anchor_state)))

    circuit = da_circuit(nbit, ell, parse(Float64, ENV["TAU_PER_LAYER"]) / kac_norm(nbit, alpha), alpha,
                         parse(Float64, ENV["H_FIELD"]))
    initial_params = read_initial_params(ENV["INIT_FILE"], seed, nparameters(circuit))
    loss_history = train_on_anchor!(circuit, anchor_state, initial_params)

    test_loss = test_losses(circuit, test_states)
    mkpath(dirname(out_file))
    npzwrite(out_file, Dict(
        "test_loss" => test_loss,
        "loss_history" => loss_history,
        "final_params" => parameters(circuit),
    ))
    println("loss=$(round(loss_history[end], digits = 4))  $(round(Int, time() - start))s  $out_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
