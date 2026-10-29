# Fig. 6 and Fig. 14: cluster-SPT phase classification of exact ground states at one (N, ell, alpha, seed).
# Trains on the 50 training states of the data file (25 SPT, 25 non-SPT).
# Label 0 (non-SPT) is outcome |0> on the readout qubit, label 1 (SPT) outcome |1>, so 1 - p0 is the SPT probability.
# 500 iterations, each on the whole training set; initial parameters from INIT_FILE (data/initial_params_julia110.txt).
# ENV: N_QUBITS, ELL, H_FIELD, ALPHA, TAU_PER_LAYER, SEED, POS_MEASURE, DATA_DIR, INIT_FILE, OUT_DIR
# Output: OUT_DIR/N{N_QUBITS}_l{ELL}_a{ALPHA}_s{SEED}.npz with test_prob_spt (441,), acc_test (1,), acc_train (1,),
#         loss_history (500,), final_params (n_params,)

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

const N_ITERATIONS_SPT = 500

"""One-hot labels of shape (samples, 2): column 1 is label 0 (non-SPT), column 2 is label 1 (SPT)."""
one_hot(labels) = Float64[labels[k] == c for k in eachindex(labels), c in (0, 1)]

"""p0 = P(|0>) on Yao qubit `readout` after the circuit, clipped to [0, 1], one value per column of states (2^N, samples)."""
function readout_p0(circuit, states, readout::Int)
    projector0 = put(nqudits(circuit), readout => matblock(ComplexF64[1 0; 0 0]))
    return [clamp(real(expect(projector0, ArrayReg(states[:, k:k]) => circuit)), 0.0, 1.0) for k in 1:size(states, 2)]
end

"""Fraction of samples whose prediction (SPT, label 1, if p0 < 1/2, else label 0) equals the label."""
function accuracy(p0, labels)
    predictions = [p < 0.5 ? 1 : 0 for p in p0]
    return count(predictions .== labels) / length(labels)
end

function run_one_point()
    nbit = parse(Int, ENV["N_QUBITS"])
    ell = parse(Int, ENV["ELL"])
    alpha = parse(Float64, ENV["ALPHA"])
    seed = parse(Int, ENV["SEED"])
    readout = parse(Int, ENV["POS_MEASURE"])
    out_file = joinpath(ENV["OUT_DIR"], "N$(nbit)_l$(ell)_a$(ENV["ALPHA"])_s$(seed).npz")
    if isfile(out_file)
        println("[SKIP] $out_file")
        return
    end
    start = time()

    # Rows of the stored arrays are samples; the circuit takes states as columns of shape (2^N, samples).
    file = npzread(joinpath(ENV["DATA_DIR"], "cluster_spt_N$(nbit).npz"))
    train_labels = file["train_labels"]
    test_labels = file["test_labels"]
    train_states = permutedims(file["train_states"])
    test_states = permutedims(file["test_states"])
    data = (states_train = train_states, labels_train = one_hot(train_labels))

    circuit = da_circuit(nbit, ell, parse(Float64, ENV["TAU_PER_LAYER"]) / kac_norm(nbit, alpha), alpha,
                         parse(Float64, ENV["H_FIELD"]))
    initial_params = read_initial_params(ENV["INIT_FILE"], seed, nparameters(circuit))
    loss_history = Float64[]
    train_circuit!(circuit, data, readout; seed = seed, n_iterations = N_ITERATIONS_SPT,
                   initial_params = initial_params, loss_history = loss_history)

    p0_train = readout_p0(circuit, train_states, readout)
    p0_test = readout_p0(circuit, test_states, readout)
    acc_test = accuracy(p0_test, test_labels)
    mkpath(dirname(out_file))
    npzwrite(out_file, Dict(
        "test_prob_spt" => 1 .- p0_test,
        "acc_test" => [acc_test],
        "acc_train" => [accuracy(p0_train, train_labels)],
        "loss_history" => loss_history,
        "final_params" => parameters(circuit),
    ))
    println("acc_test=$(round(acc_test, digits = 4))  $(round(Int, time() - start))s  $out_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
