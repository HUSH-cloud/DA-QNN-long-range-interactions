# Fig. 5 and Fig. 13: pair 3 vs 8 with one PCA-to-site placement (a folder written by data/prepare_data.py), read out on site Q_MEAS.
# ENV: ENC_NAME (placement folder), Q_MEAS, N_QUBITS, ELL, H_FIELD, ALPHA, TAU_PER_LAYER, SEED, ENCODING_ROOT, OUT_DIR
# Output: OUT_DIR/acc_{ENC_NAME}_q{Q_MEAS}_N{N_QUBITS}_l{ELL}_a{ALPHA}_h{H_FIELD}_s{SEED}.npy

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

function run_one_point()
    readout = parse(Int, ENV["Q_MEAS"])
    nbit = parse(Int, ENV["N_QUBITS"])
    ell = parse(Int, ENV["ELL"])
    run_and_save(
        joinpath(ENV["OUT_DIR"], "acc_$(ENV["ENC_NAME"])_q$(readout)_N$(nbit)_l$(ell)_a$(ENV["ALPHA"])_h$(ENV["H_FIELD"])_s$(ENV["SEED"]).npy"),
        joinpath(ENV["ENCODING_ROOT"], ENV["ENC_NAME"]),
        nbit, ell, parse(Float64, ENV["TAU_PER_LAYER"]),
        parse(Float64, ENV["ALPHA"]), parse(Float64, ENV["H_FIELD"]);
        seed = parse(Int, ENV["SEED"]), readout = readout, prob_floor = 1e-15, prob_offset = 1e-15,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
