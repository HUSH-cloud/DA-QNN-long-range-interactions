# Fig. 4, Fig. 8 and Fig. 12: one digit pair at one (N, ell, alpha).
# SIT=sitA labels the h = 1 runs in the stored file names.
# ENV: SIT, N_QUBITS, ELL, ALPHA, PAIR_A, PAIR_B, TAU_PER_LAYER, SEED, POS_MEASURE, DATA_DIR, OUT_DIR
# Output: OUT_DIR/acc_{SIT}_l{ELL}_N{N_QUBITS}_a{ALPHA}_p{PAIR_A}{PAIR_B}_s{SEED}.npy

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

const FIELD_BY_SIT = Dict("sitA" => 1.0)

function run_one_point()
    nbit = parse(Int, ENV["N_QUBITS"])
    ell = parse(Int, ENV["ELL"])
    pair = "$(ENV["PAIR_A"])$(ENV["PAIR_B"])"
    run_and_save(
        joinpath(ENV["OUT_DIR"], "acc_$(ENV["SIT"])_l$(ell)_N$(nbit)_a$(ENV["ALPHA"])_p$(pair)_s$(ENV["SEED"]).npy"),
        joinpath(ENV["DATA_DIR"], "$(ENV["PAIR_A"])_$(ENV["PAIR_B"])"),
        nbit, ell, parse(Float64, ENV["TAU_PER_LAYER"]), parse(Float64, ENV["ALPHA"]), FIELD_BY_SIT[ENV["SIT"]];
        seed = parse(Int, ENV["SEED"]), readout = parse(Int, ENV["POS_MEASURE"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
