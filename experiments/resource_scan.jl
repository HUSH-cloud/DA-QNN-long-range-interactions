# Fig. 2 columns ell = 1, 3, Fig. 3 and Fig. 11: pair 3 vs 8 at one (N, ell, h, alpha, tau).
# ENV: N_QUBITS, ELL, H_FIELD, ALPHA, TAU_PER_LAYER, TAU_IDX (file label), SEED, POS_MEASURE, DATA_DIR, OUT_DIR
# Output: OUT_DIR/acc_a{ALPHA}_t{TAU_IDX}_s{SEED}.npy

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

function run_one_point()
    run_and_save(
        joinpath(ENV["OUT_DIR"], "acc_a$(ENV["ALPHA"])_t$(ENV["TAU_IDX"])_s$(ENV["SEED"]).npy"),
        joinpath(ENV["DATA_DIR"], "3_8"),
        parse(Int, ENV["N_QUBITS"]), parse(Int, ENV["ELL"]), parse(Float64, ENV["TAU_PER_LAYER"]),
        parse(Float64, ENV["ALPHA"]), parse(Float64, ENV["H_FIELD"]);
        seed = parse(Int, ENV["SEED"]), readout = parse(Int, ENV["POS_MEASURE"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
