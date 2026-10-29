# Fig. 10: pair 3 vs 8 over the (h, alpha) plane.
# ENV: N_QUBITS, ELL, H_FIELD, ALPHA, TAU_PER_LAYER, SEED, POS_MEASURE, DATA_DIR, OUT_DIR
# Output: OUT_DIR/acc_h{H_FIELD}_a{ALPHA}_s{SEED}.npy

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

function run_one_point()
    run_and_save(
        joinpath(ENV["OUT_DIR"], "acc_h$(ENV["H_FIELD"])_a$(ENV["ALPHA"])_s$(ENV["SEED"]).npy"),
        joinpath(ENV["DATA_DIR"], "3_8"),
        parse(Int, ENV["N_QUBITS"]), parse(Int, ENV["ELL"]), parse(Float64, ENV["TAU_PER_LAYER"]),
        parse(Float64, ENV["ALPHA"]), parse(Float64, ENV["H_FIELD"]);
        seed = parse(Int, ENV["SEED"]), readout = parse(Int, ENV["POS_MEASURE"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
