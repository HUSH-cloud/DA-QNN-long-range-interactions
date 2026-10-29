# Fig. 2 main grid: pair 3 vs 8, alpha = 1, tau = 0.2 pi ... 1.6 pi, ell = 2, 4, 6, 8, 10.
# ENV: N_QUBITS, H_FIELD, TAU_IDX (1-8), ELL_IDX (1-5), SEED, POS_MEASURE, DATA_DIR, OUT_DIR
# Output: OUT_DIR/acc_t{TAU_IDX}_l{ELL_IDX}.npy

include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

const TAU_VALUES = collect((pi / 5):(pi / 5):(8 * pi / 5))
const ELL_VALUES = [2, 4, 6, 8, 10]

function run_one_point()
    tau_idx = parse(Int, ENV["TAU_IDX"])
    ell_idx = parse(Int, ENV["ELL_IDX"])
    run_and_save(
        joinpath(ENV["OUT_DIR"], "acc_t$(tau_idx)_l$(ell_idx).npy"),
        joinpath(ENV["DATA_DIR"], "3_8"),
        parse(Int, ENV["N_QUBITS"]), ELL_VALUES[ell_idx], TAU_VALUES[tau_idx], 1.0, parse(Float64, ENV["H_FIELD"]);
        seed = parse(Int, ENV["SEED"]), readout = parse(Int, ENV["POS_MEASURE"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_one_point()
end
