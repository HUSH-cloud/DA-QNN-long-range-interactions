# Initial circuit parameters of the SPT runs (Figs. 6, 14) and the XXZ runs (Figs. 7, 16):
# randn(MersenneTwister(seed), n_params) as drawn by Julia 1.10. Julia 1.12 draws a different stream for the
# same seed, so the SPT and XXZ experiment scripts read these stored values (INIT_FILE) instead of calling randn.
# Run from the repository root with Julia 1.10 (Random stdlib only, no project environment):
#   julia +1.10 --startup-file=no data/initial_params_julia110.jl [output file]
# Output: data/initial_params_julia110.txt (default), one header line, then one line "seed n_params v1 ... vn"
# per seed and parameter count. repr prints the shortest decimal that parses back to the same Float64.

using Random

const SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2718, 3141, 5678]
# n_params = 3 N (ell + 1). SPT: N = 6, 8, 10 with ell = 1, 2, and N = 8 with ell = 4, 8. XXZ: N = 8, ell = 2.
const N_PARAMS = [36, 48, 54, 60, 72, 90, 120, 216]

@assert v"1.10" <= VERSION < v"1.11" "run with julia +1.10; Julia 1.12 does not reproduce the Julia 1.10 randn stream"

out_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "initial_params_julia110.txt")
mkpath(dirname(abspath(out_file)))
open(out_file, "w") do io
    println(io, "# Initial circuit parameters of the SPT and XXZ runs, randn(MersenneTwister(seed), n_params) drawn with ",
            "Julia 1.10 by data/initial_params_julia110.jl; each line holds seed, n_params and the n_params values.")
    for seed in SEEDS, n_params in N_PARAMS
        values = randn(MersenneTwister(seed), n_params)
        println(io, seed, " ", n_params, " ", join(repr.(values), " "))
    end
end
