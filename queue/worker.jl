# Load one experiment script once, then run every task in a shard file.
#   julia worker.jl <experiment.jl> <shard.txt>
# Each task line is KEY=VAL;KEY=VAL;... and is copied into ENV before run_one_point().

const SCRIPT, SHARD = ARGS
include(SCRIPT)

const TASKS = filter(line -> !isempty(strip(line)), readlines(SHARD))
for (k, line) in enumerate(TASKS)
    for entry in split(line, ';'; keepempty = false)
        key, value = split(entry, '='; limit = 2)
        ENV[strip(key)] = strip(value)
    end
    start = time()
    try
        run_one_point()
    catch err
        println("FAILED ($line): ", sprint(showerror, err))
    end
    println("done $k/$(length(TASKS)) in $(round(Int, time() - start))s")
    flush(stdout)
end
