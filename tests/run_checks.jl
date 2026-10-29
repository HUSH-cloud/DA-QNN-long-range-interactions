# Checks of the conventions the paper relies on. Run from the repository root:
#   julia --project=. tests/run_checks.jl [pair folder, default data/digit_data/3_8]
# Checks 1-3 cover the MNIST data and its encoding, checks 4-7 the analog block, checks 8-10 the cluster-SPT and XXZ
# data in data/spt_data and data/xxz_data.
# Every check throws on failure; the last line prints "all checks passed".

using Yao, LinearAlgebra, NPZ, Statistics
include(joinpath(@__DIR__, "..", "src", "daqnn.jl"))

pair_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "digit_data", "3_8")
spt_file = joinpath(@__DIR__, "..", "data", "spt_data", "cluster_spt_N8.npz")
xxz_file = joinpath(@__DIR__, "..", "data", "xxz_data", "xxz_N8_grid50.npz")

# 1. Qubit labels. Encoded qubit j (row j of the data file) sits on chain site N + 1 - j, and the
#    readout site 1 is therefore the last encoded qubit.
let n = 8, theta = zeros(n), phi = zeros(n)
    theta[1] = pi                                              # encoded qubit 1 flipped to |1>
    reg = ArrayReg(reshape(product_state(theta, phi), :, 1))
    @assert real(Yao.probs(density_matrix(reg, (n,)))[2]) ≈ 1 "encoded qubit 1 must be site N"
    @assert real(Yao.probs(density_matrix(reg, (1,)))[1]) ≈ 1 "site 1 must be encoded qubit N"
end

# 2. On the real data the readout probability of the encoded input state equals cos^2(theta/2) of the
#    last encoded qubit, and the center-out placement puts PC1 (largest variance) on site N/2 + 1.
let n = 8, x = npzread(joinpath(pair_dir, "x_test_n$(n).npy")), d = load_states(pair_dir, n)
    p0 = real(Yao.probs(density_matrix(ArrayReg(d.states_test[:, 1:1]), (1,)))[1])
    @assert p0 ≈ cos(x[n, 1] / 2)^2 "readout site 1 does not carry encoded qubit N"
    pc1 = argmax(vec(var(x[1:n, :]; dims = 2)))
    @assert n + 1 - pc1 == n ÷ 2 + 1 "PC1 of the center-out placement must sit on site N/2 + 1, found site $(n + 1 - pc1)"
end

# 3. Labels. Column 1 of y is digit a of the pair, which the classifier assigns to outcome |0> on the readout.
let y = npzread(joinpath(pair_dir, "y_train.npy"))
    @assert size(y, 2) == 2 && all(sum(y; dims = 2) .== 1) "labels must be one-hot with two columns"
    @assert abs(sum(y[:, 1]) - size(y, 1) / 2) <= 1 "training set must be balanced"
end

# 4. Kac time. The code's Hamiltonian divided by kac_norm is Eq. (1) of the paper, built here from Pauli operators,
#    so evolving it for tau / kac_norm equals evolving Eq. (1) for tau.
let n = 6, alpha = 1.0, h = 1.0, tau = 0.4pi
    kac = kac_norm(n, alpha)
    zz = sum(abs(i - j)^(-alpha) * put(n, (i, j) => kron(Z, Z)) for i in 1:n for j in (i + 1):n)
    H_eq1 = Matrix(mat(-(1 / kac) * zz - h * sum(put(n, i => X) for i in 1:n)))     # Eq. (1) with J_0 = 1
    H = Matrix(lrtfim_hamiltonian(n, alpha, h))
    @assert opnorm(H ./ kac - H_eq1) < 1e-12 "H / kac_norm is not Eq. (1)"
    @assert opnorm(exp(-im * tau / kac * H) - exp(-im * tau * H_eq1)) < 1e-12 "Kac time convention broken"
end

# 5. The analog block is unitary, and the full circuit preserves the norm of a state.
let n = 8, t = 2pi / kac_norm(n, 1.0)
    c = da_circuit(n, 2, t, 1.0, 1.0)
    U = Yao.mat(ComplexF64, c[2])
    @assert opnorm(U' * U - I) < 1e-10 "analog block is not unitary"
    @assert abs(norm(statevec(rand_state(n) |> c)) - 1) < 1e-10 "state norm changes through the circuit"
end

# 6. The Hamiltonian is reflection symmetric, so a placement counted from either end of the chain is equivalent.
let n = 6, H = Matrix(lrtfim_hamiltonian(n, 1.0, 1.0))
    P = zeros(2^n, 2^n)
    for s in 0:(2^n - 1)
        r = sum(((s >> k) & 1) << (n - 1 - k) for k in 0:(n - 1))
        P[r + 1, s + 1] = 1
    end
    @assert opnorm(P * H * P' - H) < 1e-12 "Hamiltonian is not reflection symmetric"
end

# 7. The nearest-neighbor limit is the open chain with unit couplings and no Kac factor.
let n = 4, h = 0.7
    H_nn = -sum(put(n, (i, i + 1) => kron(Z, Z)) for i in 1:(n - 1)) - h * sum(put(n, i => X) for i in 1:n)
    @assert opnorm(Matrix(lrtfim_hamiltonian(n, Inf, h)) - Matrix(mat(H_nn))) < 1e-12 "alpha = Inf is not the nearest-neighbor chain"
    @assert kac_norm(n, Inf) == 1.0
end

"""Value of a cubic with coefficients in numpy.polyval order (highest power first) at x."""
cubic(coefficients, x) = evalpoly(x, reverse(coefficients))

"""Norm of H psi - <psi|H|psi> psi, and <psi|H|psi> - E_0 with E_0 the lowest eigenvalue, for a normalized state psi."""
function eigen_residuals(H, psi)
    energy = real(dot(psi, H * psi))
    return norm(H * psi - energy * psi), energy - minimum(eigvals(Hermitian(H)))
end

# 8. Cluster-SPT labels. Label 1 (SPT) exactly when anti(h1) < h2 < para(h1), and every test point lies at least
#    0.01 in h2 from both boundaries.
let d = npzread(spt_file)
    for (params, labels) in ((d["train_params"], d["train_labels"]), (d["test_params"], d["test_labels"]))
        for k in eachindex(labels)
            h1, h2 = params[k, 1], params[k, 2]
            inside = cubic(d["anti_boundary"], h1) < h2 < cubic(d["para_boundary"], h1)
            @assert labels[k] == (inside ? 1 : 0) "SPT label of point $k at (h1, h2) = ($h1, $h2) breaks the boundary rule"
        end
    end
    for k in axes(d["test_params"], 1)
        h1, h2 = d["test_params"][k, 1], d["test_params"][k, 2]
        distance = min(abs(h2 - cubic(d["anti_boundary"], h1)), abs(h2 - cubic(d["para_boundary"], h1)))
        @assert distance >= 0.01 "test point $k lies $distance from a phase boundary, closer than the margin 0.01"
    end
end

# 9. Ground states. A stored cluster-SPT state solves Eq. (3) and a stored XXZ grid state solves Eq. (4), both built here
#    from Pauli operators (open chain with J = 1; periodic chain with J6 = 1, so J3 is the stored ratio J3/J6).
let d = npzread(spt_file), n = 8, k = 1
    h1, h2 = d["train_params"][k, 1], d["train_params"][k, 2]
    H_spt = -sum(put(n, (i, i + 1, i + 2) => kron(Z, X, Z)) for i in 1:(n - 2)) - h1 * sum(put(n, i => X) for i in 1:n) -
            h2 * sum(put(n, (i, i + 1) => kron(X, X)) for i in 1:(n - 1))
    residual, excess = eigen_residuals(Matrix(mat(H_spt)), d["train_states"][k, :])
    @assert residual < 1e-10 && excess < 1e-10 "cluster-SPT training state $k is not a ground state of Eq. (3)"
end

"""Eq. (4) on n periodic sites with J6 = 1 and J3 = j3_ratio, as a dense matrix of shape (2^n, 2^n)."""
function xxz_hamiltonian(n, j3_ratio, lam)
    bond(P, i, d) = put(n, Tuple(sort([i, mod1(i + d, n)])) => kron(P, P))
    xy = sum(bond(X, i, 1) + bond(Y, i, 1) + lam * (bond(X, i, 2) + bond(Y, i, 2)) for i in 1:n)
    zz = sum(bond(Z, i, 1) + lam^2 * bond(Z, i, 2) for i in 1:n)
    return Matrix(mat(j3_ratio * xy + zz))
end

let d = npzread(xxz_file), n = 8, k = 1234
    H_xxz = xxz_hamiltonian(n, d["test_params"][k, 1], d["test_params"][k, 2])
    residual, excess = eigen_residuals(H_xxz, d["test_states"][k, :])
    @assert residual < 1e-10 && excess < 1e-10 "XXZ grid state $k is not a ground state of Eq. (4)"
end

# 10. zAFM anchor. Its two lowest levels form a doublet split by less than 1e-6, and the stored training state is
#     the second eigenvector (index 1 counted from 0) up to a global phase.
let d = npzread(xxz_file), n = 8
    energies, vectors = eigen(Hermitian(xxz_hamiltonian(n, d["anchor_params"][1, 1], d["anchor_params"][1, 2])))
    @assert energies[2] - energies[1] < 1e-6 "zAFM anchor: the two lowest levels are split by $(energies[2] - energies[1])"
    @assert abs(dot(vectors[:, 2], d["anchor_states"][1, :])) > 1 - 1e-8 "zAFM anchor state is not eigenvector index 1"
end

println("all checks passed")
