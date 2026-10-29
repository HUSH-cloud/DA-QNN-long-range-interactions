"""Fig. 13: the encoding comparison of Fig. 5 repeated at ell = 4.

N = 8, h = 1.0, ell = 4, digit pair 3 vs 8; mean over 10 seeds (SEEDS_10). Encodings E1-E6 as in fig05_encodings.py.
Error bars: 95% confidence interval of the mean over seeds (Student t, ddof=1), drawn on E1 and E6 only.
Input:  RESULTS/fig5_encodings/acc_{placement}_q{site}_N8_l4_a{alpha}_h1.0_s{seed}.npy
Output: OUT/fig_encodings_ell4.pdf
"""
from fig05_encodings import plot_encodings

if __name__ == "__main__":
    plot_encodings(ell=4, stem="fig_encodings_ell4")
