"""Fig. 12: same as Fig. 4 but a single panel at ell = 4.

Test accuracy versus N = 3..10 at h = 1.0 for five alpha, averaged over six digit pairs and 20 seeds.
Error bars: +-1 standard deviation (ddof = 0) over the 120 pair-seed runs, on alpha = 0.2 and alpha -> inf only.

Input:  RESULTS/fig4_sizes/acc_sitA_l4_N{N}_a{alpha}_p{pair}_s{seed}.npy
Output: OUT/fig_alphaN_acc_ell4.pdf
"""
from fig04_size_scan import draw

if __name__ == "__main__":
    draw([4], "fig_alphaN_acc_ell4")
