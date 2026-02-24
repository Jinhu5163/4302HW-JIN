import numpy as np
import matplotlib.pyplot as plt

m_vals = np.array([40, 80, 160, 320, 640, 1280], dtype=float)
h_vals = 1.0 / m_vals


err_k1 = np.array([
    5.023920940926e-04,  # m=40
    1.257960768166e-04,  # m=80
    3.147470911301e-05,  # m=160
    7.871946502190e-06,  # m=320
    1.968398945792e-06,  # m=640
    4.921519615457e-07   # m=1280
], dtype=float)

err_k5 = np.array([
    1.265333861435e-02,  # m=40
    3.150730544203e-03,  # m=80
    7.872319001570e-04,  # m=160
    1.968214272043e-04,  # m=320
    4.921139569469e-05,  # m=640
    1.230387616560e-05   # m=1280
], dtype=float)
err_k10 = np.array([
    5.414122578700e-02,  # m=40
    1.327432757158e-02,  # m=80
    3.304162937906e-03,  # m=160
    8.253360918649e-04,  # m=320
    2.063139840221e-04,  # m=640
    5.158023630201e-05   # m=1280
], dtype=float)

def fit_order(h, e):
    """
    Fit log(e) = p*log(h) + b, return slope p (order) and intercept b.
    """
    coeffs = np.polyfit(np.log(h), np.log(e), 1)
    p = coeffs[0]
    b = coeffs[1]
    return p, b

def pairwise_orders(e):
    """
    Since h halves each time, estimated local order is log2(e_i / e_{i+1}).
    """
    return np.log2(e[:-1] / e[1:])


p1, b1 = fit_order(h_vals, err_k1)
p5, b5 = fit_order(h_vals, err_k5)
p10, b10 = fit_order(h_vals, err_k10)

print("Estimated convergence orders from log-log fit:")
print(f"  k=1  : p = {p1:.6f}")
print(f"  k=5  : p = {p5:.6f}")
print(f"  k=10 : p = {p10:.6f}")

print("\nPairwise estimated orders (log2(E_coarse/E_fine)):")
print("  k=1 :", pairwise_orders(err_k1))
print("  k=5 :", pairwise_orders(err_k5))
print("  k=10:", pairwise_orders(err_k10))


ref_scale = err_k5[0] / (h_vals[0] ** 2) if err_k5[0] > 0 else 1.0
err_ref = ref_scale * h_vals**2


plt.figure(figsize=(9, 6))
plt.loglog(h_vals, err_k1, 'o-', label=f'k=1 (fit slope ~ {p1:.2f})')
plt.loglog(h_vals, err_k5, 's-', label=f'k=5 (fit slope ~ {p5:.2f})')
plt.loglog(h_vals, err_k10, '^-', label=f'k=10 (fit slope ~ {p10:.2f})')
plt.loglog(h_vals, err_ref, '--', label=r'Reference $O(h^2)$')



plt.xlabel('h = 1/m')
plt.ylabel('Relative error')
plt.title('BVP Convergence of Error vs h (gamma = 0)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
