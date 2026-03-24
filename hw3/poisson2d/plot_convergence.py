import matplotlib.pyplot as plt

iters = [0,1,2,3,4,5,6]

residuals = [
    1.838084130533,
    1.652067028742,
    1.157486858538,
    7.086743981894e-02,
    3.070239545525e-03,
    6.616007282621e-06,
    1.035838953976e-10
]

plt.figure()
plt.semilogy(iters, residuals, marker='o')

plt.xlabel('Newton iteration')
plt.ylabel('||F(u)||_2')
plt.title('Convergence history of SNES')
plt.grid(True)

plt.savefig('convergence.png', dpi=300)
plt.show()