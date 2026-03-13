**Ultimate Goal: Prepare a thermal state
$$e^{- \beta H} / Z$$
approximately for a large system.**

## Schwinger Hamiltonian
The following function creates the Schwinger model Hamiltonian as a SparsePauliOp.
$$H = H_{kin} + H_{mass} + H_{elec}$$
where
$$H_{kin} \equiv \frac{1}{2a} \sum_{i = 0}^{L - 2} (\sigma_i^{+} \sigma_{i+1}^{-} + \sigma_{i+1}^{+} \sigma_i^{-})$$
$$H_{mass} \equiv m \sum_{i=0}^{L-1} O_i, O_i \equiv (-1)^i \frac{\sigma_i^z + 1}{2a}$$
$$H_{elec} \equiv \frac{a}{2} \sum_{i=0}^L E_i^2, E_i \equiv e(l_0 + \frac{1}{2} \sum_{j=0}^{i-1}(\sigma_j^z + (-1)^j)$$

a: lattice spacing

n: index of lattice position $x = na$

L: number of qubits

$\sigma^{\pm} = (\sigma_x \pm i\sigma_y)/2)$: creation/annihilation operators

We set $l_0 = 0$.

### Environmental Correlator D(x)
Regarding the environmental correlator, $x_1 = n_1 a$ and $x_2 = n_2 a$ are discrete spatial coordinates.

For short range correlations, use a delta function:
$$D_{\delta}(x) = D_0 \delta_{0x}.$$

For intermediate-range correlations, use a Gaussian:
$$D_G(x) = D_0 \exp(-\frac{x^2}{2\sigma^2}) \equiv D_0 G(x, \sigma).$$

For long-range correlations, use a constant function:
$$D_C(x) = D_0.$$

### Now building the Liouvillian superoperator
In building the Liouvillian superoperator, the following ways of rewriting the Lindblad equation are used:

For the Hamiltonian part,
$$-i[H, \rho] \rightarrow -i (I \otimes H - H^T \otimes I) vec(\rho)$$

For the dissipator part,
$$a^2 \sum_{x_1, x_2} D(x_1 - x_2) [(L^{\dagger}(x_1))^T \otimes L(x_2) - \frac{1}{2}(I \otimes M) - \frac{1}{2} (M^T \otimes I)] vec(\rho)$$ where $M = L^{\dagger}(x_1)L(x_2).$

We can do this thanks to the following identity:
$$vec(A \rho B) = (B^T \otimes A) vec(\rho).$$

This way, the Lindblad equation is a linear ODE:
$$\frac{d}{dt}|\rho> = \mathcal{L}|\rho>$$

### Checking J Operator Construction
Block (j, 0) in the aux basis should equal $L_j$, and block (0, j) should equal $L^{\dagger}_j$ with all other blocks zero.

The block structure should match:
$$J[j,0] = L_j$$
$$J[0,j] = L^{\dagger}_j$$

## Measuring the Energy Density of a Single Qubit
$$H = \sum_n h_n$$
$$\epsilon_n \equiv \langle h_n \rangle_t = Tr(h_n \rho(t))$$

In our case, our Hamiltonian is a sum of Pauli strings. We can define $h_n$ by assigning each Pauli term's energy to sites:
 - If a term acts on k qubits, each qubit gets $1/k$ of its coefficient.
 - 1-qubit terms belong fully to a qubit.
 - Identity terms are just a global shift; we can ignore them.

 