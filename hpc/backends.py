"""Where the Lindblad right-hand side actually gets evaluated.

Every backend computes the SAME quantity as the notebook's `lindblad_rhs`,

    drho/dt = K rho + rho K^dag + sum_k L_k rho L_k^dag,
    K = -i H - (1/2) sum_k L_k^dag L_k,

and the same Uhlmann fidelity.  They differ only in who multiplies the matrices:

  scipy  the notebook's own code, unchanged.  Single-threaded.  The reference.
  mkl    same products through MKL's threaded sparse x dense kernel
         (pip install sparse-dot-mkl).  Scales with --cpus-per-task.
  gpu    cupy / cuSPARSE / cuSOLVER.  Needs a card with enough memory to hold
         about eight d_sec x d_sec complex matrices.

Why this exists: the cost per RK4 step measured on a laptop is 2.1 s at L = 12,
49 s at L = 14 and an extrapolated ~880 s at L = 16.  At L = 16 that is roughly a
month of single-core time, so the largest size is only reachable if the sparse
products thread or move to a GPU.

Both accelerated backends use the identity

    rho @ B^dag == (B @ rho^dag)^dag

to turn every dense-times-sparse product into sparse-times-dense, which is the only
orientation MKL and cuSPARSE implement.  That is an algebraic identity, not an
assumption about rho -- but `run_relax.py --selftest --backend <name>` reproduces the
notebook's L = 6 and L = 8 numbers on whichever backend you are about to use, so a
backend that is subtly wrong never gets as far as a production size.
"""
import numpy as np
import scipy.sparse as sp


class ScipyBackend:
    """The notebook's own path, imported rather than re-implemented."""
    name = "scipy"

    def __init__(self):
        from thermal_hpc import lindblad_rhs, fidelity_to
        self._rhs, self._fid = lindblad_rhs, fidelity_to

    def array(self, a):
        return np.asarray(a)

    def to_numpy(self, a):
        return np.asarray(a)

    def rhs(self, H_sec, L_sec):
        return self._rhs(H_sec, L_sec)

    def fidelity_to(self, target):
        return self._fid(target)

    def info(self):
        return "scipy sparse, single-threaded"


class _SparseDenseBackend:
    """Shared skeleton for the backends that only do sparse x dense."""

    def _spmm(self, A, B):                      # sparse A, dense B -> dense
        raise NotImplementedError

    def _H(self, x):                            # conjugate transpose, dense
        return x.conj().T

    def rhs(self, H_sec, L_sec):
        K_h = (-1j * H_sec
               - 0.5 * sum(Lk.conj().T.tocsr() @ Lk for Lk in L_sec)).tocsr()
        K = self.sparse(K_h)
        Ls = [self.sparse(Lk.tocsr()) for Lk in L_sec]
        spmm, H = self._spmm, self._H

        def rhs(rho):
            # K rho + rho K^dag, the second written as (K rho^dag)^dag
            out = spmm(K, rho) + H(spmm(K, H(rho)))
            for Lk in Ls:
                A = spmm(Lk, rho)               # L_k rho
                out = out + H(spmm(Lk, H(A)))   # (L_k (L_k rho)^dag)^dag
            return out
        return rhs

    def fidelity_to(self, target):
        """F(., target) with the target's square root hoisted out, as in the notebook."""
        S = self._sqrtm_psd(self.array(target))

        def F(rho):
            M = self._sqrtm_psd(rho) @ S
            return float(min(max(float(self._svdvals(M).sum()), 0.0), 1.0))
        return F


class ReferenceBackend(_SparseDenseBackend):
    """The accelerated skeleton, but with scipy doing the products.

    Neither MKL nor cupy can be installed on a laptop, yet both run through
    `_SparseDenseBackend` -- the reorientation of every dense-times-sparse product
    into sparse-times-dense, and the fidelity written against a generic array
    library.  This backend exercises exactly that shared code with the one library
    that is always present, so

        python run_relax.py --selftest --backend reference

    verifies the skeleton anywhere.  It is not faster than `scipy` (it is slightly
    slower, from the extra transposes) and is only meant for that check.
    """
    name = "reference"

    def sparse(self, A):
        return A.tocsr()

    def array(self, a):
        return np.asarray(a, dtype=np.complex128)

    def to_numpy(self, a):
        return np.asarray(a)

    def _spmm(self, A, B):
        return A @ B

    def _sqrtm_psd(self, A):
        w, v = np.linalg.eigh(A)
        return (v * np.sqrt(np.clip(w.real, 0.0, None))) @ v.conj().T

    def _svdvals(self, M):
        return np.linalg.svd(M, compute_uv=False)

    def info(self):
        return "scipy products through the accelerated skeleton (for testing)"


class MklBackend(_SparseDenseBackend):
    name = "mkl"

    def __init__(self):
        from sparse_dot_mkl import dot_product_mkl
        self._dot = dot_product_mkl

    def sparse(self, A):
        return A.tocsr()

    def array(self, a):
        return np.ascontiguousarray(a, dtype=np.complex128)

    def to_numpy(self, a):
        return np.asarray(a)

    def _spmm(self, A, B):
        return self._dot(A, np.ascontiguousarray(B))

    def _sqrtm_psd(self, A):
        w, v = np.linalg.eigh(A)
        return (v * np.sqrt(np.clip(w.real, 0.0, None))) @ v.conj().T

    def _svdvals(self, M):
        return np.linalg.svd(M, compute_uv=False)

    def info(self):
        import os
        return "MKL threaded sparse, MKL_NUM_THREADS={}".format(
            os.environ.get("MKL_NUM_THREADS", "unset"))


class GpuBackend(_SparseDenseBackend):
    name = "gpu"

    def __init__(self):
        import cupy
        import cupyx.scipy.sparse as cusp
        self.cp, self.cusp = cupy, cusp

    def sparse(self, A):
        return self.cusp.csr_matrix(A.astype(np.complex128))

    def array(self, a):
        return self.cp.asarray(np.asarray(a, dtype=np.complex128))

    def to_numpy(self, a):
        return self.cp.asnumpy(a)

    def _spmm(self, A, B):
        return A @ B

    def _sqrtm_psd(self, A):
        cp = self.cp
        w, v = cp.linalg.eigh(A)
        return (v * cp.sqrt(cp.clip(w.real, 0.0, None))) @ v.conj().T

    def _svdvals(self, M):
        return self.cp.linalg.svd(M, compute_uv=False)

    def info(self):
        dev = self.cp.cuda.Device()
        free, total = dev.mem_info
        return "cupy on device {}, {:.0f} GiB free of {:.0f} GiB".format(
            dev.id, free / 2 ** 30, total / 2 ** 30)


BACKENDS = {"scipy": ScipyBackend, "reference": ReferenceBackend,
            "mkl": MklBackend, "gpu": GpuBackend}


def get_backend(name):
    if name not in BACKENDS:
        raise SystemExit("unknown backend {!r}; pick from {}".format(
            name, ", ".join(BACKENDS)))
    try:
        return BACKENDS[name]()
    except ImportError as exc:
        raise SystemExit(
            "backend {!r} is not available here ({}).\n"
            "  mkl needs:  pip install sparse-dot-mkl\n"
            "  gpu needs:  pip install cupy-cuda12x   (and a GPU node)".format(
                name, exc))
