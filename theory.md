# One-Sided Jacobi SVD — Theory Notes

*This document outlines the theoretical structure. Fill in mathematical content and algorithm details progressively as the implementation develops.*

---

## Algorithm Overview

High-level description of what the one-sided Jacobi SVD does and why it is useful.

Computing the singular value decomposition of a general matrix with high relative accuracy.

Apply orthogonal transformations from one side only (right-multiplication by Jacobi rotations, and sometimes it is called as Givens rotation in literature) to iteratively orthogonalize the columns of $A$.

The two-sided Jacobi is completely different from the one-sided Jacobi. Two-sided Jacobi is created for computing eigenvalues of a symmetric matrix, whereas the one-sided Jacobi is created for computing singular values of a general matrix. 

The most important part of the one-sided Jacobi algorithm is its stopping criterion, which will be seen in Pseudo-code.

---

## Mathematical Derivation

Detailed derivations behind the algorithm. Each subsection is a placeholder — replace TODOs with actual formulas and reasoning.

- Suppose the input matrix is $m\times n$ where $m\ge n$. At the $k$th step, we have pivot columns $(i_k, j_k)$, where $(i_k,j_k)$ are chosen from $(1,2),...,(1,n),(2,3),...,(2,n),...(m,n)$.

- The rotation matrix at $k$th step, the current matrix is $A^{(k)}$. In order to generate rotation matrix $J(i,j)$ where $(i,j)$ is the abbreviation of $(i_k, j_k)$. We first compute the angle $\phi$, then the orthogonal matrix $V_k$ that will be applied on the right of the matrix is defined as the identity but with $V_k(i,i) = \cos(\phi)$, $V_k(j,j) = \cos(\phi)$, $V_k(i,j) = \sin(\phi)$, and $V_k(j,i) = -1*\sin(\phi)$, which is a Givens rotation. 

- The angle $\phi$ is computed by the following formula
$$ 
\cot (2\phi) = \frac{A^{(k)}(i,i)-A^{(k)}(j,j)}{2*A^{(k)}(i,j)}, 
    \tan(\phi) = \sign(\cot (2\phi))/(|\cot (2\phi)| + \sqrt{1+(\cot (2\phi))^2}) \in (-\pi/4,\pi/4],
$$
where $\phi$ is the smaller of two angles satisfying the requirement.

- Orthogonality of the result products of $V_k$ is obvious, since each component is numerical orthogonal.

- Accumulation of right singular vectors: update V = V * J each rotation is sufficient. A new way will be after convergence, we have $A^{(\infty)} = U\Sigma = AV$, where $V$ is unknown. Then we are able to compute $V$ by solving the linear system $A^{(\infty)} = AV$.

- After convergence, normalize columns of A to obtain U

- For numerical considerations, such as zero columns, cancellation, tie-breaking for theta. See the file <./dgesvj.txt>.

- Convergence: we have a special stopping criterion: The algorithm will stop only if 
  $|A^{(k)}(:,i)^T A^{(k)}(:,j)| <= tol * \|A^{(k)}(:,i)\|_2 \|A^{(k)}(:,j)\|_2$
  where tol is a user defined or predefined as unit roundoff of current working precision multiplied by $\sqrt{m*n}$.

---

## Notation and Conventions

Define the symbols, indices, and naming conventions used throughout the theory and code.

- TODO: Matrix naming — A (input m×n, m ≥ n), U (left singular vectors), Σ (singular values), V (right singular vectors)
- TODO: Index conventions — columns of A as a_j, a_i; use 1-based or 0-based indexing consistently
- TODO: Norm definitions — Euclidean (L2) norm of a vector, Frobenius norm of a matrix, off-diagonal Frobenius norm (off(A))
- TODO: Sweep counter — k, iteration index within a sweep
- TODO: Convergence threshold — ε (epsilon), tolerance for declaring a sweep converged
- TODO: Rotation parameters — c = cos(θ), s = sin(θ), t = tan(θ)

---

## Pseudo-Code / Procedure Outline

Structured, high-level outline of the algorithm. Actual implementation lives in `src/one_sided_jacobi/`.

See <./dgesvj.txt>.

---

## Convergence

Discussion of convergence properties and stopping criteria.

See <./dgesvj.txt>.

---

## Performance Considerations

Practical aspects for efficient implementation, especially in a Python + Cython context.

- Preconditioning — column scaling, QR pre-processing to reduce initial off-diagonal norm
- Parallel sweep ordering — row-cyclic vs. round-robin, and how ordering affects cache locality and parallelism
- Memory layout — column-major (Fortran order) vs. row-major (C order); implications for Cython inner loops
- Incremental norm updates — avoid full O(n²) norm recomputation after each rotation (update formulas)
- BLAS/LAPACK integration — whether to call optimized kernels or hand-write the inner loop in Cython
- Vectorization opportunities — SIMD within rotation application, batch rotations when columns are independent
- Numerical stability — choice between computing angle via atan2, directly from inner product, or via the Brent-Luk variant
- Handling tall-and-skinny matrices (m ≫ n) — pre-multiply with QR to work on the square R factor

---

## References

Key papers, books, and resources to consult during implementation.

- TODO: Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. — Chapters on Jacobi methods and SVD
- TODO: Demmel, J., & Veselić, K. (1992). Jacobi's method is more accurate than QR. *SIAM Journal on Matrix Analysis and Applications*, 13(4), 1204–1245.
- TODO: Brent, R. P., Luk, F. T., & Van Loan, C. F. (1985). Computation of the singular value decomposition using mesh-connected processors. *Journal of VLSI and Computer Systems*, 1(3), 242–270.
- TODO: Van der Vorst, H. A., & Golub, G. H. (1997). 150 years old and still alive: Eigenproblems. In *The State of the Art in Numerical Analysis*, Oxford University Press.
- TODO: Additional references on parallel Jacobi methods, convergence proofs, and Cython/numerical computing best practices

---

## Implementation Mapping

Where each theoretical component lives in the codebase.

| Component | Location |
|-----------|----------|
| Python package root | `src/one_sided_jacobi/` |
| Package init (`__init__.py`) | `src/one_sided_jacobi/__init__.py` |
| Cython extension (inner-loop optimizations) | `src/one_sided_jacobi/_core.pyx` |
| Cython declaration file (type declarations) | `src/one_sided_jacobi/_core.pxd` |
| Build system integration | `src/one_sided_jacobi/_cybuild.py` |
| Build configuration | `pyproject.toml` (setuptools + Cython) |
| Pure Python utilities (to add) | Additional `.py` modules under `src/one_sided_jacobi/` |

The Cython extension (`_core.pyx`) is the performance-critical inner loop: column-pair updates, norm recomputation, and rotation application. Declarations in `_core.pxd` expose typed C functions for use by other Cython modules. Build orchestration (`_cybuild.py`) hooks setuptools to compile Cython extensions transparently during `pip install`.
