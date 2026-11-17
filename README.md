# Exploring Tensor Decomposition Strategies for Entanglement Analysis in Quantum Systems

Exam project for the course *Quantum Information and Computing* (University of Padova, A.Y. 2023/2024), by Filippo Pra Floriani and Alessio Saccomani.

This project develops a numerical scheme to detect **how entanglement is distributed among the legs of a rank-3 tensor**, a basic building block of tensor networks. It was the preliminary study that later grew into my master's thesis on tripartite entanglement decompositions for tensor networks.

## The problem

Given a rank-3 tensor $T$ with legs $(d_1, d_2, d_3)$, i.e. a state of three subsystems, we want to tell whether the subsystems are

- completely **separable**
- **biseparable**, with entanglement only between one pair of legs (and which pair)
- **genuinely tripartite** entangled

## Method

1. **Build test tensors with known structure.** $T$ is obtained by contracting a random three-site MPS. The bond dimensions decide where entanglement is placed, so every test case has a known answer.
2. **Sample Haar-random unitaries.** A numerically stable sampler based on the QR decomposition of a complex Gaussian matrix, with the phase correction on the diagonal of $R$. Its correctness is checked against the expected eigenvalue density (uniform on the unit circle) and the Wigner-surmise spacing distribution.
3. **Contract and split.** A random unitary $U$ is contracted with $T$ on leg $d_1$, and $d_1$ is split into two factors $(d_4, d_5)$ with $d_1 = d_4 \cdot d_5$, giving a rank-4 tensor $T'$.
4. **SVD and entanglement entropy.** $T'$ is decomposed via SVD along two different bipartitions (vertical and horizontal), and the von Neumann entropy $S = -\sum_k \lambda_k^2 \log \lambda_k^2$ is computed from the singular values (Schmidt coefficients).
5. **Classify.** The pattern of the entropies identifies the entanglement structure.

## Results

- Separable and biseparable tensors are correctly classified, including which pair of legs is entangled.
- Genuine tripartite entanglement is detected, since no splitting gives zero entropy. In this case the entropy profile over all splittings gives a qualitative indication of which pair of legs is more strongly entangled.
- The Haar sampler reproduces the expected eigenvalue and spacing statistics.

<!-- Suggested: add one or two figures here, e.g. the eigenvalue spacing histogram vs Wigner surmise and an entropy-vs-splitting plot, saved in a figures/ folder -->

## Repository structure

| File | Content |
|---|---|
| `tn_functions.py` | All functions (Haar sampling, MPS construction, contraction, splitting, SVD, entropy, classification) |
| `tn.ipynb` | Tensor decomposition and entanglement classification. Change the physical and bond dimensions to test different configurations |
| `haar_measure.ipynb` | Generation and validation of Haar-random unitaries |
| `PraFloriani_Saccomani_project.pdf` | Project presentation with the full derivation |

## How to run

```bash
pip install numpy scipy matplotlib tensornetwork
jupyter notebook tn.ipynb
```

Built with Python, NumPy and Google's [TensorNetwork](https://github.com/google/TensorNetwork) library.
