# LibRI

Read this in other languages: [English](./README.md), [中文](./README.cn.md)


## RI technique

Conventional Density functional theory (DFT) methods are not sufficient to properly describe certain types of binding interactions.
It is often necessary to use higher-level methods like hybrid functionals, random-phase approximation (RPA), GW methods and beyond to accurately treat.
However, the computational cost of these methods are prohibitively expensive.

For local bases, invoking the resolution-of-the-identity (RI) significantly reduces the storage requirement and the computational cost.


## requirements for LibRI

LibRI is a C++ library using RI technique for methods beyond DFT. It is a head-only library.

These are necessary for LibRI:
- C++ compiler, supported C++11 and OpenMP.
- MPI library, used for inter-process communication.
- BLAS and LAPACK libraries, used as backends for Tensor operations.
  > If Math Kernel Library (MKL) is linked as the BLAS and LAPACK backend, it's recommended to define macro `__MKL_RI` before including any LibRI's header file. Some LibRI's functions will be silently substituted with calls to MKL routines.
- [cereal](https://uscilab.github.io/cereal/) library, a head-only library for serialization.
- [LibComm](https://github.com/abacusmodeling/LibComm.git) library, a head-only library for inter-process communication.

## Local interaction integrals

`RI/physics/LocalInteraction.h` provides occupation-independent local
four-index integrals. It accepts a supplied interaction metric; it does not
choose a Coulomb kernel, an HSE hybrid fraction, occupied states, local
projectors or a double-counting model.

```cpp
#include <RI/physics/LocalInteraction.h>
namespace LI = RI::Local_Interaction;

// Raw first-end LRI Cs[I][{J,R}](mu,i,j), shape {naux_I,nao_I,nao_J}.
// orbitals[I][A](i,a), shape {nao_I,M}, may be complex.
// metric[I][{J,L}](mu,nu) = V_{I0,JL}, shape {naux_I,naux_J}.
const auto pairs = LI::build_lri_pair_vertices(Cs, orbitals, M);
const auto integrals = LI::contract_real_space_pair_vertices(pairs, metric, M);
const auto uj = LI::extract_direct_exchange(integrals);
// integrals(a,b,c,d) = integral phi_a* phi_b v phi_c* phi_d.
// uj.direct(a,b) = I(a,a,b,b); uj.exchange(a,b) = I(a,b,b,a).
```

The orbital map is
`std::map<TA,std::map<std::array<Tcell,Ndim>,RI::Tensor<T>>>`.
Cells are explicit physical translations, with no modulo folding or image
replication. The pair builder includes both LRI endpoint contributions and
requires **raw first-end** coefficients, not an already completed AO product.
Missing orbital cells, Cs blocks and directed metric blocks mean zero.
Both directed metric blocks must therefore be supplied when both are nonzero.
The host must provide all blocks needed by the selected finite orbital support
and verify its projector normalization, basis identity, units and convergence.
The library does not infer Hermitian completion or validate completeness of
sparse maps. MPI distribution and reduction remain with the host.

For already formed vertices, `contract_pair_vertices(left,V,right,nl,nr)`
uses matrices of shape `{nl*nl,naux_left}` and `{nr*nr,naux_right}`, with
ordered pair index `a*n+b`, and returns `{nl,nl,nr,nr}`. This is the **bilinear**
contraction `left * V * transpose(right)`, without complex conjugation.
Real/complex inputs are promoted to their common scalar type; U/J extraction
retains that type rather than discarding imaginary parts or applying averages.

For Fourier vertices with real auxiliary functions,
`contract_ordered_pair_vertices(Qleft,Vq,Qright,nl,nr)` instead evaluates
`conj(Qleft(ba,mu;q)) * Vq(mu,nu) * Qright(cd,nu;q)`.
This equals `Qleft(ab;-q) * Vq * Qright(cd;q)` under the corresponding pair
conjugation identity. The host applies the complete q sum and its weights;
no Gamma-only substitution or inverse-N factor is inserted by the library.

The standalone control in
`unittests/physics/local_interaction_standalone.cpp` can be compiled with
C++11, the `include` directory and the platform's BLAS link flags. It compares
against an explicit small AO four-index tensor, complex orthonormal local
orbitals, direct two-cell quadrature and a normalized full-q sum. It also
checks complex pair reversal, rectangular mixed-type contractions and invalid
input rejection. It does not require MPI initialization.
