# LinAlg-DSL

Rust Tensor Engine from Scratch
A minimal, from-scratch tensor and linear algebra engine in Rust, built to deeply understand how modern ML frameworks work under the hood. This project focuses on:

Hand-written tensor kernels (no ndarray, no BLAS).

LU-based linear solves and triangular systems.

A small AST layer to represent and evaluate tensor expressions.

It’s designed as a learning and portfolio project, not a production library.

Project Goals:
Implement a Tensor type with explicit rows, cols, and manual indexing.

Write all core kernels by hand:

Elementwise ops: add, sub, scale, ReLU.

Matrix–vector and matrix–matrix routines.

Triangular solvers and permutation application.

Build a simple AST (Abstract Syntax Tree) to represent tensor expressions and evaluate them using your kernels.

Use this as a stepping stone toward:

Custom ML layers and small models.

Deeper work in compilers, numerical methods, and systems-level ML.

Project Structure
Rough module layout (names may vary as you refactor):

text
src/
  main.rs        # small demos / manual tests
  tensor.rs      # Tensor struct and all numerical kernels
  ast.rs         # AST definitions and evaluation over tensors
  lib.rs         # (optional) re-exports and module wiring
  tensor.rs      # This file implements the core numeric engine

Tensor type:

Stores data as a flat buffer (e.g., Vec<f32>) plus rows, cols.

Indexing helpers: at(i, j), set(i, j, val).

Elementwise kernels:

add(a, b) -> Tensor – elementwise A + B.

sub(a, b) -> Tensor – elementwise A - B.

scale(c, x) -> Tensor – scalar–tensor multiply c * X.

relu(a) -> Tensor – applies max(0,x) elementwise.

Linear-algebra kernels:

Diagonal mat–vec: diag_matvec(d, x).

Forward substitution: solve_lower(L, b) for L y = b.

Backward substitution: solve_upper(U, y) for U x = y.

Permutation matrix–vector: matvec_perm(P, x) for P x.

Safety checks:

Assertions on shape compatibility (matching dimensions, square matrices where needed).

Assertions to validate permutation matrices (exactly one 1 per row).

These functions are intentionally low-level and explicit to make the math and indexing transparent.