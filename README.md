# Rust Tensor Engine from Scratch

A minimal, from-scratch tensor and linear algebra engine in Rust, built to deeply understand how modern machine learning frameworks work under the hood.

This project intentionally avoids external numerical libraries (`ndarray`, BLAS, etc.) and instead implements all tensor operations and linear algebra kernels manually.

It is designed as a **learning and portfolio project**, not a production library.

---

## Motivation

Modern ML frameworks like PyTorch, TensorFlow, and JAX rely on highly optimized tensor engines and linear algebra kernels. To truly understand how these systems work, it is essential to build these components from first principles.

This project focuses on:

- Writing tensor kernels completely by hand  
- Implementing core linear algebra routines from scratch  
- Representing tensor computations using a small AST layer  
- Building a foundation for deeper work in:
  - ML systems engineering  
  - compiler design  
  - numerical computing  
  - high-performance computing  

---

## Features

### Core Tensor Engine

- Custom `Tensor` type backed by a flat memory buffer (`Vec<f64>` or generic equivalent)
- Explicit row and column tracking
- Manual indexing with helper functions

### Hand-Written Numerical Kernels

#### Elementwise Operations

- `add(a, b)` → elementwise tensor addition
- `sub(a, b)` → elementwise tensor subtraction
- `scale(c, x)` → scalar-tensor multiplication
- `relu(a)` → elementwise ReLU activation

#### Linear Algebra Operations

- Matrix–vector multiplication
- Matrix–matrix multiplication
- Diagonal matrix–vector multiplication (`diag_matvec`)
- Forward substitution (`solve_lower`)
- Backward substitution (`solve_upper`)
- Permutation matrix application (`matvec_perm`)

All algorithms are implemented manually without BLAS.

---

## AST-Based Expression Evaluation

Includes a small Abstract Syntax Tree (AST) layer that allows:

- Representation of tensor expressions as computation graphs
- Evaluation of expressions using the custom tensor kernels
- Separation of computation representation and execution

This mirrors the architecture used in real ML frameworks.

---

## Project Goals

The primary goals of this project are:

- Implement a `Tensor` type with:
  - explicit rows and columns
  - flat buffer storage
  - manual indexing

- Write all numerical kernels from scratch:
  - Elementwise operations
  - Matrix operations
  - Triangular solvers
  - Permutation application

- Build an AST layer for tensor expression evaluation

- Use this engine as a foundation for:
  - custom ML layers
  - small neural networks
  - compiler experiments
  - ML runtime experimentation

---

## Project Structure

```
src/
├── main.rs      # demos, manual tests, experimentation
├── tensor.rs    # core Tensor type and numerical kernels
├── ast.rs       # AST representation and evaluation logic
├── lib.rs       # optional module exports and wiring
```

---

## Tensor Implementation Details

### Tensor Representation

The `Tensor` struct stores:

- Flat contiguous buffer (`Vec<f64>`)
- Number of rows
- Number of columns

Example conceptual layout:

```rust
struct Tensor {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}
```

### Indexing Helpers

Functions provide safe and explicit access:

- `at(i, j)` → read element
- `set(i, j, value)` → write element

Manual indexing ensures full control over memory layout and access patterns.

---

## Implemented Kernels

### Elementwise Kernels

```rust
add(a, b)     -> Tensor
sub(a, b)     -> Tensor
scale(c, x)   -> Tensor
relu(a)       -> Tensor
```

These operate directly on the flat buffer with explicit loops.

---

### Linear Algebra Kernels

#### Diagonal Matrix–Vector Multiply

```rust
diag_matvec(d, x)
```

Computes:

```
y = D x
```

where `D` is diagonal.

---

#### Forward Substitution

```rust
solve_lower(L, b)
```

Solves:

```
L y = b
```

where `L` is lower triangular.

---

#### Backward Substitution

```rust
solve_upper(U, y)
```

Solves:

```
U x = y
```

where `U` is upper triangular.

---

#### Permutation Application

```rust
matvec_perm(P, x)
```

Computes:

```
y = P x
```

where `P` is a permutation matrix.

---

## Safety and Validation

The implementation includes runtime checks for correctness:

- Shape compatibility assertions
- Square matrix validation where required
- Permutation matrix validation:
  - Exactly one `1` per row
  - All other entries `0`

These checks ensure correctness while keeping implementation transparent.

---

## Why This Matters

This project builds foundational knowledge required for:

- ML framework development
- Compiler work (LLVM, MLIR, XLA, etc.)
- GPU kernel development
- Numerical libraries
- Scientific computing systems

It demonstrates understanding of:

- memory layout
- numerical algorithms
- linear algebra implementation
- systems-level performance considerations

---

## Future Extensions

Planned improvements include:

- Automatic differentiation (autograd)
- Computation graph optimization
- SIMD optimizations
- GPU backend experimentation
- JIT compilation experiments
- Small neural network implementations

---

## Example Usage

```rust
let a = Tensor::new(...);
let b = Tensor::new(...);

let c = add(&a, &b);
let d = relu(&c);
```

---

