// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#![warn(missing_docs)]

//! # tetra3
//!
//! `tetra3` is a fast, highly-optimized Rust implementation of the lost-in-space
//! star identification and celestial plate solving algorithms derived from `cedar-solve` and `esa/tetra3`.
//!
//! ## Overview
//!
//! The crate provides two primary operational components:
//! 1. **Centroid Extraction**: Locates candidate star centroids in optical sensor images.
//!    - [`extractor`]: The standard extraction pipeline matching `cedar-solve` and `esa/tetra3`,
//!      supporting local/global median/mean background subtraction and moments calculation.
//!    - [`fast_extractor`]: A zero-allocation, pre-allocated pipeline tailored for embedded
//!      microprocessors, featuring SIMD/integer optimizations and custom background modes
//!      ([`FastBgSubMode::BlockMedian`],
//!      [`FastBgSubMode::LineMedian`]).
//! 2. **Plate Solving**: Matches 2D image centroids against an indexed star database.
//!    - [`solver`]: Identifies 4-star asterisms using geometric hash tables and verifies
//!      candidate attitudes via k-d tree spatial indexing, with optional horizon rejection.
//!
//! The top-level [`Tetra3`] struct ties extraction and plate-solving together with lazy
//! database initialization.
//!
//! ## Features
//!
//! - `python`: Enables PyO3/NumPy integration and dictionary converters for Python bindings.
//! - `force-32bit-solver`: Compiles the solver using single-precision (`f32`) arithmetic instead of `f64`,
//!   optimized for 32-bit embedded platforms (e.g. ARMv6/ARMv7).

/// Star centroid extraction algorithms (standard pipeline matching cedar-solve).
pub mod extractor;
/// High-performance, zero-allocation star centroid extraction with custom background subtraction modes.
pub mod fast_extractor;
/// Core star catalog pattern matching and lost-in-space plate solver.
pub mod solver;
/// Unified facade struct integrating star extraction and plate solving.
pub mod tetra3;

#[cfg(feature = "python")]
/// Python/PyO3 interoperability helpers.
pub mod python;

pub use crate::extractor::*;
pub use crate::fast_extractor::*;
pub use crate::solver::*;
pub use crate::tetra3::*;
