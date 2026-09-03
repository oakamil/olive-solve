// Copyright (c) 2026 Omair Kamil
//
// This file is a derivative work, inspired from `tetra3.py` of the cedar-solve and
// esa/tetra3 projects. This file has major optimizations of algorithms in those
// works with additional original computational logic.
//
// This derivative work is licensed under the Apache License, Version 2.0 (the
// "License"). You may not use this file except in compliance with the License.
// A copy of the License is located in the LICENSE.md file in the root of this
// repository.
//
//
// Cedar Solve license:
//    Copyright 2023 Steven Rosenthal smr@dt3.org
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        https://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
//
// tetra3 license:
//    Copyright 2019 the European Space Agency
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        https://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
//
// Original Tetra license notice:
//    Copyright (c) 2016 brownj4
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.

//! Star pattern plate solver and catalog identification engine.
//!
//! Implements geometric hash lookup of 4-star asterisms against indexed databases
//! (supporting both `tetra3` and `cedar-solve` database formats). Includes spatial k-d tree
//! verification, attitude determination, lens distortion estimation, and horizon-based filtering.

use ndarray::Array2;

// --- Data Structures & Options ---

/// Indicates the status of a plate solve attempt.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SolveStatus {
    /// A high-confidence match was found that satisfies `options.match_threshold`.
    MatchFound,
    #[default]
    /// No match was found across the catalog search space.
    NoMatch,
    /// The solve attempt timed out before evaluating all candidate patterns.
    Timeout,
    /// The solve attempt was cancelled via an external signal.
    Cancelled,
    /// Too few stars were provided to form a pattern (< 4 centroids).
    TooFew,
    /// A candidate match was identified, but its false-positive probability (`prob`)
    /// exceeded `options.match_threshold`. Returned only when `return_best_failed_match` is true.
    LowConfidenceMatch,
}

/// Configuration options for the plate solving process.
#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Estimated field of view in degrees (horizontal width). If `None`, searches entire database FOV range.
    pub fov_estimate: Option<f64>,
    /// Maximum allowable deviation from `fov_estimate` in degrees.
    pub fov_max_error: Option<f64>,
    /// Radius (as a fraction of FOV) within which candidate catalog stars must match image centroids.
    pub match_radius: f64,
    /// Maximum false-positive probability threshold required to accept a match as [`SolveStatus::MatchFound`].
    pub match_threshold: f64,
    /// Timeout in milliseconds after which the solve search will abort.
    pub solve_timeout_ms: Option<f64>,
    /// Optional known radial lens distortion parameter.
    pub distortion: Option<f64>,
    /// Maximum allowable angular error when matching asterism edge ratios.
    pub match_max_error: f64,
    /// If true, populates matched centroid, star, and catalog ID arrays in [`Solution`].
    pub return_matches: bool,
    /// If true, populates `catalog_stars` with all catalog stars in the camera field of view.
    pub return_catalog: bool,
    /// If true, calculates and returns the 3x3 camera-to-celestial rotation matrix in [`Solution`].
    pub return_rotation_matrix: bool,
    /// Optional pixel coordinates `(y, x)` to calculate celestial RA/Dec coordinates for.
    pub target_pixel: Option<Array2<f64>>,
    /// Optional celestial coordinates `(RA, Dec)` to calculate camera pixel coordinates for.
    pub target_sky_coord: Option<Array2<f64>>,
    /// If true, allows target pixels located outside the image sensor boundary.
    pub allow_out_of_bounds_target_pixel: Option<bool>,

    /// Optional: The observer's latitude in degrees. If provided alongside `observer_lst`,
    /// the solver will instantly reject patterns containing stars physically below the horizon.
    pub observer_latitude: Option<f64>,

    /// Optional: The observer's Local Sidereal Time in degrees. Required for horizon-based early rejection.
    pub observer_lst: Option<f64>,

    /// Optional: The minimum allowed altitude (in degrees) for the camera's center pixel (boresight).
    /// 0.0° represents the physical horizon. This value can be negative to allow the camera's center
    /// to point slightly downwards into the ground, which is necessary if the telescope (target pixel)
    /// is mounted at an offset and is still pointing at the sky. Used to reject mathematically valid
    /// false-positives that imply an impossibly low camera orientation.
    pub min_boresight_altitude: Option<f64>,

    /// Optional: If true, when no candidate pattern meets the strict `match_threshold`,
    /// the solver will return the globally best candidate evaluated during the search
    /// with status `SolveStatus::LowConfidenceMatch` and its calculated `prob`.
    /// This prevents premature termination on weak candidates (which occurs with relaxed thresholds)
    /// while still recovering a solution in obstructed/cluttered star fields.
    pub return_best_failed_match: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        SolveOptions {
            fov_estimate: None,
            fov_max_error: None,
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(5000.0),
            distortion: None,
            match_max_error: 0.002,
            return_matches: false,
            return_catalog: false,
            return_rotation_matrix: false,
            target_pixel: None,
            target_sky_coord: None,
            allow_out_of_bounds_target_pixel: None,
            observer_latitude: None,
            observer_lst: None,
            min_boresight_altitude: None,
            return_best_failed_match: false,
        }
    }
}

/// Contains the result of a plate solve attempt.
#[derive(Debug, Default, Clone)]
pub struct Solution {
    /// Right ascension of the center in degrees (J2000 equinox).
    pub ra: Option<f64>,
    /// Declination of the center in degrees (J2000 equinox).
    pub dec: Option<f64>,
    /// Celestial roll / position angle of the camera in degrees.
    pub roll: Option<f64>,
    /// Solved horizontal field of view in degrees.
    pub fov: Option<f64>,
    /// Estimated lens radial distortion coefficient.
    pub distortion: Option<f64>,
    /// Root-mean-square error of matched stars in pixels.
    pub rmse: Option<f64>,
    /// 90th-percentile error of matched stars in pixels.
    pub p90e: Option<f64>,
    /// Maximum error among all matched stars in pixels.
    pub maxe: Option<f64>,
    /// Total count of catalog stars matched to image centroids.
    pub matches: Option<usize>,
    /// Estimated false-positive probability across the entire catalog database (`prob_mismatch * num_patterns`).
    /// Lower is better. A value `< match_threshold` indicates a high-confidence match (`MatchFound`),
    /// while a value `>= match_threshold` is returned on `LowConfidenceMatch`.
    pub prob: Option<f64>,
    /// Equinox epoch of the star coordinates.
    pub epoch_equinox: Option<f64>,
    /// Proper motion epoch applied to catalog positions.
    pub epoch_proper_motion: Option<f64>,
    /// Status outcome of the plate solve operation.
    pub status: SolveStatus,
    /// Total solve execution time in milliseconds.
    pub t_solve_ms: f64,
    /// True if the field was solved under mirrored (horizontally flipped) optical parity.
    pub is_mirrored: bool,
    /// 3x3 rotation matrix mapping camera coordinate frame to celestial J2000 coordinate frame.
    pub rotation_matrix: Option<Array2<f64>>,
    /// Celestial Right Ascension (degrees) corresponding to each requested pixel in `target_pixel`.
    pub target_ra: Option<Vec<f64>>,
    /// Celestial Declination (degrees) corresponding to each requested pixel in `target_pixel`.
    pub target_dec: Option<Vec<f64>>,
    /// Image Y pixel coordinate corresponding to each requested celestial target in `target_sky_coord`.
    pub target_y: Option<Vec<Option<f64>>>,
    /// Image X pixel coordinate corresponding to each requested celestial target in `target_sky_coord`.
    pub target_x: Option<Vec<Option<f64>>>,
    /// Pixel coordinates `[y, x]` of detected centroids that matched catalog stars.
    pub matched_centroids: Option<Vec<[f64; 2]>>,
    /// Celestial coordinates and magnitude `[ra, dec, mag]` of matched catalog stars.
    pub matched_stars: Option<Vec<[f64; 3]>>,
    /// Catalog identifiers for matched stars.
    pub matched_cat_id: Option<Vec<Vec<u32>>>,
    /// All catalog stars located within the camera field of view as `(ra, dec, mag, y, x)`.
    pub catalog_stars: Option<Vec<(f64, f64, f64, f64, f64)>>,
}

/// Representation of a star loaded from the indexed catalog database.
#[derive(Clone, Copy)]
pub struct StarMetadata {
    /// Right ascension in degrees.
    pub ra: f64,
    /// Declination in degrees.
    pub dec: f64,
    /// Apparent visual magnitude.
    pub mag: f64,
}

/// 32-bit single-precision (`f32`) solver implementation.
pub mod solver_32 {
    /// Internal floating-point precision type (`f32`).
    pub type Flt = f32;
    include!("solver_core.rs");
}

/// 64-bit double-precision (`f64`) solver implementation.
pub mod solver_64 {
    /// Internal floating-point precision type (`f64`).
    pub type Flt = f64;
    include!("solver_core.rs");
}

// Re-export active solver based on build configuration
#[cfg(not(feature = "force-32bit-solver"))]
pub use solver_64::{Scratchpads, Solver};

#[cfg(feature = "force-32bit-solver")]
pub use solver_32::{Scratchpads, Solver};
