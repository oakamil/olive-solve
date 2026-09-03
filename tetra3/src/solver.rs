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

use ndarray::Array2;

// --- Data Structures & Options ---

#[derive(Debug, Clone, PartialEq, Default)]
#[allow(missing_docs)]
/// Indicates the status of a plate solve attempt.
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

#[derive(Debug, Clone)]
#[allow(missing_docs)]
/// Configuration options for the plate solving process.
pub struct SolveOptions {
    pub fov_estimate: Option<f64>,
    pub fov_max_error: Option<f64>,
    pub match_radius: f64,
    pub match_threshold: f64,
    pub solve_timeout_ms: Option<f64>,
    pub distortion: Option<f64>,
    pub match_max_error: f64,
    pub return_matches: bool,
    pub return_catalog: bool,
    pub return_rotation_matrix: bool,
    pub target_pixel: Option<Array2<f64>>,     // N x 2 (y, x)
    pub target_sky_coord: Option<Array2<f64>>, // N x 2 (ra, dec)
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

#[derive(Debug, Default, Clone)]
#[allow(missing_docs)]
/// Contains the result of a plate solve attempt.
pub struct Solution {
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub roll: Option<f64>,
    pub fov: Option<f64>,
    pub distortion: Option<f64>,
    pub rmse: Option<f64>,
    pub p90e: Option<f64>,
    pub maxe: Option<f64>,
    pub matches: Option<usize>,
    /// Estimated false-positive probability across the entire catalog database (`prob_mismatch * num_patterns`).
    /// Lower is better. A value `< match_threshold` indicates a high-confidence match (`MatchFound`),
    /// while a value `>= match_threshold` is returned on `LowConfidenceMatch`.
    pub prob: Option<f64>,
    pub epoch_equinox: Option<f64>,
    pub epoch_proper_motion: Option<f64>,
    pub status: SolveStatus,
    pub t_solve_ms: f64,
    pub is_mirrored: bool,
    pub rotation_matrix: Option<Array2<f64>>,
    pub target_ra: Option<Vec<f64>>,
    pub target_dec: Option<Vec<f64>>,
    pub target_y: Option<Vec<Option<f64>>>,
    pub target_x: Option<Vec<Option<f64>>>,
    pub matched_centroids: Option<Vec<[f64; 2]>>,
    pub matched_stars: Option<Vec<[f64; 3]>>, // ra, dec, mag
    pub matched_cat_id: Option<Vec<Vec<u32>>>,
    pub catalog_stars: Option<Vec<(f64, f64, f64, f64, f64)>>, // ra, dec, mag, y, x
}

#[derive(Clone, Copy)]
#[allow(missing_docs)]
/// Internal representation of a star loaded from the cedar-solve database.
pub struct StarMetadata {
    pub ra: f64,
    pub dec: f64,
    pub mag: f64,
}

pub mod solver_32 {
    pub type Flt = f32;
    include!("solver_core.rs");
}

pub mod solver_64 {
    pub type Flt = f64;
    include!("solver_core.rs");
}

// Re-export active solver based on build configuration
#[cfg(not(feature = "force-32bit-solver"))]
pub use solver_64::{Scratchpads, Solver};

#[cfg(feature = "force-32bit-solver")]
pub use solver_32::{Scratchpads, Solver};
