// Copyright (c) 2026 Omair Kamil
//
// This file is a derivative work - a Python interface to the optimized Rust port
// of the cedar-solve and esa/tetra3 projects. The original underlying code is
// licensed under the Apache License, Version 2.0.
// Original Copyright (c) European Space Agency, Steven Rosenthal, brownj4, and
// contributors.
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

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SVD};
use ndarray::Array2;
use npyz::NpyFile;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zip::ZipArchive;

const MAGIC_RAND: u64 = 2654435761;

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

// Removed ProbeEntry
// --- High-Performance Native Math Helpers ---

#[inline(always)]
fn angle_from_distance(dist: f64) -> f64 {
    2.0 * (0.5 * dist).asin()
}

#[inline(always)]
fn distance_from_angle(angle: f64) -> f64 {
    2.0 * (angle / 2.0).sin()
}

// OPTIMIZATION: Replaces incredibly heavy `statrs::Binomial` cdf calculation inside hot loops.
// Nanosecond execution time for small N probabilities, exactly mirroring scipy.stats.binom.cdf.
fn fast_binomial_cdf(k: i64, n: u64, p: f64) -> f64 {
    if k < 0 {
        return 0.0;
    }
    if k >= n as i64 {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let mut cdf = 0.0;
    let mut term = (1.0 - p).powi(n as i32);
    cdf += term;
    for i in 1..=(k as u64) {
        term = term * (n - i + 1) as f64 / i as f64 * p / (1.0 - p);
        cdf += term;
    }
    cdf
}

fn compute_vectors_flat(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    fov: f64,
) -> Vec<[f64; 3]> {
    let scale_factor = (fov / 2.0).tan() / width * 2.0;
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;
    let mut out = Vec::with_capacity(centroids.len());

    for c in centroids {
        let v0 = 1.0;
        let v1 = (img_center_x - c[1]) * scale_factor;
        let v2 = (img_center_y - c[0]) * scale_factor;
        let inv_norm = 1.0 / (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        out.push([v0 * inv_norm, v1 * inv_norm, v2 * inv_norm]);
    }
    out
}

// OPTIMIZATION: Zero-allocation inner loop alternative for compute_vectors
fn compute_vectors_inplace(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    fov: f64,
    out: &mut Vec<[f64; 3]>,
    len: usize,
) {
    let scale_factor = (fov / 2.0).tan() / width * 2.0;
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    out.clear();
    for i in 0..len {
        let v0 = 1.0;
        let v1 = (img_center_x - centroids[i][1]) * scale_factor;
        let v2 = (img_center_y - centroids[i][0]) * scale_factor;
        let inv_norm = 1.0 / (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        out.push([v0 * inv_norm, v1 * inv_norm, v2 * inv_norm]);
    }
}

// OPTIMIZATION: Zero-allocation inner loop alternative for compute_centroids
fn compute_centroids_inplace(
    vectors: &[[f64; 3]],
    height: f64,
    width: f64,
    fov: f64,
    out_centroids: &mut [[f64; 2]],
    out_kept: &mut Vec<usize>,
    len: usize,
    filter_bounds: bool,
) {
    out_kept.clear();
    let scale_factor = -width / 2.0 / (fov / 2.0).tan();
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    for i in 0..len {
        let v0 = vectors[i][0];
        if v0 <= 0.0 {
            // Point is behind the focal plane
            continue;
        }
        let inv_v0 = 1.0 / v0;
        let cy = scale_factor * (vectors[i][2] * inv_v0) + img_center_y;
        let cx = scale_factor * (vectors[i][1] * inv_v0) + img_center_x;
        out_centroids[i][0] = cy;
        out_centroids[i][1] = cx;

        if !filter_bounds || (cy > 0.0 && cx > 0.0 && cy < height && cx < width) {
            out_kept.push(i);
        }
    }
}

fn undistort_centroids(centroids: &[[f64; 2]], height: f64, width: f64, k: f64) -> Vec<[f64; 2]> {
    let mut undistorted = Vec::with_capacity(centroids.len());
    undistort_centroids_inplace(centroids, height, width, k, &mut undistorted);
    undistorted
}

fn undistort_centroids_inplace(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    k: f64,
    out: &mut Vec<[f64; 2]>,
) {
    out.clear();
    out.extend_from_slice(centroids);
    let kp = k * (2.0 / width).powi(2);
    let inv_1_k = 1.0 / (1.0 - k);
    let h_2 = height / 2.0;
    let w_2 = width / 2.0;

    for row in out.iter_mut() {
        let dy = row[0] - h_2;
        let dx = row[1] - w_2;
        let r_dist_sq = dy * dy + dx * dx;
        let scale = (1.0 - kp * r_dist_sq) * inv_1_k;
        row[0] = (dy * scale) + h_2;
        row[1] = (dx * scale) + w_2;
    }
}

fn distort_centroids(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    k: f64,
    tol: f64,
    maxiter: usize,
) -> Vec<[f64; 2]> {
    let kp = k * (2.0 / width).powi(2);
    let mut distorted = centroids.to_vec();
    let inv_1_k = 1.0 / (1.0 - k);
    let h_2 = height / 2.0;
    let w_2 = width / 2.0;

    for row in distorted.iter_mut() {
        let dy = row[0] - h_2;
        let dx = row[1] - w_2;
        let r_undist = (dy * dy + dx * dx).sqrt();

        if r_undist < 1e-8 {
            continue;
        }

        let mut r_dist = r_undist;
        for _ in 0..maxiter {
            let r_undist_est = r_dist * (1.0 - kp * r_dist.powi(2)) * inv_1_k;
            let dru_drd = (1.0 - 2.0 * kp * r_dist) * inv_1_k;
            let error = r_undist - r_undist_est;
            r_dist += error / dru_drd;
            if error.abs() < tol {
                break;
            }
        }
        let scale = r_dist / r_undist;
        row[0] = (dy * scale) + h_2;
        row[1] = (dx * scale) + w_2;
    }
    distorted
}

// OPTIMIZATION: Zero-allocation inner loop alternative for sort_vectors_by_radius
fn sort_vectors_by_radius_inplace(
    vectors: &[[f64; 3]],
    sorted_out: &mut [[f64; 3]],
    radii_scratch: &mut Vec<(f64, usize)>,
    len: usize,
) {
    let mut centroid = [0.0, 0.0, 0.0];
    for i in 0..len {
        centroid[0] += vectors[i][0];
        centroid[1] += vectors[i][1];
        centroid[2] += vectors[i][2];
    }
    centroid[0] /= len as f64;
    centroid[1] /= len as f64;
    centroid[2] /= len as f64;

    radii_scratch.clear();
    for i in 0..len {
        let dx = vectors[i][0] - centroid[0];
        let dy = vectors[i][1] - centroid[1];
        let dz = vectors[i][2] - centroid[2];
        radii_scratch.push(((dx * dx + dy * dy + dz * dz), i));
    }

    radii_scratch.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (new_idx, &(_, old_idx)) in radii_scratch.iter().enumerate() {
        sorted_out[new_idx] = vectors[old_idx];
    }
}

// OPTIMIZATION: Fully unrolled, zero-allocation rotation helper.
fn rotate_vectors_inplace(
    rot: &Matrix3<f64>,
    vecs: &[[f64; 3]],
    transpose_rot: bool,
    out: &mut Vec<[f64; 3]>,
    len: usize,
) {
    let r = if transpose_rot { rot.transpose() } else { *rot };
    out.clear();
    for i in 0..len {
        let mut row = [0.0; 3];
        row[0] = r[(0, 0)] * vecs[i][0] + r[(0, 1)] * vecs[i][1] + r[(0, 2)] * vecs[i][2];
        row[1] = r[(1, 0)] * vecs[i][0] + r[(1, 1)] * vecs[i][1] + r[(1, 2)] * vecs[i][2];
        row[2] = r[(2, 0)] * vecs[i][0] + r[(2, 1)] * vecs[i][1] + r[(2, 2)] * vecs[i][2];
        out.push(row);
    }
}

// OPTIMIZATION: Pure-Rust stack-allocated SVD (via nalgebra). Replaces dynamic DMatrix matching.
fn find_rotation_matrix_and_det_inplace(
    image_vectors: &[[f64; 3]],
    catalog_vectors: &[[f64; 3]],
    len: usize,
) -> Option<(Matrix3<f64>, f64)> {
    let mut h = Matrix3::<f64>::zeros();
    // H = image_vectors.T * catalog_vectors
    for i in 0..len {
        for r in 0..3 {
            for c in 0..3 {
                h[(r, c)] += image_vectors[i][r] * catalog_vectors[i][c];
            }
        }
    }

    let svd = SVD::new(h, true, true);
    if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
        let rot = u * vt;
        Some((rot, rot.determinant()))
    } else {
        None
    }
}

// OPTIMIZATION: Zero-allocation inner loop alternative to matching logic
fn find_centroid_matches_inplace(
    image_centroids: &[[f64; 2]],
    img_len: usize,
    catalog_centroids: &[[f64; 2]],
    cat_len: usize,
    r: f64,
    out_matches: &mut Vec<(usize, usize)>,
    matches: &mut Vec<(usize, usize)>,
    matches1: &mut Vec<(usize, usize)>,
) {
    out_matches.clear();
    matches.clear();
    matches1.clear();
    let r_sq = r * r;

    // Step 1: Find all matches
    for i in 0..img_len {
        for j in 0..cat_len {
            let dy = image_centroids[i][0] - catalog_centroids[j][0];
            if dy.abs() >= r {
                continue;
            }
            let dx = image_centroids[i][1] - catalog_centroids[j][1];
            if dx.abs() >= r {
                continue;
            }
            if (dy * dy + dx * dx) < r_sq {
                matches.push((i, j));
            }
        }
    }

    // Step 2: unique_col1 = np.unique(matches[:, 1], return_index=True)
    matches.sort_by_key(|&(i, j)| (j, i));
    let mut last_j = usize::MAX;
    for &(i, j) in matches.iter() {
        if j != last_j {
            matches1.push((i, j));
            last_j = j;
        }
    }

    // Step 3: unique_col0 = np.unique(matches1[:, 0], return_index=True)
    matches1.sort_by_key(|&(i, j)| (i, j));
    let mut last_i = usize::MAX;
    for &(i, j) in matches1.iter() {
        if i != last_i {
            out_matches.push((i, j));
            last_i = i;
        }
    }
}

enum VerificationResult {
    Success(Solution),
    NewBestFailed(f64, Solution),
    None,
}

// Helper to build the solution
#[allow(clippy::too_many_arguments)]
fn verify_and_build_solution(
    scratch: &mut Scratchpads,
    star_kd_tree: &ImmutableKdTree<f64, 3>,
    star_vectors: &[[f64; 3]],
    star_metadata: &[StarMetadata],
    star_catalog_ids: &Option<Array2<u32>>,
    db_props: &HashMap<String, f64>,
    num_patterns: usize,
    rotation_matrix: &Matrix3<f64>,
    mut fov: f64,
    height: f64,
    width: f64,
    options: &SolveOptions,
    num_extracted_stars: usize,
    match_threshold: f64,
    best_prob_mismatch: f64,
    t0_solve: Instant,
) -> VerificationResult {
    // Find all catalog stars inside the FOV diagonal
    let fov_diagonal_rad = fov * ((width * width + height * height).sqrt() / width);
    let image_center_vector = [
        rotation_matrix[(0, 0)],
        rotation_matrix[(0, 1)],
        rotation_matrix[(0, 2)],
    ];

    // Epsilon to capture borders safely in f64
    let max_dist_sq = distance_from_angle(fov_diagonal_rad / 2.0).powi(2) + 1e-8;
    // OPTIMIZATION: Allocation-Free KD-Tree Parsing (Eliminated .collect() vector allocations)
    let mut nearby_nodes =
        star_kd_tree.within_unsorted::<SquaredEuclidean>(&image_center_vector, max_dist_sq);

    // Re-sort KDTree return list by index to prioritize brighter stars exactly like Python
    nearby_nodes.sort_unstable_by_key(|n| n.item);

    let num_nearby = nearby_nodes.len();
    if num_nearby == 0 {
        return VerificationResult::None;
    }

    let target_crop_len = 2 * num_extracted_stars;

    if scratch.sp_valid_cat_centroids.len() < target_crop_len {
        let new_size = target_crop_len.max(scratch.sp_valid_cat_centroids.len() * 2);
        scratch.sp_valid_cat_centroids.resize(new_size, [0.0; 2]);
        scratch.sp_valid_cat_vectors.resize(new_size, [0.0; 3]);
    }
    scratch.sp_valid_cat_inds.clear();

    let scale_factor_cent = -width / 2.0 / (fov / 2.0).tan();
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    let r = rotation_matrix;
    let mut crop_len = 0;

    for node in &nearby_nodes {
        let star_idx = node.item as usize;
        let vec = star_vectors[star_idx];

        let v0 = r[(0, 0)] * vec[0] as f64 + r[(0, 1)] * vec[1] as f64 + r[(0, 2)] * vec[2] as f64;
        let v1 = r[(1, 0)] * vec[0] as f64 + r[(1, 1)] * vec[1] as f64 + r[(1, 2)] * vec[2] as f64;
        let v2 = r[(2, 0)] * vec[0] as f64 + r[(2, 1)] * vec[1] as f64 + r[(2, 2)] * vec[2] as f64;

        let inv_v0 = 1.0 / v0;
        let cy = scale_factor_cent * (v2 * inv_v0) + img_center_y;
        let cx = scale_factor_cent * (v1 * inv_v0) + img_center_x;

        if cy > 0.0 && cx > 0.0 && cy < height && cx < width {
            scratch.sp_valid_cat_centroids[crop_len] = [cy, cx];
            scratch.sp_valid_cat_vectors[crop_len] = [vec[0] as f64, vec[1] as f64, vec[2] as f64];
            scratch.sp_valid_cat_inds.push(star_idx);
            crop_len += 1;
            if crop_len >= target_crop_len {
                break;
            }
        }
    }

    find_centroid_matches_inplace(
        &scratch.sp_image_centroids_undist,
        num_extracted_stars,
        &scratch.sp_valid_cat_centroids,
        crop_len,
        width * options.match_radius,
        &mut scratch.sp_matched_stars,
        &mut scratch.sp_matches_scratch,
        &mut scratch.sp_matches1_scratch,
    );
    let matched_stars = &scratch.sp_matched_stars;

    // Probability calculation
    let num_star_matches = matched_stars.len();
    // A minimum of 3 non-collinear star matches is mathematically required to determine a
    // unique 3D celestial attitude (rotation matrix R) and overdetermine the 2-parameter distortion fit.
    // While standard Tetra3 solves match 4+ stars, 3 matches can occur when one pattern star
    // is lost due to sensor edge optical distortion, partial obstruction, or catalog brightness cutoff.
    // These 3-match candidates will produce a LowConfidenceMatch (prob >= match_threshold) that downstream
    // systems (e.g. olive-solve IMU fusion) can safely cross-verify against hardware sensors.
    // Any candidate with < 3 matches (0, 1, or 2) is mathematically indeterminate or degenerate and is rejected.
    if num_star_matches < 3 {
        return VerificationResult::None;
    }
    let prob_single_star_mismatch = (crop_len as f64) * options.match_radius.powi(2);
    let p_raw = 1.0 - prob_single_star_mismatch;
    let k_raw = num_extracted_stars as i64 - (num_star_matches as i64 - 2);

    // Safe bounds bypass replicating scipy.stats.binom.cdf behavior
    let prob_mismatch = fast_binomial_cdf(k_raw, num_extracted_stars as u64, p_raw);
    let is_match = prob_mismatch < match_threshold;

    // Early exit if this candidate fails the match_threshold AND we either:
    // 1) are not tracking the best failed match, or
    // 2) this candidate is not strictly better than our current best candidate.
    if !is_match && (!options.return_best_failed_match || prob_mismatch >= best_prob_mismatch) {
        return VerificationResult::None;
    }

    // We passed verification (or found a new best fallback candidate). Complete exact solution details.
    scratch.sp_matched_img_cents.clear();
    scratch.sp_matched_cat_vecs.clear();
    for &(img_idx, cat_idx) in matched_stars {
        scratch
            .sp_matched_img_cents
            .push(scratch.sp_image_centroids_undist[img_idx]);
        scratch
            .sp_matched_cat_vecs
            .push(scratch.sp_valid_cat_vectors[cat_idx]);
    }

    compute_vectors_inplace(
        &scratch.sp_matched_img_cents,
        height,
        width,
        fov,
        &mut scratch.sp_matched_img_vecs,
        num_star_matches,
    );
    let (precise_rotation_matrix, precise_det) = match find_rotation_matrix_and_det_inplace(
        &scratch.sp_matched_img_vecs,
        &scratch.sp_matched_cat_vecs,
        num_star_matches,
    ) {
        Some(res) => res,
        None => return VerificationResult::None,
    };

    let mut k_final = options.distortion;
    if options.distortion.is_some() {
        // Refine fov & distortion using Least Squares System
        // A = [tangent, radius^3], b = [radius]
        // Note: To fully map lstsq in Rust precisely, build A and B for all matched_stars
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &scratch.sp_matched_cat_vecs,
            false,
            &mut scratch.sp_derotated_matched_cat,
            num_star_matches,
        );

        let mut ata_00 = 0.0;
        let mut ata_01 = 0.0;
        let mut ata_11 = 0.0;
        let mut atb_0 = 0.0;
        let mut atb_1 = 0.0;

        for (m_idx, &(img_idx, _)) in matched_stars.iter().enumerate() {
            let r_cent = scratch.sp_image_centroids[img_idx];
            let r_dist = ((r_cent[0] - height / 2.0).powi(2) + (r_cent[1] - width / 2.0).powi(2))
                .sqrt()
                / width
                * 2.0;
            let cat_derot = scratch.sp_derotated_matched_cat[m_idx];
            let tangent = (cat_derot[1].powi(2) + cat_derot[2].powi(2)).sqrt() / cat_derot[0];

            let a0 = tangent;
            let a1 = r_dist.powi(3);
            let b = r_dist;

            ata_00 += a0 * a0;
            ata_01 += a0 * a1;
            ata_11 += a1 * a1;
            atb_0 += a0 * b;
            atb_1 += a1 * b;
        }

        // OPTIMIZATION: Direct 2x2 Matrix Inversion using Cramer's rule
        let det = ata_00 * ata_11 - ata_01 * ata_01;
        if det.abs() > 1e-12 {
            let sol_0 = (ata_11 * atb_0 - ata_01 * atb_1) / det;
            let sol_1 = (ata_00 * atb_1 - ata_01 * atb_0) / det;

            let f_val = sol_0 / (1.0 - sol_1);
            if f_val > 0.0 && f_val.is_finite() {
                k_final = Some(sol_1);
                fov = 2.0 * (1.0 / f_val).atan();
                let centroids = &scratch.sp_image_centroids;
                let out = &mut scratch.sp_image_centroids_undist;
                undistort_centroids_inplace(centroids, height, width, sol_1, out);
                for (m_idx, &(img_idx, _)) in matched_stars.iter().enumerate() {
                    scratch.sp_matched_img_cents[m_idx] =
                        scratch.sp_image_centroids_undist[img_idx];
                }
            }
        }
    }

    compute_vectors_inplace(
        &scratch.sp_matched_img_cents,
        height,
        width,
        fov,
        &mut scratch.sp_matched_img_vecs,
        num_star_matches,
    );
    rotate_vectors_inplace(
        &precise_rotation_matrix,
        &scratch.sp_matched_img_vecs,
        true,
        &mut scratch.sp_final_derotated,
        num_star_matches,
    );

    scratch.sp_distances.clear();
    for m_idx in 0..num_star_matches {
        let row_f = scratch.sp_final_derotated[m_idx];
        let row_c = scratch.sp_matched_cat_vecs[m_idx];
        let dist = ((row_f[0] - row_c[0]).powi(2)
            + (row_f[1] - row_c[1]).powi(2)
            + (row_f[2] - row_c[2]).powi(2))
        .sqrt();
        if dist.is_finite() {
            scratch.sp_distances.push(dist);
        }
    }
    scratch
        .sp_distances
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if scratch.sp_distances.is_empty() {
        return VerificationResult::None;
    }

    let p90_idx = ((0.9 * (scratch.sp_distances.len() - 1) as f64) as usize)
        .min(scratch.sp_distances.len() - 1);
    let p90_err_angle = angle_from_distance(scratch.sp_distances[p90_idx]).to_degrees() * 3600.0;
    let max_err_angle =
        angle_from_distance(*scratch.sp_distances.last().unwrap_or(&0.0)).to_degrees() * 3600.0;

    let mut rms_sum = 0.0;
    for &d in &scratch.sp_distances {
        let a = angle_from_distance(d);
        rms_sum += a * a;
    }
    let rms_err_angle = (rms_sum / scratch.sp_distances.len() as f64)
        .sqrt()
        .to_degrees()
        * 3600.0;

    let ra = precise_rotation_matrix[(0, 1)]
        .atan2(precise_rotation_matrix[(0, 0)])
        .to_degrees()
        .rem_euclid(360.0);
    let dec = precise_rotation_matrix[(0, 2)]
        .atan2(
            (precise_rotation_matrix[(1, 2)].powi(2) + precise_rotation_matrix[(2, 2)].powi(2))
                .sqrt(),
        )
        .to_degrees();

    if let (Some(lat), Some(lst), Some(min_alt)) = (
        options.observer_latitude,
        options.observer_lst,
        options.min_boresight_altitude,
    ) {
        let lat_rad = lat.to_radians();
        let lst_rad = lst.to_radians();
        let zenith = [
            lat_rad.cos() * lst_rad.cos(),
            lat_rad.cos() * lst_rad.sin(),
            lat_rad.sin(),
        ];

        let boresight = [
            precise_rotation_matrix[(0, 0)],
            precise_rotation_matrix[(0, 1)],
            precise_rotation_matrix[(0, 2)],
        ];
        let sin_alt =
            boresight[0] * zenith[0] + boresight[1] * zenith[1] + boresight[2] * zenith[2];
        if sin_alt < min_alt.to_radians().sin() {
            return VerificationResult::None; // Boresight is pointing below the allowed horizon limit
        }
    }

    let mut roll = precise_rotation_matrix[(1, 2)]
        .atan2(precise_rotation_matrix[(2, 2)])
        .to_degrees();
    if precise_det < 0.0 {
        // Assume horizontal mirroring (X-axis flipped).
        // By negating the extracted roll, we recover the true physical spacecraft roll.
        roll = -roll;
    }
    let roll = roll.rem_euclid(360.0);

    let mut precise_rot_arr2 = Array2::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            precise_rot_arr2[[r, c]] = precise_rotation_matrix[(r, c)];
        }
    }

    let status = if is_match {
        SolveStatus::MatchFound
    } else {
        SolveStatus::LowConfidenceMatch
    };

    let mut solution = Solution {
        ra: Some(ra),
        dec: Some(dec),
        roll: Some(roll),
        fov: Some(fov.to_degrees()),
        distortion: k_final,
        rmse: Some(rms_err_angle),
        p90e: Some(p90_err_angle),
        maxe: Some(max_err_angle),
        matches: Some(num_star_matches),
        prob: Some(prob_mismatch * (num_patterns as f64)),
        epoch_equinox: db_props.get("epoch_equinox").cloned(),
        epoch_proper_motion: db_props.get("epoch_proper_motion").cloned(),
        status,
        t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
        is_mirrored: precise_det < 0.0,
        ..Default::default()
    };

    if options.return_rotation_matrix {
        solution.rotation_matrix = Some(precise_rot_arr2);
    }
    if let Some(target_px) = &options.target_pixel {
        let mut px_flat = Vec::with_capacity(target_px.nrows());
        for row in 0..target_px.nrows() {
            px_flat.push([target_px[[row, 0]], target_px[[row, 1]]]);
        }
        if let Some(k) = k_final {
            px_flat = undistort_centroids(&px_flat, height, width, k);
        }
        let target_vector = compute_vectors_flat(&px_flat, height, width, fov);
        let mut rotated_target_vector = vec![[0.0; 3]; target_vector.len()];
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &target_vector,
            true,
            &mut rotated_target_vector,
            target_vector.len(),
        );
        let mut target_ra = Vec::new();
        let mut target_dec = Vec::new();
        for v in rotated_target_vector {
            target_ra.push(v[1].atan2(v[0]).to_degrees().rem_euclid(360.0));
            target_dec.push(90.0 - v[2].acos().to_degrees());
        }
        solution.target_ra = Some(target_ra);
        solution.target_dec = Some(target_dec);
    }
    if let Some(target_sky) = &options.target_sky_coord {
        let mut target_sky_vecs = Vec::with_capacity(target_sky.nrows());
        for row in 0..target_sky.nrows() {
            let ra_rad = target_sky[[row, 0]].to_radians();
            let dec_rad = target_sky[[row, 1]].to_radians();
            target_sky_vecs.push([
                ra_rad.cos() * dec_rad.cos(),
                ra_rad.sin() * dec_rad.cos(),
                dec_rad.sin(),
            ]);
        }
        let mut target_sky_vecs_derot = vec![[0.0; 3]; target_sky_vecs.len()];
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &target_sky_vecs,
            false,
            &mut target_sky_vecs_derot,
            target_sky_vecs.len(),
        );
        let mut target_centroids = vec![[0.0; 2]; target_sky_vecs.len()];
        let mut kept_sky = Vec::new();
        compute_centroids_inplace(
            &target_sky_vecs_derot,
            height,
            width,
            fov,
            &mut target_centroids,
            &mut kept_sky,
            target_sky_vecs.len(),
            !options.allow_out_of_bounds_target_pixel.unwrap_or(false),
        );
        if let Some(k) = k_final {
            for &k_idx in &kept_sky {
                let distorted =
                    distort_centroids(&[target_centroids[k_idx]], height, width, k, 1e-6, 30);
                target_centroids[k_idx] = distorted[0];
            }
        }
        let mut target_y = vec![None; target_sky.nrows()];
        let mut target_x = vec![None; target_sky.nrows()];
        for &k_idx in &kept_sky {
            target_y[k_idx] = Some(target_centroids[k_idx][0]);
            target_x[k_idx] = Some(target_centroids[k_idx][1]);
        }
        solution.target_y = Some(target_y);
        solution.target_x = Some(target_x);
    }
    if options.return_matches {
        let mut m_cents = Vec::new();
        let mut m_stars = Vec::new();
        let mut m_ids = Vec::new();
        for &(img_idx, cat_idx) in matched_stars {
            m_cents.push(scratch.sp_image_centroids_undist[img_idx]);
            let star_idx = scratch.sp_valid_cat_inds[cat_idx];
            let meta = &star_metadata[star_idx];
            m_stars.push([meta.ra.to_degrees(), meta.dec.to_degrees(), meta.mag]);

            // Extract the catalog ID as a Vec (1 element for hip_main/bsc5, 3 elements for tyc_main)
            if let Some(ids) = star_catalog_ids {
                let mut row_ids = Vec::with_capacity(ids.ncols());
                for c in 0..ids.ncols() {
                    row_ids.push(ids[[star_idx, c]]);
                }
                m_ids.push(row_ids);
            }
        }
        solution.matched_centroids = Some(m_cents);
        solution.matched_stars = Some(m_stars);
        if !m_ids.is_empty() {
            solution.matched_cat_id = Some(m_ids);
        }
    }
    if options.return_catalog {
        let mut cat_stars = Vec::new();
        for c_idx in 0..crop_len {
            let star_idx = scratch.sp_valid_cat_inds[c_idx];
            let meta = &star_metadata[star_idx];
            cat_stars.push((
                meta.ra.to_degrees(),
                meta.dec.to_degrees(),
                meta.mag,
                scratch.sp_valid_cat_centroids[c_idx][0],
                scratch.sp_valid_cat_centroids[c_idx][1],
            ));
        }
        solution.catalog_stars = Some(cat_stars);
    }

    if is_match {
        VerificationResult::Success(solution)
    } else {
        VerificationResult::NewBestFailed(prob_mismatch, solution)
    }
}

// OPTIMIZATION: Zero-Allocation Pipeline
// All dynamically sized Vecs required by the setup and combinatorics phases are stored here.
// Reusing this struct completely eliminates all dynamic heap allocations inside solve(),
// preventing allocator fragmentation over hundreds of runs.
#[derive(Default)]
#[doc(hidden)]
pub struct Scratchpads {
    pub sp_pattern_key_list: Vec<u64>,
    pub sp_image_centroids: Vec<[f64; 2]>,
    pub sp_image_centroids_vectors: Vec<[f64; 3]>,
    pub sp_pattern_centroids_inds: Vec<usize>,

    // Core matching scratchpads
    pub sp_image_centroids_undist: Vec<[f64; 2]>,
    pub sp_cat_edges_list: Vec<[f64; 6]>,
    pub sp_cat_vectors_list: Vec<[[f64; 3]; 4]>,
    pub sp_p_cents: Vec<[f64; 2]>,
    pub sp_p_vecs: Vec<[f64; 3]>,
    pub sp_image_pattern_vectors_sorted: Vec<[f64; 3]>,
    pub sp_radii_scratch: Vec<(f64, usize)>,
    pub sp_catalog_pattern_vectors_sorted: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_vectors: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_vectors_derot: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_centroids_all: Vec<[f64; 2]>,
    pub sp_kept: Vec<usize>,
    pub sp_valid_cat_centroids: Vec<[f64; 2]>,
    pub sp_valid_cat_vectors: Vec<[f64; 3]>,
    pub sp_valid_cat_inds: Vec<usize>,
    pub sp_hash_match_inds: Vec<usize>,
    pub sp_precomputed_angles: Vec<f64>,
    pub sp_keep_for_patterns: Vec<bool>,
    pub sp_matched_stars: Vec<(usize, usize)>,
    pub sp_matched_img_cents: Vec<[f64; 2]>,
    pub sp_matched_cat_vecs: Vec<[f64; 3]>,
    pub sp_matched_img_vecs: Vec<[f64; 3]>,
    pub sp_derotated_matched_cat: Vec<[f64; 3]>,
    pub sp_final_derotated: Vec<[f64; 3]>,
    pub sp_distances: Vec<f64>,
    pub sp_matches_scratch: Vec<(usize, usize)>,
    pub sp_matches1_scratch: Vec<(usize, usize)>,
}

impl Scratchpads {
    #[allow(missing_docs)]
    pub fn new(p_size: usize) -> Self {
        let max_size = p_size.max(6);
        Self {
            sp_pattern_key_list: Vec::with_capacity(512),
            sp_image_centroids: Vec::with_capacity(256),
            sp_image_centroids_vectors: Vec::with_capacity(256),
            sp_pattern_centroids_inds: Vec::with_capacity(256),
            sp_image_centroids_undist: Vec::with_capacity(256),
            sp_cat_edges_list: Vec::with_capacity(32),
            sp_cat_vectors_list: Vec::with_capacity(32),

            sp_p_cents: vec![[0.0; 2]; max_size],
            sp_p_vecs: vec![[0.0; 3]; max_size],
            sp_image_pattern_vectors_sorted: vec![[0.0; 3]; max_size],
            sp_radii_scratch: Vec::with_capacity(max_size),
            sp_catalog_pattern_vectors_sorted: vec![[0.0; 3]; max_size],

            sp_nearby_cat_star_vectors: Vec::with_capacity(256),
            sp_nearby_cat_star_vectors_derot: Vec::with_capacity(256),
            sp_nearby_cat_star_centroids_all: Vec::with_capacity(256),
            sp_kept: Vec::with_capacity(256),

            sp_valid_cat_centroids: Vec::with_capacity(256),
            sp_valid_cat_vectors: Vec::with_capacity(256),
            sp_valid_cat_inds: Vec::with_capacity(256),
            sp_hash_match_inds: Vec::with_capacity(32),
            sp_keep_for_patterns: vec![false; max_size],
            sp_precomputed_angles: vec![0.0; max_size * max_size],
            sp_matched_stars: Vec::with_capacity(256),
            sp_matched_img_cents: Vec::with_capacity(256),
            sp_matched_cat_vecs: Vec::with_capacity(256),
            sp_matched_img_vecs: Vec::with_capacity(256),
            sp_derotated_matched_cat: Vec::with_capacity(256),
            sp_final_derotated: Vec::with_capacity(256),
            sp_distances: Vec::with_capacity(256),
            sp_matches_scratch: Vec::with_capacity(64),
            sp_matches1_scratch: Vec::with_capacity(64),
        }
    }
}

// --- Main Engine ---
//
// This highly optimized solver supports standard databases generated by esa/tetra3 and
// smroid/cedar-solve. The databases must use a pattern size of 4, which is hard-coded in the
// database generation functions of both tetra3 and cedar-solve. A smaller pattern size reduces
// the search space but generates false positives. A larger pattern size increases the solve time.
// The original tetra3 algorithm seems to have settled on 4 as the optimal pattern size - its name
// includes 'tetra' after all.
//
//
#[allow(missing_docs)]
/// The main solver engine. Highly optimized for plate solving using cedar-solve databases.
pub struct Solver {
    // OPTIMIZATION: Highly optimized cache-aligned struct lists
    pub star_vectors: Vec<[f64; 3]>,
    pub star_metadata: Vec<StarMetadata>,
    pub pattern_catalog_flat: Vec<u32>,
    // OPTIMIZATION: u16 Hash Probe Table (Memory / Cache Optimization)
    // We use a flat u16 array instead of u32 or a full struct. Halving the footprint from
    // 4 bytes to 2 bytes ensures it fits better into constrained L2 caches like on the Pi Zero 2W.
    pub probe_table: Vec<u16>,
    pub star_kd_tree: ImmutableKdTree<f64, 3>,
    pub pattern_largest_edge: Option<Vec<f32>>,
    pub has_pattern_key_hashes: bool,
    pub star_catalog_ids: Option<Array2<u32>>,
    pub db_props: HashMap<String, f64>,
    pub num_patterns: usize,
    pub linear_probe: bool,
    pub scratch: Scratchpads, // OPTIMIZATION: Instance-level persistent memory context

    // Control flags for Python threading
    pub is_cancelled: Arc<AtomicBool>,
}

impl Solver {
    /// Loads a cedar-solve database from the given path.
    pub fn load_database(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;

        let read_star_table = |arc: &mut ZipArchive<File>,
                               name: &str|
         -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            let mut zf = arc.by_name(name)?;
            let mut buf = Vec::new();
            zf.read_to_end(&mut buf)?;

            let mut cursor = Cursor::new(&buf);
            let npy = NpyFile::new(&mut cursor)?;
            let shape = npy.shape().to_vec();
            if shape.len() != 2 || shape[1] != 6 {
                return Err("star_table must be 2D with 6 columns".into());
            }

            let mut cursor = Cursor::new(&buf);
            let npy2 = NpyFile::new(&mut cursor)?;
            if let Ok(data) = npy2.into_vec::<f64>() {
                return Ok(data);
            }

            let mut cursor = Cursor::new(&buf);
            let npy3 = NpyFile::new(&mut cursor)?;
            let data_f32: Vec<f32> = npy3.into_vec()?;
            let data: Vec<f64> = data_f32.into_iter().map(|v| v as f64).collect();
            Ok(data)
        };

        // tetra3.py optimizes the data type of the pattern_catalog based on the number of patterns.
        let read_pattern_catalog =
            |arc: &mut ZipArchive<File>,
             name: &str|
             -> Result<(Vec<u32>, usize), Box<dyn std::error::Error>> {
                let mut zf = arc.by_name(name)?;
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf)?;

                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor)?;
                let shape = npy.shape().to_vec();
                if shape.len() != 2 {
                    return Err("pattern_catalog must be 2D".into());
                }
                let nrows = shape[0] as usize;

                // Fallback 1: Try u8 (tetra3 uses this for very small catalogs)
                let mut cursor = Cursor::new(&buf);
                if let Ok(npy) = NpyFile::new(&mut cursor) {
                    if let Ok(data_u8) = npy.into_vec::<u8>() {
                        let data: Vec<u32> = data_u8.into_iter().map(|v| v as u32).collect();
                        return Ok((data, nrows));
                    }
                }

                // Fallback 2: Try u16
                let mut cursor = Cursor::new(&buf);
                if let Ok(npy) = NpyFile::new(&mut cursor) {
                    if let Ok(data_u16) = npy.into_vec::<u16>() {
                        let data: Vec<u32> = data_u16.into_iter().map(|v| v as u32).collect();
                        return Ok((data, nrows));
                    }
                }

                // Fallback 3: Try u32
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor)?;
                let data_u32: Vec<u32> = npy.into_vec()?;
                Ok((data_u32, nrows))
            };

        let read_1d_f16_to_f32 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Vec<f32>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                let vec_f16: Vec<half::f16> = npy.into_vec().ok()?;
                Some(vec_f16.into_iter().map(|f| f.to_f32()).collect())
            })
        };

        let read_1d_u16 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Vec<u16>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                npy.into_vec().ok()
            })
        };

        let read_star_catalog_ids =
            |arc: &mut ZipArchive<File>, name: &str| -> Option<Array2<u32>> {
                arc.by_name(name).ok().and_then(|mut zf| {
                    let mut buf = Vec::new();
                    zf.read_to_end(&mut buf).ok()?;

                    // Try 1D u32 (hip_main)
                    let try_1d_u32 = || -> Option<Array2<u32>> {
                        let mut cursor = Cursor::new(&buf);
                        let npy = NpyFile::new(&mut cursor).ok()?;
                        if npy.shape().len() != 1 {
                            return None;
                        }
                        let data = npy.into_vec::<u32>().ok()?;
                        Array2::from_shape_vec((data.len(), 1), data).ok()
                    };

                    // Try 1D u16 (bsc5)
                    let try_1d_u16 = || -> Option<Array2<u32>> {
                        let mut cursor = Cursor::new(&buf);
                        let npy = NpyFile::new(&mut cursor).ok()?;
                        if npy.shape().len() != 1 {
                            return None;
                        }
                        let data = npy.into_vec::<u16>().ok()?;
                        let data_u32: Vec<u32> = data.into_iter().map(|v| v as u32).collect();
                        Array2::from_shape_vec((data_u32.len(), 1), data_u32).ok()
                    };

                    // Try 2D u16 (tyc_main)
                    let try_2d_u16 = || -> Option<Array2<u32>> {
                        let mut cursor = Cursor::new(&buf);
                        let npy = NpyFile::new(&mut cursor).ok()?;
                        let shape = npy.shape();
                        if shape.len() != 2 || shape[1] != 3 {
                            return None;
                        }

                        let rows = shape[0] as usize;
                        let cols = shape[1] as usize;
                        let data = npy.into_vec::<u16>().ok()?;
                        let data_u32: Vec<u32> = data.into_iter().map(|v| v as u32).collect();
                        Array2::from_shape_vec((rows, cols), data_u32).ok()
                    };

                    // Chain the attempts
                    try_1d_u32().or_else(try_1d_u16).or_else(try_2d_u16)
                })
            };

        let (pattern_catalog_flat, num_patterns_from_arr) =
            read_pattern_catalog(&mut archive, "pattern_catalog.npy")?;
        let star_table_data = read_star_table(&mut archive, "star_table.npy")?;
        let pattern_largest_edge = read_1d_f16_to_f32(&mut archive, "pattern_largest_edge.npy");
        let pattern_key_hashes = read_1d_u16(&mut archive, "pattern_key_hashes.npy");
        let star_catalog_ids = read_star_catalog_ids(&mut archive, "star_catalog_IDs.npy");

        // OPTIMIZATION: Convert massive Array2 allocations to deeply mapped fast internal native slices
        let num_stars = star_table_data.len() / 6;
        let mut star_vectors = Vec::with_capacity(num_stars);
        let mut star_metadata = Vec::with_capacity(num_stars);
        for chunk in star_table_data.chunks_exact(6) {
            star_metadata.push(StarMetadata {
                ra: chunk[0],
                dec: chunk[1],
                mag: chunk[5],
            });
            star_vectors.push([chunk[2], chunk[3], chunk[4]]);
        }

        let num_patterns_allocated = pattern_catalog_flat.len() / 4;
        let mut probe_table = Vec::with_capacity(num_patterns_allocated);
        let has_pattern_key_hashes = pattern_key_hashes.is_some();
        let hashes_ref = pattern_key_hashes.as_ref().map(|a| a.as_slice());

        for i in 0..num_patterns_allocated {
            let row_start = i * 4;
            if pattern_catalog_flat[row_start] == 0
                && pattern_catalog_flat[row_start + 1] == 0
                && pattern_catalog_flat[row_start + 2] == 0
                && pattern_catalog_flat[row_start + 3] == 0
            {
                probe_table.push(u16::MAX);
            } else {
                if has_pattern_key_hashes {
                    let mut h = hashes_ref.unwrap()[i];
                    if h == u16::MAX {
                        h = u16::MAX - 1;
                    }
                    probe_table.push(h);
                } else {
                    probe_table.push(0);
                }
            }
        }

        let mut points: Vec<[f64; 3]> = Vec::with_capacity(star_vectors.len());
        for vec in star_vectors.iter() {
            points.push(*vec);
        }
        let star_kd_tree = ImmutableKdTree::new_from_slice(&points);

        let mut num_patterns = num_patterns_from_arr / 2;
        let mut db_props = HashMap::new();
        let mut linear_probe = false;

        db_props.insert("pattern_size".to_string(), 4.0);
        db_props.insert("pattern_bins".to_string(), 50.0);
        db_props.insert("pattern_max_error".to_string(), 0.002);
        db_props.insert("verification_stars_per_fov".to_string(), 10.0);
        db_props.insert("max_fov".to_string(), 20.0);
        db_props.insert("min_fov".to_string(), 20.0);
        db_props.insert("epoch_equinox".to_string(), 2000.0);
        db_props.insert("presort_patterns".to_string(), 0.0);

        if let Ok(mut zf) = archive.by_name("props_packed.npy") {
            let mut buf = Vec::new();
            if zf.read_to_end(&mut buf).is_ok() {
                let mut cursor = Cursor::new(&buf);
                // NpyFile::new parses and skips the header, advancing the cursor to the payload
                if NpyFile::new(&mut cursor).is_ok() {
                    let mut data = Vec::new();
                    // Read the remaining payload bytes directly from the cursor
                    if cursor.read_to_end(&mut data).is_ok() {
                        let len = data.len();

                        // 828 bytes = cedar-solve schema
                        if len >= 828 {
                            let mut hash_type = String::new();
                            for i in 0..64 {
                                let offset = 256 + (i * 4);
                                let c = data[offset];
                                if c == 0 {
                                    break;
                                }
                                hash_type.push(c as char);
                            }
                            if hash_type.trim() == "linear_probe" {
                                linear_probe = true;
                            }

                            let p_size = u16::from_le_bytes([data[512], data[513]]);
                            db_props.insert("pattern_size".to_string(), p_size as f64);
                            let p_bins = u16::from_le_bytes([data[514], data[515]]);
                            db_props.insert("pattern_bins".to_string(), p_bins as f64);
                            let p_max_err =
                                f32::from_le_bytes([data[516], data[517], data[518], data[519]]);
                            db_props.insert("pattern_max_error".to_string(), p_max_err as f64);
                            let max_fov =
                                f32::from_le_bytes([data[520], data[521], data[522], data[523]]);
                            db_props.insert("max_fov".to_string(), max_fov as f64);
                            let min_fov =
                                f32::from_le_bytes([data[524], data[525], data[526], data[527]]);
                            db_props.insert("min_fov".to_string(), min_fov as f64);
                            let eq = u16::from_le_bytes([data[784], data[785]]);
                            db_props.insert("epoch_equinox".to_string(), eq as f64);
                            let pm =
                                f32::from_le_bytes([data[786], data[787], data[788], data[789]]);
                            db_props.insert("epoch_proper_motion".to_string(), pm as f64);
                            let vs = u16::from_le_bytes([data[800], data[801]]);
                            db_props.insert("verification_stars_per_fov".to_string(), vs as f64);
                            let presort = data[823] != 0;
                            db_props.insert(
                                "presort_patterns".to_string(),
                                if presort { 1.0 } else { 0.0 },
                            );
                            num_patterns =
                                u32::from_le_bytes([data[824], data[825], data[826], data[827]])
                                    as usize;
                        }
                        // ~560/568 bytes = standard tetra3 schema
                        else if len >= 560 {
                            let p_size = u16::from_le_bytes([data[256], data[257]]);
                            db_props.insert("pattern_size".to_string(), p_size as f64);
                            let p_bins = u16::from_le_bytes([data[258], data[259]]);
                            db_props.insert("pattern_bins".to_string(), p_bins as f64);
                            let p_max_err =
                                f32::from_le_bytes([data[260], data[261], data[262], data[263]]);
                            db_props.insert("pattern_max_error".to_string(), p_max_err as f64);
                            let max_fov =
                                f32::from_le_bytes([data[264], data[265], data[266], data[267]]);
                            db_props.insert("max_fov".to_string(), max_fov as f64);
                            let min_fov =
                                f32::from_le_bytes([data[268], data[269], data[270], data[271]]);
                            db_props.insert("min_fov".to_string(), min_fov as f64);
                            let eq = u16::from_le_bytes([data[528], data[529]]);
                            db_props.insert("epoch_equinox".to_string(), eq as f64);
                            let pm =
                                f32::from_le_bytes([data[530], data[531], data[532], data[533]]);
                            db_props.insert("epoch_proper_motion".to_string(), pm as f64);
                            let vs = u16::from_le_bytes([data[536], data[537]]);
                            db_props.insert("verification_stars_per_fov".to_string(), vs as f64);
                            let presort = data[559] != 0;
                            db_props.insert(
                                "presort_patterns".to_string(),
                                if presort { 1.0 } else { 0.0 },
                            );
                            // num_patterns is already set globally based on pattern_catalog_arr.nrows() / 2
                        }
                    }
                }
            }
        }

        let p_size = *db_props.get("pattern_size").unwrap_or(&4.0) as usize;
        if p_size != 4 {
            return Err("Only databases with a pattern size of 4 are supported".into());
        }

        let is_cancelled = Arc::new(AtomicBool::new(false));

        Ok(Solver {
            star_vectors,
            star_metadata,
            pattern_catalog_flat,
            probe_table,
            star_kd_tree,
            pattern_largest_edge,
            has_pattern_key_hashes,
            star_catalog_ids,
            db_props,
            num_patterns,
            linear_probe,
            scratch: Scratchpads::new(p_size),
            is_cancelled,
        })
    }

    /// Cancels an ongoing plate solve attempt (usually called from another thread).
    pub fn cancel_solve(&self) {
        self.is_cancelled.store(true, Ordering::Relaxed);
    }

    #[inline(always)]
    fn compute_pattern_key_hash(pattern_key: &[usize; 5], mults: &[u64; 4]) -> u64 {
        (pattern_key[0] as u64)
            + (pattern_key[1] as u64) * mults[0]
            + (pattern_key[2] as u64) * mults[1]
            + (pattern_key[3] as u64) * mults[2]
            + (pattern_key[4] as u64) * mults[3]
    }

    #[inline(always)]
    fn pattern_key_hash_to_index(hash: u64, max_index: u64, linear_probe: bool) -> u64 {
        if linear_probe {
            hash % max_index
        } else {
            hash.wrapping_mul(MAGIC_RAND) % max_index
        }
    }

    // OPTIMIZATION: u64 Primitive Sorting (Memory / Happy Path)
    // Packs a 5-element pattern key (10 bits each) and its distance into a single 64-bit unsigned integer.
    // This massively speeds up the combinations sort_unstable() and reduces memory allocation.
    #[inline(always)]
    fn encode_pattern_key(dist: usize, k: [usize; 5]) -> u64 {
        ((dist as u64) << 40)
            | ((k[4] as u64) << 32)
            | ((k[3] as u64) << 24)
            | ((k[2] as u64) << 16)
            | ((k[1] as u64) << 8)
            | (k[0] as u64)
    }

    #[inline(always)]
    fn decode_pattern_key(val: u64) -> [usize; 5] {
        [
            (val & 0xff) as usize,
            ((val >> 8) & 0xff) as usize,
            ((val >> 16) & 0xff) as usize,
            ((val >> 24) & 0xff) as usize,
            ((val >> 32) & 0xff) as usize,
        ]
    }

    fn get_table_indices_from_hash_inplace(
        probe_table: &[u16],
        hash_index: u64,
        linear_probe: bool,
        has_hashes: bool,
        key_hash16: u16,
        out_found: &mut Vec<usize>,
    ) {
        out_found.clear();
        let max_ind = probe_table.len();
        let mut i = (hash_index % (max_ind as u64)) as usize;

        if linear_probe {
            loop {
                let probe_val = probe_table[i];
                if probe_val == u16::MAX {
                    break;
                }
                if !has_hashes || probe_val == key_hash16 {
                    out_found.push(i);
                }
                i += 1;
                if i == max_ind {
                    i = 0;
                }
            }
        } else {
            let mut step = 1;
            loop {
                let probe_val = probe_table[i];
                if probe_val == u16::MAX {
                    break;
                }
                if !has_hashes || probe_val == key_hash16 {
                    out_found.push(i);
                }
                i += step;
                while i >= max_ind {
                    i -= max_ind;
                }
                step += 2;
            }
        }
    }

    fn get_all_patterns_for_index_inplace(
        pattern_key_hash: u64,
        hash_index: u64,
        inv_largest_edge: f64,
        fov_estimate: Option<f64>,
        fov_max_error: Option<f64>,
        pattern_catalog_flat: &[u32],
        probe_table: &[u16],
        p_size: usize,
        has_pattern_key_hashes: bool,
        pattern_largest_edge: &Option<Vec<f32>>,
        star_vectors: &[[f64; 3]],
        linear_probe: bool,
        sp_hash_match_inds: &mut Vec<usize>,
        out_edges: &mut Vec<[f64; 6]>,
        out_vectors: &mut Vec<[[f64; 3]; 4]>,
    ) {
        let mut key_hash16 = (pattern_key_hash & 0xffff) as u16;
        if key_hash16 == u16::MAX {
            key_hash16 = u16::MAX - 1;
        }

        Self::get_table_indices_from_hash_inplace(
            probe_table,
            hash_index,
            linear_probe,
            has_pattern_key_hashes,
            key_hash16,
            sp_hash_match_inds,
        );
        if sp_hash_match_inds.is_empty() {
            out_edges.clear();
            out_vectors.clear();
            return;
        }

        if let (Some(largest_edges), Some(f_est), Some(f_err)) =
            (pattern_largest_edge, fov_estimate, fov_max_error)
        {
            let fov_factor = f_est * 0.001 * inv_largest_edge;
            sp_hash_match_inds.retain(|&idx| {
                let cat_largest_edge = largest_edges[idx] as f64;
                let fov2 = cat_largest_edge * fov_factor;
                (fov2 - f_est).abs() < f_err
            });
        }

        let num_matches = sp_hash_match_inds.len();
        while out_edges.len() < num_matches {
            out_edges.push([0.0; 6]);
        }
        while out_vectors.len() < num_matches {
            out_vectors.push([[0.0; 3]; 4]);
        }
        out_edges.truncate(num_matches);
        out_vectors.truncate(num_matches);

        for (out_idx, &idx) in sp_hash_match_inds.iter().enumerate() {
            let row_start = idx * p_size;
            let vecs = &mut out_vectors[out_idx];
            for i in 0..p_size {
                let star_id = pattern_catalog_flat[row_start + i] as usize;
                let v = star_vectors[star_id];
                vecs[i] = v;
            }

            let edges_vec = &mut out_edges[out_idx];
            let mut e_idx = 0;
            for i in 0..p_size {
                for j in (i + 1)..p_size {
                    let d0 = vecs[i][0] - vecs[j][0];
                    let d1 = vecs[i][1] - vecs[j][1];
                    let d2 = vecs[i][2] - vecs[j][2];
                    edges_vec[e_idx] = angle_from_distance((d0 * d0 + d1 * d1 + d2 * d2).sqrt());
                    e_idx += 1;
                }
            }
            macro_rules! swap_e {
                ($a:expr, $b:expr) => {
                    if edges_vec[$a] > edges_vec[$b] {
                        let tmp = edges_vec[$a];
                        edges_vec[$a] = edges_vec[$b];
                        edges_vec[$b] = tmp;
                    }
                };
            }
            swap_e!(1, 2);
            swap_e!(4, 5);
            swap_e!(0, 2);
            swap_e!(3, 5);
            swap_e!(0, 1);
            swap_e!(3, 4);
            swap_e!(1, 4);
            swap_e!(0, 3);
            swap_e!(2, 5);
            swap_e!(1, 3);
            swap_e!(2, 4);
            swap_e!(2, 3);
        }
    }

    /// Evaluates a plate solution against an arbitrary list of image centroids to
    /// determine the actual matching centroids.
    ///
    /// The function takes the sky attitude (rotation matrix) and Field of View (FOV)
    /// from the provided `Solution`, queries the star catalog KD-Tree for nearby stars,
    /// projects them back onto the image plane, and runs the centroid matching algorithm
    /// against the provided `image_centroids`.
    ///
    /// # Arguments
    ///
    /// * `solution` - The successful `Solution` containing the rotation matrix and FOV.
    /// * `image_centroids` - A slice of `[y, x]` coordinates representing the centroids to match.
    /// * `size` - The `(height, width)` of the sensor image.
    /// * `options` - The `SolveOptions` that supply the maximum match radius.
    ///
    /// # Returns
    /// Returns `Some(Vec<[f64; 2]>)` containing the actual provided `image_centroids` that
    /// matched with projected catalog stars, or `None` if the provided solution
    /// is missing a rotation matrix or FOV.
    pub fn get_matches_for_centroids(
        &self,
        solution: &Solution,
        image_centroids: &[[f64; 2]],
        size: (f64, f64),
        options: &SolveOptions,
    ) -> Option<Vec<[f64; 2]>> {
        let (height, width) = size;
        let fov = solution.fov?.to_radians();
        let rotation_matrix = solution.rotation_matrix.as_ref()?;
        let num_extracted_stars = image_centroids.len();

        let fov_diagonal_rad = fov * ((width * width + height * height).sqrt() / width);
        let image_center_vector = [
            rotation_matrix[(0, 0)],
            rotation_matrix[(0, 1)],
            rotation_matrix[(0, 2)],
        ];

        let max_dist_sq = distance_from_angle(fov_diagonal_rad / 2.0).powi(2) + 1e-8;
        let mut nearby_nodes = self
            .star_kd_tree
            .within_unsorted::<SquaredEuclidean>(&image_center_vector, max_dist_sq);

        nearby_nodes.sort_unstable_by_key(|n| n.item);

        let num_nearby = nearby_nodes.len();
        if num_nearby == 0 {
            return Some(Vec::new());
        }

        let scale_factor_cent = -width / 2.0 / (fov / 2.0).tan();
        let img_center_y = height / 2.0;
        let img_center_x = width / 2.0;

        let r = rotation_matrix;

        let mut valid_cat_centroids = Vec::with_capacity(nearby_nodes.len());

        for node in &nearby_nodes {
            let star_idx = node.item as usize;
            let vec = self.star_vectors[star_idx];

            let v0 =
                r[(0, 0)] * vec[0] as f64 + r[(0, 1)] * vec[1] as f64 + r[(0, 2)] * vec[2] as f64;
            let v1 =
                r[(1, 0)] * vec[0] as f64 + r[(1, 1)] * vec[1] as f64 + r[(1, 2)] * vec[2] as f64;
            let v2 =
                r[(2, 0)] * vec[0] as f64 + r[(2, 1)] * vec[1] as f64 + r[(2, 2)] * vec[2] as f64;

            let cy = scale_factor_cent * (v2 / v0) + img_center_y;
            let cx = scale_factor_cent * (v1 / v0) + img_center_x;

            if cy > 0.0 && cx > 0.0 && cy < height && cx < width {
                valid_cat_centroids.push([cy, cx]);
            }
        }

        let mut scratch_sp_image_centroids_undist = Vec::new();
        undistort_centroids_inplace(
            image_centroids,
            height,
            width,
            solution.distortion.unwrap_or(0.0),
            &mut scratch_sp_image_centroids_undist,
        );
        let image_centroids_undist = &scratch_sp_image_centroids_undist;

        let mut sp_matched_stars = Vec::new();
        let mut sp_matches_scratch = Vec::new();
        let mut sp_matches1_scratch = Vec::new();

        find_centroid_matches_inplace(
            image_centroids_undist,
            num_extracted_stars,
            &valid_cat_centroids,
            valid_cat_centroids.len(),
            width * options.match_radius,
            &mut sp_matched_stars,
            &mut sp_matches_scratch,
            &mut sp_matches1_scratch,
        );

        let mut result_centroids = Vec::with_capacity(sp_matched_stars.len());
        for (img_idx, _) in sp_matched_stars {
            result_centroids.push(image_centroids[img_idx]);
        }

        Some(result_centroids)
    }

    /// Attempts to plate solve using the given pre-extracted star centroids.
    pub fn solve(
        &mut self,
        star_centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
    ) -> Solution {
        let t0_solve = Instant::now();
        let (height, width) = size;

        self.is_cancelled.store(false, Ordering::Relaxed);

        let fov_initial = options
            .fov_estimate
            .map(|f| f.to_radians())
            .unwrap_or_else(|| {
                let max_f = self.db_props.get("max_fov").unwrap_or(&20.0);
                let min_f = self.db_props.get("min_fov").unwrap_or(&10.0);
                ((max_f + min_f) / 2.0).to_radians()
            });

        // Hardcoded pattern size
        let p_size = 4;

        let p_bins = *self.db_props.get("pattern_bins").unwrap_or(&50.0) as usize;
        let bf = p_bins as u64;

        // OPTIMIZATION: Hash Constant Hoisting
        // Precalculate the power multipliers for p_bins once per solve to avoid
        // doing 3 redundant multiplications per generated combination.
        let hash_mults = [bf, bf * bf, bf * bf * bf, bf * bf * bf * bf];

        let verification_stars = *self
            .db_props
            .get("verification_stars_per_fov")
            .unwrap_or(&10.0) as usize;

        // OPTIMIZATION: Loop-Invariant Code Motion
        // Pre-convert to radians so we don't do floating-point math in the hot loop
        let fov_est_rad = options.fov_estimate.map(|x| x.to_radians());
        let fov_err_rad = options.fov_max_error.map(|x| x.to_radians());
        let p_max_err = options
            .match_max_error
            .max(*self.db_props.get("pattern_max_error").unwrap_or(&0.002));
        let match_threshold = options.match_threshold / (self.num_patterns as f64);
        let presorted = *self.db_props.get("presort_patterns").unwrap_or(&0.0) == 1.0;

        // OPTIMIZATION: Early physical rejection
        // Pre-compute zenith vector once per solve
        let observer_zenith =
            if let (Some(lat), Some(lst)) = (options.observer_latitude, options.observer_lst) {
                let lat_rad = lat.to_radians();
                let lst_rad = lst.to_radians();
                Some([
                    lat_rad.cos() * lst_rad.cos(),
                    lat_rad.cos() * lst_rad.sin(),
                    lat_rad.sin(),
                ])
            } else {
                None
            };

        let num_centroids_raw = star_centroids.nrows();
        if num_centroids_raw < p_size {
            return Solution {
                status: SolveStatus::TooFew,
                t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }

        // OPTIMIZATION: O(N * K) Spatial Thinning
        // Mathematically simplified: width * (0.6 * fov / sqrt(stars)) / fov removes the fov operations.
        // We also iterate exclusively over previously kept stars (`sp_pattern_centroids_inds`)
        // instead of an O(N^2) loop with boolean checks, heavily reducing inner iterations on dense fields.
        let pattern_stars_separation_pixels = width * 0.6 / (verification_stars as f64).sqrt();
        let sep_sq = pattern_stars_separation_pixels.powi(2);

        let scratch = &mut self.scratch;
        scratch.sp_pattern_centroids_inds.clear();

        for i in 0..num_centroids_raw {
            let mut occupied = false;
            let c_i_0 = star_centroids[[i, 0]];
            let c_i_1 = star_centroids[[i, 1]];

            for &j in &scratch.sp_pattern_centroids_inds {
                if (c_i_0 - star_centroids[[j, 0]]).powi(2)
                    + (c_i_1 - star_centroids[[j, 1]]).powi(2)
                    < sep_sq
                {
                    occupied = true;
                    break;
                }
            }
            if !occupied {
                scratch.sp_pattern_centroids_inds.push(i);
            }
        }

        let mut num_extracted_stars = num_centroids_raw;
        if num_centroids_raw > verification_stars {
            num_extracted_stars = verification_stars;
            scratch
                .sp_pattern_centroids_inds
                .retain(|&i| i < num_extracted_stars);
        }

        // Maintain the original full set of image_centroids for the final matrix building
        scratch.sp_image_centroids.clear();
        for i in 0..num_extracted_stars {
            scratch
                .sp_image_centroids
                .push([star_centroids[[i, 0]], star_centroids[[i, 1]]]);
        }

        if let Some(k) = options.distortion {
            let cents = &scratch.sp_image_centroids;
            let undist = &mut scratch.sp_image_centroids_undist;
            undistort_centroids_inplace(cents, height, width, k, undist);
        } else {
            scratch.sp_image_centroids_undist.clear();
            scratch
                .sp_image_centroids_undist
                .extend_from_slice(&scratch.sp_image_centroids);
        }

        let undist = &scratch.sp_image_centroids_undist;
        let vecs = &mut scratch.sp_image_centroids_vectors;
        compute_vectors_inplace(undist, height, width, fov_initial, vecs, undist.len());

        // OPTIMIZATION: Lazy Angle Precomputation (Happy Path Optimization)
        // We only compute distances on-demand and cache them. For successful solves
        // this skips thousands of unnecessary sqrt/asin operations.
        let num_vecs = scratch.sp_image_centroids_vectors.len();
        scratch.sp_precomputed_angles.clear();
        scratch
            .sp_precomputed_angles
            .resize(num_vecs * num_vecs, -1.0);

        let scratch = &mut self.scratch;
        let star_kd_tree = &self.star_kd_tree;
        let star_vectors = &self.star_vectors;
        let star_metadata = &self.star_metadata;
        let pattern_catalog_flat = &self.pattern_catalog_flat;
        let linear_probe = self.linear_probe;

        let n_inds = scratch.sp_pattern_centroids_inds.len();

        // Fail fast: if spatial thinning leaves us with too few stars to form a single pattern, abort.
        if n_inds < p_size {
            return Solution {
                status: SolveStatus::NoMatch,
                t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }

        // Tracking state for recovering the best match that failed the threshold.
        // When `options.return_best_failed_match` is enabled, we track the candidate pattern
        // with the globally lowest false-positive probability (`prob_mismatch`).
        // If no pattern reaches the strict `match_threshold`, this best candidate is returned
        // at the end with `SolveStatus::LowConfidenceMatch`.
        let mut best_prob_mismatch = f64::INFINITY;
        let mut best_candidate_solution: Option<Solution> = None;

        // -------------------------------------------------------------
        // Allocation-free native iteration mirroring breadth-first order
        // -------------------------------------------------------------
        let mut solver_idx = 0;
        let p_bins_f64 = p_bins as f64;
        for l in 3..n_inds {
            for k in 2..l {
                for j in 1..k {
                    solver_idx += 1;
                    if solver_idx & 127 == 0 {
                        if let Some(timeout_ms) = options.solve_timeout_ms {
                            if t0_solve.elapsed().as_secs_f64() * 1000.0 > timeout_ms {
                                return Solution {
                                    status: SolveStatus::Timeout,
                                    t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                                    ..Default::default()
                                };
                            }
                        }
                        if self.is_cancelled.load(Ordering::Relaxed) {
                            return Solution {
                                status: SolveStatus::Cancelled,
                                t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                                ..Default::default()
                            };
                        }
                    }
                    for i in 0..j {
                        let p_i = scratch.sp_pattern_centroids_inds[i];
                        let p_j = scratch.sp_pattern_centroids_inds[j];
                        let p_k = scratch.sp_pattern_centroids_inds[k];
                        let p_l = scratch.sp_pattern_centroids_inds[l];

                        macro_rules! get_angle {
                            ($p1:expr, $p2:expr) => {{
                                let idx = $p1 * num_vecs + $p2;
                                let mut ang = scratch.sp_precomputed_angles[idx];
                                if ang < 0.0 {
                                    let v_i = scratch.sp_image_centroids_vectors[$p1];
                                    let v_j = scratch.sp_image_centroids_vectors[$p2];
                                    let dist = ((v_i[0] - v_j[0]).powi(2)
                                        + (v_i[1] - v_j[1]).powi(2)
                                        + (v_i[2] - v_j[2]).powi(2))
                                    .sqrt();
                                    ang = angle_from_distance(dist);
                                    scratch.sp_precomputed_angles[$p1 * num_vecs + $p2] = ang;
                                    scratch.sp_precomputed_angles[$p2 * num_vecs + $p1] = ang;
                                }
                                ang
                            }};
                        }

                        // Lazy lookup/compute for pairwise distance angle metrics
                        let edges = [
                            get_angle!(p_i, p_j),
                            get_angle!(p_i, p_k),
                            get_angle!(p_i, p_l),
                            get_angle!(p_j, p_k),
                            get_angle!(p_j, p_l),
                            get_angle!(p_k, p_l),
                        ];

                        // Fast 6-element sorting network
                        let mut e = edges;
                        macro_rules! swap {
                            ($a:expr, $b:expr) => {
                                if e[$a] > e[$b] {
                                    let tmp = e[$a];
                                    e[$a] = e[$b];
                                    e[$b] = tmp;
                                }
                            };
                        }
                        swap!(1, 2);
                        swap!(4, 5);
                        swap!(0, 2);
                        swap!(3, 5);
                        swap!(0, 1);
                        swap!(3, 4);
                        swap!(1, 4);
                        swap!(0, 3);
                        swap!(2, 5);
                        swap!(1, 3);
                        swap!(2, 4);
                        swap!(2, 3);
                        let edges = e;

                        let image_pattern_largest_edge = edges[5];

                        // OPTIMIZATION: Division Elimination (Happy Path Optimization)
                        // Precalculate the reciprocal to replace 5 heavy floating-point divisions with fast multiplications.
                        let inv_largest_edge = 1.0 / image_pattern_largest_edge;

                        // Min/max edge ratio bounds
                        let mut key_space_min = [0; 5];
                        let mut key_space_max = [0; 5];
                        let mut target_keys = [0isize; 5];

                        for x in 0..5 {
                            let ratio = edges[x] * inv_largest_edge;
                            key_space_min[x] = ((ratio - p_max_err).max(0.0) * p_bins_f64) as usize;
                            key_space_max[x] = ((ratio + p_max_err).min(1.0) * p_bins_f64) as usize;
                            target_keys[x] = (ratio * p_bins_f64) as isize;
                        }

                        // Generate search space combinations via zero-allocation DFS (replaces Cartesian product)
                        scratch.sp_pattern_key_list.clear();
                        for k0 in key_space_min[0]..=key_space_max[0] {
                            // OPTIMIZATION: Loop-Invariant Code Motion (Hoisted hash key difference calculations)
                            let diff0 = k0 as isize - target_keys[0];
                            let dist0 = diff0 * diff0;
                            for k1 in key_space_min[1].max(k0)..=key_space_max[1] {
                                let diff1 = k1 as isize - target_keys[1];
                                let dist1 = dist0 + diff1 * diff1;
                                for k2 in key_space_min[2].max(k1)..=key_space_max[2] {
                                    let diff2 = k2 as isize - target_keys[2];
                                    let dist2 = dist1 + diff2 * diff2;
                                    for k3 in key_space_min[3].max(k2)..=key_space_max[3] {
                                        let diff3 = k3 as isize - target_keys[3];
                                        let dist3 = dist2 + diff3 * diff3;
                                        for k4 in key_space_min[4].max(k3)..=key_space_max[4] {
                                            let diff4 = k4 as isize - target_keys[4];
                                            let dist4 = dist3 + diff4 * diff4;
                                            scratch.sp_pattern_key_list.push(
                                                Self::encode_pattern_key(
                                                    dist4 as usize,
                                                    [k0, k1, k2, k3, k4],
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // OPTIMIZATION: Sort by distance to center to search the most likely patterns first.
                        scratch.sp_pattern_key_list.sort_unstable();

                        let mut image_pattern_largest_distance = None;

                        for key_idx in 0..scratch.sp_pattern_key_list.len() {
                            let pattern_key =
                                Self::decode_pattern_key(scratch.sp_pattern_key_list[key_idx]);
                            let pattern_key_hash =
                                Self::compute_pattern_key_hash(&pattern_key, &hash_mults);
                            let hash_index = Self::pattern_key_hash_to_index(
                                pattern_key_hash,
                                (pattern_catalog_flat.len() / p_size) as u64,
                                linear_probe,
                            );

                            Self::get_all_patterns_for_index_inplace(
                                pattern_key_hash,
                                hash_index,
                                inv_largest_edge,
                                fov_est_rad,
                                fov_err_rad,
                                pattern_catalog_flat,
                                &self.probe_table,
                                p_size,
                                self.has_pattern_key_hashes,
                                &self.pattern_largest_edge,
                                &self.star_vectors,
                                linear_probe,
                                &mut scratch.sp_hash_match_inds,
                                &mut scratch.sp_cat_edges_list,
                                &mut scratch.sp_cat_vectors_list,
                            );

                            // OPTIMIZATION: Reciprocal Multiplication (Replaces expensive floating-point divisions)
                            let inv_img_largest = 1.0 / image_pattern_largest_edge;
                            let min_0 = edges[0] * inv_img_largest - p_max_err;
                            let max_0 = edges[0] * inv_img_largest + p_max_err;
                            let min_1 = edges[1] * inv_img_largest - p_max_err;
                            let max_1 = edges[1] * inv_img_largest + p_max_err;
                            let min_2 = edges[2] * inv_img_largest - p_max_err;
                            let max_2 = edges[2] * inv_img_largest + p_max_err;
                            let min_3 = edges[3] * inv_img_largest - p_max_err;
                            let max_3 = edges[3] * inv_img_largest + p_max_err;
                            let min_4 = edges[4] * inv_img_largest - p_max_err;
                            let max_4 = edges[4] * inv_img_largest + p_max_err;

                            for cat_idx in 0..scratch.sp_cat_edges_list.len() {
                                let cat_edges = &scratch.sp_cat_edges_list[cat_idx];
                                let catalog_largest_edge = cat_edges[5];
                                let inv_cat = 1.0 / catalog_largest_edge;

                                let mut valid = true;
                                // OPTIMIZATION: Unrolled Loops (Unrolled 5-element edge comparison loop)
                                let c0 = cat_edges[0] * inv_cat;
                                if c0 < min_0 || c0 > max_0 {
                                    valid = false;
                                } else {
                                    let c1 = cat_edges[1] * inv_cat;
                                    if c1 < min_1 || c1 > max_1 {
                                        valid = false;
                                    } else {
                                        let c2 = cat_edges[2] * inv_cat;
                                        if c2 < min_2 || c2 > max_2 {
                                            valid = false;
                                        } else {
                                            let c3 = cat_edges[3] * inv_cat;
                                            if c3 < min_3 || c3 > max_3 {
                                                valid = false;
                                            } else {
                                                let c4 = cat_edges[4] * inv_cat;
                                                if c4 < min_4 || c4 > max_4 {
                                                    valid = false;
                                                }
                                            }
                                        }
                                    }
                                }
                                if !valid {
                                    continue;
                                }

                                // We have a matched pattern! Calculate refined FOV
                                let fov;
                                if options.fov_estimate.is_some() {
                                    fov = catalog_largest_edge / image_pattern_largest_edge
                                        * fov_initial;
                                } else {
                                    if image_pattern_largest_distance.is_none() {
                                        let pts = [
                                            scratch.sp_image_centroids_undist[p_i],
                                            scratch.sp_image_centroids_undist[p_j],
                                            scratch.sp_image_centroids_undist[p_k],
                                            scratch.sp_image_centroids_undist[p_l],
                                        ];
                                        let mut max_dist = 0.0;
                                        for r in 0..4 {
                                            for c in (r + 1)..4 {
                                                let d = ((pts[r][0] - pts[c][0]).powi(2)
                                                    + (pts[r][1] - pts[c][1]).powi(2))
                                                .sqrt();
                                                if d > max_dist {
                                                    max_dist = d;
                                                }
                                            }
                                        }
                                        image_pattern_largest_distance = Some(max_dist);
                                    }
                                    let f = image_pattern_largest_distance.unwrap()
                                        / 2.0
                                        / (catalog_largest_edge / 2.0).tan();
                                    fov = 2.0 * (width / 2.0 / f).atan();
                                }

                                // Re-calculate vectors uniquely sorted by radius to centroid
                                let pts = [
                                    scratch.sp_image_centroids_undist[p_i],
                                    scratch.sp_image_centroids_undist[p_j],
                                    scratch.sp_image_centroids_undist[p_k],
                                    scratch.sp_image_centroids_undist[p_l],
                                ];
                                compute_vectors_inplace(
                                    &pts,
                                    height,
                                    width,
                                    fov,
                                    &mut scratch.sp_p_vecs,
                                    4,
                                );
                                sort_vectors_by_radius_inplace(
                                    &scratch.sp_p_vecs,
                                    &mut scratch.sp_image_pattern_vectors_sorted,
                                    &mut scratch.sp_radii_scratch,
                                    4,
                                );

                                if presorted {
                                    scratch.sp_catalog_pattern_vectors_sorted[..4].copy_from_slice(
                                        &scratch.sp_cat_vectors_list[cat_idx][..4],
                                    );
                                } else {
                                    sort_vectors_by_radius_inplace(
                                        &scratch.sp_cat_vectors_list[cat_idx],
                                        &mut scratch.sp_catalog_pattern_vectors_sorted,
                                        &mut scratch.sp_radii_scratch,
                                        4,
                                    );
                                };

                                // If any of the 4 matched catalog stars are physically below the horizon
                                // right now, this pattern is mathematically impossible to see. Reject instantly.
                                if let Some(zenith) = observer_zenith {
                                    let mut invalid = false;
                                    for i in 0..4 {
                                        let star = scratch.sp_catalog_pattern_vectors_sorted[i];
                                        let sin_alt = star[0] * zenith[0]
                                            + star[1] * zenith[1]
                                            + star[2] * zenith[2];
                                        if sin_alt < 0.0 {
                                            invalid = true;
                                            break;
                                        }
                                    }
                                    if invalid {
                                        continue;
                                    }
                                }

                                let (rotation_matrix, _det) =
                                    match find_rotation_matrix_and_det_inplace(
                                        &scratch.sp_image_pattern_vectors_sorted,
                                        &scratch.sp_catalog_pattern_vectors_sorted,
                                        4,
                                    ) {
                                        Some(res) => res,
                                        None => continue,
                                    };

                                // OPTIMIZATION: Early Rejection SVD Pre-pass
                                // EARLY REJECTION OF FALSE POSITIVES
                                // SVD will find a rotation matrix even if the geometric shape of the 4 stars is totally wrong.
                                // If the rotated catalog stars do not closely align with the image stars, it is a false positive.
                                let mut valid_shape = true;
                                let r = &rotation_matrix;
                                // options.match_radius is a fraction of the image width.
                                // Multiply by fov to get approximate max error in radians. Use generous 2.0 multiplier.
                                let max_dist_sq = (options.match_radius * fov * 2.0).powi(2);

                                for i in 0..4 {
                                    let vec = scratch.sp_catalog_pattern_vectors_sorted[i];
                                    let img_vec = scratch.sp_image_pattern_vectors_sorted[i];

                                    let v0 = r[(0, 0)] * vec[0]
                                        + r[(0, 1)] * vec[1]
                                        + r[(0, 2)] * vec[2];
                                    let v1 = r[(1, 0)] * vec[0]
                                        + r[(1, 1)] * vec[1]
                                        + r[(1, 2)] * vec[2];
                                    let v2 = r[(2, 0)] * vec[0]
                                        + r[(2, 1)] * vec[1]
                                        + r[(2, 2)] * vec[2];

                                    let dist_sq = (v0 - img_vec[0]).powi(2)
                                        + (v1 - img_vec[1]).powi(2)
                                        + (v2 - img_vec[2]).powi(2);
                                    if dist_sq > max_dist_sq {
                                        valid_shape = false;
                                        break;
                                    }
                                }

                                if !valid_shape {
                                    continue;
                                }

                                match verify_and_build_solution(
                                    scratch,
                                    star_kd_tree,
                                    star_vectors,
                                    star_metadata,
                                    &self.star_catalog_ids,
                                    &self.db_props,
                                    self.num_patterns,
                                    &rotation_matrix,
                                    fov,
                                    height,
                                    width,
                                    &options,
                                    num_extracted_stars,
                                    match_threshold,
                                    best_prob_mismatch,
                                    t0_solve,
                                ) {
                                    VerificationResult::Success(solution) => return solution,
                                    VerificationResult::NewBestFailed(prob, candidate) => {
                                        // Candidate verified geometrically and produced a new globally
                                        // lowest false-positive probability among threshold-failing patterns.
                                        best_prob_mismatch = prob;
                                        best_candidate_solution = Some(candidate);
                                    }
                                    VerificationResult::None => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no candidate met the strict match_threshold, check if the client requested
        // the best fallback candidate that failed the threshold.
        if options.return_best_failed_match {
            if let Some(candidate) = best_candidate_solution {
                return candidate;
            }
        }

        Solution {
            status: SolveStatus::NoMatch,
            t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {}
}
