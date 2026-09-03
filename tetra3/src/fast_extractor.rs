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
//! High-performance, zero-allocation star centroid extraction engine.
//!
//! Designed for real-time stellar navigation on embedded microprocessors (e.g. Raspberry Pi
//! or Rockchip SoC). Maintains fixed-resolution pre-allocated scratch buffers to eliminate heap
//! allocations during continuous operation. Includes custom background subtraction algorithms
//! ([`FastBgSubMode::BlockMedian`] and [`FastBgSubMode::LineMedian`]).

use ndarray::{ArrayBase, Data, Ix2};
use std::cmp::Ordering;

/// Hardware-friendly integer downsampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastDownsample {
    /// Native sensor resolution (1x).
    None,
    /// 2x2 integer pixel binning.
    X2,
    /// 4x4 integer pixel binning.
    X4,
}

impl FastDownsample {
    /// Returns the integer scale divisor (1, 2, or 4).
    pub fn factor(&self) -> usize {
        match self {
            FastDownsample::None => 1,
            FastDownsample::X2 => 2,
            FastDownsample::X4 => 4,
        }
    }
}

/// Fast background subtraction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastBgSubMode {
    /// Full-frame scalar median subtraction.
    GlobalMedian,
    /// Full-frame scalar mean subtraction (fastest).
    GlobalMean,
    /// Tiled block-median subtraction with bilinear interpolation, providing localized
    /// background removal without sliding-window computational cost.
    BlockMedian {
        /// Side length of square tiles in pixels (e.g. 32 or 64).
        block_size: usize,
    },
    /// Per-row line median subtraction, specially designed to remove sensor horizontal banding noise.
    LineMedian,
}

/// Statistical noise floor estimation mode for the fast pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastSigmaMode {
    /// Full-frame median absolute deviation (1.4826 * MAD).
    GlobalMedianAbs,
    /// Full-frame root-mean-square noise estimation.
    GlobalRootSquare,
}

/// Configuration options for the [`FastExtractor`] engine.
#[derive(Debug, Clone, PartialEq)]
pub struct FastExtractOptions {
    /// Multiplier on the estimated noise standard deviation to determine detection threshold.
    pub sigma: f32,
    /// Hardware downsampling factor applied to input images.
    pub downsample: FastDownsample,
    /// Selected background subtraction algorithm, or `None` to skip subtraction.
    pub bg_sub_mode: Option<FastBgSubMode>,
    /// Statistical method used to estimate the noise floor.
    pub sigma_mode: FastSigmaMode,
    /// If true, applies morphological opening to eliminate isolated single-pixel hot pixels.
    pub binary_open: bool,
    /// Window size in pixels around detection peaks for moment calculation.
    pub centroid_window: Option<usize>,
    /// Maximum connected component area in pixels (rejects bloated artifacts).
    pub max_area: Option<usize>,
    /// Minimum connected component area in pixels (rejects spurious single-pixel noise).
    pub min_area: Option<usize>,
    /// Maximum integrated flux / pixel sum for valid candidate stars.
    pub max_sum: Option<f64>,
    /// Minimum integrated flux / pixel sum required for a candidate star.
    pub min_sum: Option<f64>,
    /// Maximum major-to-minor axis ratio to filter out satellite tracks or streaks.
    pub max_axis_ratio: Option<f64>,
    /// Optional fixed crop `(width, height)` centered on the optical axis.
    pub crop: Option<(usize, usize)>,
    /// Optional virtual crop sub-regions evaluated concurrently without re-running extraction.
    pub virtual_crops: Option<Vec<crate::extractor::Crop>>,
    /// If true, uses fast histogram/sample approximation for background calculations.
    pub approximate_background: bool,
}

/// Runtime overrides applied to [`FastExtractor`] for multi-variant passes.
#[derive(Debug, Clone, Default)]
pub struct FastExtractOptionsUpdate {
    /// Updated sigma detection threshold.
    pub sigma: Option<f32>,
    /// Updated noise filter (binary opening) flag.
    pub noise_filter: Option<bool>,
    /// Updated minimum component area in pixels.
    pub min_area: Option<usize>,
    /// Updated maximum component area in pixels.
    pub max_area: Option<usize>,
    /// Updated virtual crop regions.
    pub virtual_crops: Option<Option<Vec<crate::extractor::Crop>>>,
}

impl Default for FastExtractOptions {
    fn default() -> Self {
        FastExtractOptions {
            sigma: 2.0,
            downsample: FastDownsample::None,
            bg_sub_mode: Some(FastBgSubMode::GlobalMean),
            sigma_mode: FastSigmaMode::GlobalRootSquare,
            binary_open: true,
            centroid_window: None,
            max_area: Some(100),
            min_area: Some(5),
            max_sum: None,
            min_sum: None,
            max_axis_ratio: None,
            crop: None,
            virtual_crops: None,
            approximate_background: false,
        }
    }
}

/// Star centroid properties computed by [`FastExtractor`].
#[derive(Debug, Clone)]
pub struct FastCentroidResult {
    /// Sub-pixel vertical coordinate (row index, 0 at top).
    pub y: f64,
    /// Sub-pixel horizontal coordinate (column index, 0 at left).
    pub x: f64,
    /// Total integrated flux / pixel intensity.
    pub sum: f64,
    /// Connected component pixel area.
    pub area: usize,
    /// Major axis over minor axis ratio measuring blob roundness.
    pub axis_ratio: f64,
}

/// Represents the results of a fast extraction pass.
#[derive(Debug, Clone, Default)]
pub struct FastExtractionResult {
    /// The primary list of extracted centroids.
    pub centroids: Vec<FastCentroidResult>,
    /// Centroids filtered by virtual crop regions, if requested.
    pub virtual_crop_centroids: Option<Vec<Vec<FastCentroidResult>>>,
    /// The normalized (0-255) global background brightness of the image.
    pub background_level: f32,
}

/// FastExtractor maintains pre-allocated global buffers to eliminate OS memory allocations
/// during continuous execution. It utilizes a highly specialized integer-only pipeline
/// optimized specifically for zero-copy u8 inputs and integer downsampling.
pub struct FastExtractor {
    orig_width: usize,
    orig_height: usize,
    // -------------------------------------------------------------------------
    // Dimensions & State
    // -------------------------------------------------------------------------
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    options: FastExtractOptions,

    // -------------------------------------------------------------------------
    // Primary Buffers
    // -------------------------------------------------------------------------
    /// Stores the contiguous raw `u8` input if the provided view is strided.
    contiguous_u8: Vec<u8>,
    /// Stores the pooled sum of downsampled `u8` pixels. Uses `u32` to prevent overflow.
    downsampled_u32: Vec<u32>,

    // -------------------------------------------------------------------------
    // Fixed-Point Mathematical Pipelines
    // -------------------------------------------------------------------------
    // We use integers for the main extraction buffers instead of `f32` floats.
    // This halves memory bandwidth on memory-constrained embedded systems (like Raspberry Pi).
    // To preserve fractional precision after background subtraction, we scale the values
    // by 128.0 before storing them as integers, giving us 7 bits of subpixel precision.
    /// Used for 1x resolution. Background-subtracted intensities stored as scaled `i16`.
    image_i16: Vec<i16>,
    /// Used for downsampled data. `i32` used here because downsampled sums can exceed `i16` bounds.
    image_i32: Vec<i32>,

    // -------------------------------------------------------------------------
    // Utility & Scratch Buffers
    // -------------------------------------------------------------------------
    median_scratch_i16: Vec<i16>,
    median_scratch_i32: Vec<i32>,

    // -------------------------------------------------------------------------
    // Look-Up Tables (LUTs) for BlockMedian
    // -------------------------------------------------------------------------
    bg_grid: Vec<f32>,
    bg_gx0: Vec<usize>,
    bg_tx: Vec<f32>,
    bg_gy0: Vec<usize>,
    bg_gy1: Vec<usize>,
    bg_ty: Vec<f32>,

    // Pre-calculated weights for Centroid Window calculation
    cw_wx: Vec<f64>,
    cw_wy: Vec<f64>,
    cw_strides: Vec<usize>,

    // Morphological & Centroiding buffers
    mask: Vec<bool>,
    stack: Vec<usize>,
}

impl FastExtractor {
    /// Returns a reference to the active extraction options.
    pub fn options(&self) -> &FastExtractOptions {
        &self.options
    }

    /// Returns the original image width this extractor was initialized for.
    pub fn orig_width(&self) -> usize {
        self.orig_width
    }

    /// Returns the original image height this extractor was initialized for.
    pub fn orig_height(&self) -> usize {
        self.orig_height
    }

    /// Updates mutable runtime options (such as sigma threshold or virtual crops).
    pub fn update_options(&mut self, update: FastExtractOptionsUpdate) {
        if let Some(sigma) = update.sigma {
            self.options.sigma = sigma;
        }
        if let Some(nf) = update.noise_filter {
            self.options.binary_open = nf;
        }
        if let Some(ma) = update.min_area {
            self.options.min_area = Some(ma);
        }
        if let Some(mxa) = update.max_area {
            self.options.max_area = Some(mxa);
        }
        if let Some(vc) = update.virtual_crops {
            self.options.virtual_crops = vc;
        }
    }

    /// Creates a fully pre-allocated extractor.
    ///
    /// Pre-allocates working buffers and look-up tables based on the expected input dimensions
    /// and options to guarantee zero heap allocations during steady-state processing.
    pub fn new(width: usize, height: usize, options: FastExtractOptions) -> Self {
        let orig_width = width;
        let orig_height = height;
        let (width, height) = match options.crop {
            Some((cw, ch)) => (cw.min(width), ch.min(height)),
            None => (width, height),
        };
        let total_pixels = width * height;
        let ds = options.downsample.factor();
        let out_width = width / ds;
        let out_height = height / ds;
        let out_pixels = out_width * out_height;

        let mut bg_grid = Vec::new();
        let mut bg_gx0 = vec![0; out_width];
        let mut bg_tx = vec![0.0; out_width];
        let mut bg_gy0 = vec![0; out_height];
        let mut bg_gy1 = vec![0; out_height];
        let mut bg_ty = vec![0.0; out_height];

        // Pre-compute bilinear interpolation weights and indices (LUTs) to remove math from hot loops
        if let Some(FastBgSubMode::BlockMedian { block_size }) = options.bg_sub_mode {
            let grid_w = out_width.div_ceil(block_size);
            let grid_h = out_height.div_ceil(block_size);
            bg_grid.resize(grid_w * grid_h, 0.0);

            for x in 0..out_width {
                let cx = (x as f32 / block_size as f32) - 0.5;
                let gx0 = cx.floor().clamp(0.0, grid_w.saturating_sub(1) as f32) as usize;
                bg_gx0[x] = gx0;
                bg_tx[x] = cx - cx.floor();
            }

            for y in 0..out_height {
                let cy = (y as f32 / block_size as f32) - 0.5;
                let gy0 = cy.floor().clamp(0.0, grid_h.saturating_sub(1) as f32) as usize;
                // Pre-multiply by stride to eliminate per-row multiplications in the hot loop
                bg_gy0[y] = gy0 * grid_w;
                bg_gy1[y] = (gy0 + 1).min(grid_h.saturating_sub(1)) * grid_w;
                bg_ty[y] = cy - cy.floor();
            }
        }

        // Pre-calculate weights for Centroid Window calculation
        let mut cw_wx = Vec::new();
        let mut cw_wy = Vec::new();
        let mut cw_strides = Vec::new();

        if let Some(mut window) = options.centroid_window {
            window = window.min(height).min(width);
            cw_wx.reserve(window);
            cw_wy.reserve(window);
            cw_strides.reserve(window);
            for w in 0..window {
                cw_wx.push(w as f64 + 0.5);
                cw_wy.push(w as f64 + 0.5);
                cw_strides.push(w * width);
            }
        }

        Self {
            orig_width,
            orig_height,
            width,
            height,
            out_width,
            out_height,
            options,
            contiguous_u8: vec![0; total_pixels],
            downsampled_u32: vec![0; out_pixels],
            image_i16: vec![0; total_pixels],
            image_i32: vec![0; out_pixels],
            // Pre-allocate exact capacities for median filtering
            median_scratch_i16: Vec::with_capacity(total_pixels),
            median_scratch_i32: Vec::with_capacity(out_pixels),
            bg_grid,
            bg_gx0,
            bg_tx,
            bg_gy0,
            bg_gy1,
            bg_ty,
            cw_wx,
            cw_wy,
            cw_strides,
            mask: vec![false; out_pixels],
            stack: Vec::with_capacity(1024),
        }
    }

    /// Primary fast-path extractor. Dispatches to either the `i16` (1x) or `i32` (downsampled) pipeline.
    /// Extracts star centroids using multiple sequential configurations concurrently.
    ///
    /// The variants bypass the expensive background subtraction phase, reusing the baseline
    /// background matrix for highly efficient multi-pass extractions.
    pub fn extract_variants<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
        variants: &[FastExtractOptionsUpdate],
    ) -> Vec<FastExtractionResult>
    where
        S: Data<Elem = u8>,
    {
        if variants.is_empty() {
            return Vec::new();
        }

        debug_assert_eq!(
            input_image.dim(),
            (self.orig_height, self.orig_width),
            "Input image dimensions must match the initialized FastExtractor dimensions."
        );

        let (crop_y, crop_x) = if let Some(_) = self.options.crop {
            (
                (self.orig_height.saturating_sub(self.height)) / 2,
                (self.orig_width.saturating_sub(self.width)) / 2,
            )
        } else {
            (0, 0)
        };

        if self.options.crop.is_none() && input_image.is_standard_layout() {
            let src_slice = input_image.as_slice().unwrap();
            self.extract_core(src_slice, variants)
        } else {
            let cropped_view = input_image.slice(ndarray::s![
                crop_y..crop_y + self.height,
                crop_x..crop_x + self.width
            ]);
            for (out_row, in_row) in self
                .contiguous_u8
                .chunks_exact_mut(self.width)
                .zip(cropped_view.rows())
            {
                if let Some(slice) = in_row.as_slice() {
                    out_row.copy_from_slice(slice);
                } else {
                    for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
                        *o = i;
                    }
                }
            }
            self.extract_from_internal(variants)
        }
    }

    /// Extracts star centroids from a single `u8` grayscale image.
    pub fn extract<S>(&mut self, input_image: &ArrayBase<S, Ix2>) -> FastExtractionResult
    where
        S: Data<Elem = u8>,
    {
        self.extract_variants(input_image, &[FastExtractOptionsUpdate::default()])
            .remove(0)
    }

    /// Fast version of the extractor for f32 input images using variants.
    /// Performs an extremely fast conversion to u8 internally.
    pub fn extract_variants_f32<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
        variants: &[FastExtractOptionsUpdate],
    ) -> Vec<FastExtractionResult>
    where
        S: Data<Elem = f32>,
    {
        if variants.is_empty() {
            return Vec::new();
        }

        debug_assert_eq!(
            input_image.dim(),
            (self.orig_height, self.orig_width),
            "Input image dimensions must match the initialized FastExtractor dimensions."
        );

        let (crop_y, crop_x) = if let Some(_) = self.options.crop {
            (
                (self.orig_height.saturating_sub(self.height)) / 2,
                (self.orig_width.saturating_sub(self.width)) / 2,
            )
        } else {
            (0, 0)
        };

        let cropped_view = input_image.slice(ndarray::s![
            crop_y..crop_y + self.height,
            crop_x..crop_x + self.width
        ]);

        if let Some(s) = cropped_view.as_slice() {
            for (o, &i) in self.contiguous_u8.iter_mut().zip(s.iter()) {
                *o = i as u8;
            }
        } else {
            for (out_row, in_row) in self
                .contiguous_u8
                .chunks_exact_mut(self.width)
                .zip(cropped_view.rows())
            {
                if let Some(slice) = in_row.as_slice() {
                    for (o, &i) in out_row.iter_mut().zip(slice.iter()) {
                        *o = i as u8;
                    }
                } else {
                    for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
                        *o = i as u8;
                    }
                }
            }
        }

        self.extract_from_internal(variants)
    }

    /// Fast version of the extractor for f32 input images.
    /// Performs an extremely fast sequential conversion to u8 internally.
    pub fn extract_f32<S>(&mut self, input_image: &ArrayBase<S, Ix2>) -> FastExtractionResult
    where
        S: Data<Elem = f32>,
    {
        self.extract_variants_f32(input_image, &[FastExtractOptionsUpdate::default()])
            .remove(0)
    }

    fn apply_virtual_crops(
        &self,
        centroids: Vec<FastCentroidResult>,
        virtual_crops: &Option<Vec<crate::extractor::Crop>>,
        background_level: f32,
    ) -> FastExtractionResult {
        let virtual_crop_centroids = if let Some(crops) = virtual_crops {
            let mut crop_results = Vec::with_capacity(crops.len());
            for crop in crops {
                let (y_min, y_max, x_min, x_max) = crop.bounds(self.orig_width, self.orig_height);

                let filtered: Vec<_> = centroids
                    .iter()
                    .filter(|c| {
                        let abs_y = c.y;
                        let abs_x = c.x;
                        abs_y >= y_min as f64
                            && abs_y < y_max as f64
                            && abs_x >= x_min as f64
                            && abs_x < x_max as f64
                    })
                    .cloned()
                    .collect();

                crop_results.push(filtered);
            }
            Some(crop_results)
        } else {
            None
        };

        FastExtractionResult {
            centroids,
            virtual_crop_centroids,
            background_level,
        }
    }

    fn extract_from_internal(
        &mut self,
        variants: &[FastExtractOptionsUpdate],
    ) -> Vec<FastExtractionResult> {
        let temp_u8 = std::mem::take(&mut self.contiguous_u8);
        let results = self.extract_core(&temp_u8, variants);
        self.contiguous_u8 = temp_u8;
        results
    }

    fn extract_core(
        &mut self,
        src_slice: &[u8],
        variants: &[FastExtractOptionsUpdate],
    ) -> Vec<FastExtractionResult> {
        let ds = self.options.downsample.factor();

        let base_noise = if ds > 1 {
            // =====================================================================================
            // DOWNSAMPLED PATH (Uses `u32` for accumulation, `i32` for processing)
            // =====================================================================================

            // SIMD-friendly unrolled downsampling loops.
            if ds == 2 {
                self.downsampled_u32
                    .chunks_exact_mut(self.out_width)
                    .enumerate()
                    .for_each(|(out_y, row)| {
                        let start_y = out_y * 2;
                        for out_x in 0..self.out_width {
                            let start_x = out_x * 2;
                            unsafe {
                                let r1 = start_y * self.width + start_x;
                                let r2 = (start_y + 1) * self.width + start_x;
                                let sum = *src_slice.get_unchecked(r1) as u32
                                    + *src_slice.get_unchecked(r1 + 1) as u32
                                    + *src_slice.get_unchecked(r2) as u32
                                    + *src_slice.get_unchecked(r2 + 1) as u32;
                                *row.get_unchecked_mut(out_x) = sum;
                            }
                        }
                    });
            } else if ds == 4 {
                self.downsampled_u32
                    .chunks_exact_mut(self.out_width)
                    .enumerate()
                    .for_each(|(out_y, row)| {
                        let start_y = out_y * 4;
                        for out_x in 0..self.out_width {
                            let start_x = out_x * 4;
                            let mut sum = 0u32;
                            unsafe {
                                for dy in 0..4 {
                                    let r = (start_y + dy) * self.width + start_x;
                                    sum += *src_slice.get_unchecked(r) as u32
                                        + *src_slice.get_unchecked(r + 1) as u32
                                        + *src_slice.get_unchecked(r + 2) as u32
                                        + *src_slice.get_unchecked(r + 3) as u32;
                                }
                                *row.get_unchecked_mut(out_x) = sum;
                            }
                        }
                    });
            }

            let sum_sq_global: f64 = if let Some(bg_mode) = self.options.bg_sub_mode {
                match bg_mode {
                    FastBgSubMode::GlobalMean => {
                        // OPTIMIZATION: Calculate sum and sum-of-squares in one pass.
                        let mut sum = 0u64;
                        let mut sum_sq = 0u64;
                        for &v in &self.downsampled_u32 {
                            let v64 = v as u64;
                            sum += v64;
                            sum_sq += v64 * v64;
                        }
                        let n = self.downsampled_u32.len() as f64;
                        let mean = sum as f64 / n;
                        let total_sum_sq =
                            (sum_sq as f64) - 2.0 * mean * (sum as f64) + n * mean * mean;

                        // OPTIMIZATION: Pure integer hot loop for subtraction and scaling.
                        let scaled_mean = (mean * 128.0).round() as i32;
                        for (o, &i) in self.image_i32.iter_mut().zip(self.downsampled_u32.iter()) {
                            *o = (i as i32 * 128) - scaled_mean;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::GlobalMedian => {
                        // OPTIMIZATION: Row-skipping histogram pass is fast for finding the median.
                        let mut hist = [0u32; 4096];
                        let mut count = 0;
                        let step = if self.options.approximate_background {
                            4
                        } else {
                            1
                        };
                        for row in self
                            .downsampled_u32
                            .chunks_exact(self.out_width)
                            .step_by(step)
                        {
                            for &v in row {
                                unsafe {
                                    *hist.get_unchecked_mut(v as usize) += 1;
                                }
                            }
                            count += self.out_width;
                        }
                        let target = count.div_ceil(2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &c) in hist.iter().enumerate() {
                            accum += c;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        // OPTIMIZATION: Pure integer loop accumulates residual sum-of-squares
                        // mathematically on the fly without inner loop floats.
                        let med_i32 = (med * 128.0).round() as i32;
                        let mut sum_i = 0u64;
                        let mut sum_sq_i = 0u64;

                        for (o, &i) in self.image_i32.iter_mut().zip(self.downsampled_u32.iter()) {
                            let iv = i as u64;
                            sum_i += iv;
                            sum_sq_i += iv * iv;
                            *o = (i as i32 * 128) - med_i32;
                        }

                        let med_f64 = med as f64;
                        let n_f64 = self.downsampled_u32.len() as f64;

                        (sum_sq_i as f64) - 2.0 * med_f64 * (sum_i as f64)
                            + n_f64 * med_f64 * med_f64
                    }
                    FastBgSubMode::LineMedian => {
                        let mut hist = [0u32; 4096];
                        let mut total_sum_sq = 0.0;
                        let o_rows = self.image_i32.chunks_exact_mut(self.out_width);
                        let i_rows = self.downsampled_u32.chunks_exact(self.out_width);

                        for (o_row, i_row) in o_rows.zip(i_rows) {
                            hist.fill(0);
                            let mut count = 0;

                            // OPTIMIZATION: Cache-line aligned interleaved sampling.
                            // Reads 64 contiguous pixels, then skips 64. This halves memory
                            // bandwidth while keeping hardware prefetchers perfectly fed.
                            let mut chunks = i_row.chunks_exact(128);
                            for chunk in chunks.by_ref() {
                                for &v in &chunk[0..64] {
                                    unsafe {
                                        *hist.get_unchecked_mut(v as usize) += 1;
                                    }
                                }
                                count += 64;
                            }

                            // Handle the remainder of the row safely
                            let rem = chunks.remainder();
                            let rem_read = rem.len().min(64);
                            if rem_read > 0 {
                                for &v in &rem[0..rem_read] {
                                    unsafe {
                                        *hist.get_unchecked_mut(v as usize) += 1;
                                    }
                                }
                                count += rem_read;
                            }

                            let target = count.div_ceil(2) as u32;
                            let mut accum = 0;
                            let mut med_val = 0u32;
                            for (val, &c) in hist.iter().enumerate() {
                                accum += c;
                                if accum >= target {
                                    med_val = val as u32;
                                    break;
                                }
                            }

                            // OPTIMIZATION: Pure Integer Hot Loop
                            // Eliminates float casting, rounding, and multiplication per pixel.
                            let med_i32 = med_val as i32;
                            let mut sum_i = 0u64;
                            let mut sum_sq_i = 0u64;

                            // Standard zip loop vectorizes natively via LLVM
                            for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                let iv = i as u64;
                                sum_i += iv;
                                sum_sq_i += iv * iv;
                                *o = (i as i32 - med_i32) * 128;
                            }

                            // Calculate mathematically equivalent variance exactly once per row
                            let med_f64 = med_val as f64;
                            let n_f64 = self.out_width as f64;
                            let row_sum_sq = (sum_sq_i as f64) - 2.0 * med_f64 * (sum_i as f64)
                                + n_f64 * med_f64 * med_f64;
                            total_sum_sq += row_sum_sq;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = self.out_width.div_ceil(block_size);
                        let grid_h = self.out_height.div_ceil(block_size);
                        let mut hists = vec![0u32; grid_w * 4096];

                        for gy in 0..grid_h {
                            hists.fill(0);
                            let start_y = gy * block_size;
                            let end_y = (start_y + block_size).min(self.out_height);
                            // OPTIMIZATION: .step_by(4) row skipping reads only 25% of block pixels
                            let step = if self.options.approximate_background {
                                4
                            } else {
                                1
                            };
                            for y in (start_y..end_y).step_by(step) {
                                let row_start = y * self.out_width;
                                let row =
                                    &self.downsampled_u32[row_start..row_start + self.out_width];
                                for gx in 0..grid_w {
                                    let start_x = gx * block_size;
                                    let end_x = (start_x + block_size).min(self.out_width);
                                    let hist_offset = gx * 4096;
                                    for x in start_x..end_x {
                                        unsafe {
                                            let p = *row.get_unchecked(x);
                                            *hists.get_unchecked_mut(hist_offset + p as usize) += 1;
                                        }
                                    }
                                }
                            }
                            let bg_row = &mut self.bg_grid[gy * grid_w..(gy + 1) * grid_w];
                            for gx in 0..grid_w {
                                let hist = &hists[gx * 4096..(gx + 1) * 4096];
                                let start_x = gx * block_size;
                                let end_x = (start_x + block_size).min(self.out_width);
                                let step = if self.options.approximate_background {
                                    4
                                } else {
                                    1
                                };
                                let num_rows = (start_y..end_y).step_by(step).count();
                                let count = num_rows * (end_x - start_x);
                                let target = count.div_ceil(2) as u32;
                                let mut accum = 0;
                                for (val, &c) in hist.iter().enumerate() {
                                    accum += c;
                                    if accum >= target {
                                        bg_row[gx] = val as f32;
                                        break;
                                    }
                                }
                            }
                        }

                        let bg_grid = &self.bg_grid;
                        let bg_gx0 = &self.bg_gx0;
                        let bg_tx = &self.bg_tx;
                        let bg_gy0 = &self.bg_gy0;
                        let bg_gy1 = &self.bg_gy1;
                        let bg_ty = &self.bg_ty;

                        self.image_i32
                            .chunks_exact_mut(self.out_width)
                            .zip(self.downsampled_u32.chunks_exact(self.out_width))
                            .enumerate()
                            .map(|(y, (out_row, src_row))| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y];
                                let row1_start = bg_gy1[y];
                                let ty = bg_ty[y];
                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];
                                let mut v_grid_row = [0.0f32; 1024];
                                let mut d_grid_row = [0.0f32; 1024];
                                let active_grid_w = grid_w.min(1024);
                                for gx in 0..active_grid_w {
                                    let v0 = row_v0[gx];
                                    let v1 = row_v1[gx];
                                    v_grid_row[gx] = v0 + ty * (v1 - v0);
                                }
                                for gx in 0..active_grid_w.saturating_sub(1) {
                                    d_grid_row[gx] = v_grid_row[gx + 1] - v_grid_row[gx];
                                }
                                let mut sum_sq0_int = 0i64;
                                let mut sum_sq1_int = 0i64;
                                let mut sum_sq2_int = 0i64;
                                let mut sum_sq3_int = 0i64;

                                let mut out_chunks = out_row.chunks_exact_mut(4);
                                let mut src_chunks = src_row.chunks_exact(4);
                                let mut gx0_chunks = bg_gx0.chunks_exact(4);
                                let mut tx_chunks = bg_tx.chunks_exact(4);

                                // OPTIMIZATION: Piecewise Constant Interpolation using pure integer math.
                                // Instead of interpolating the exact background for all 4 pixels, we evaluate it once
                                // and apply it to the whole chunk. This eliminates 75% of LUT accesses and math.
                                // Manually unrolled loop with multiple independent accumulators
                                // improves instruction-level throughput.
                                if self.options.approximate_background {
                                    for (((o, s), gx), tx) in out_chunks
                                        .by_ref()
                                        .zip(src_chunks.by_ref())
                                        .zip(gx0_chunks.by_ref())
                                        .zip(tx_chunks.by_ref())
                                    {
                                        unsafe {
                                            let gx_val = *gx.get_unchecked(0);
                                            let tx_val = *tx.get_unchecked(0);
                                            let bg =
                                                v_grid_row[gx_val] + tx_val * d_grid_row[gx_val];
                                            let bg_i32 = (bg * 128.0).round() as i32;

                                            let val0 = (*s.get_unchecked(0) as i32 * 128) - bg_i32;
                                            let val1 = (*s.get_unchecked(1) as i32 * 128) - bg_i32;
                                            let val2 = (*s.get_unchecked(2) as i32 * 128) - bg_i32;
                                            let val3 = (*s.get_unchecked(3) as i32 * 128) - bg_i32;

                                            *o.get_unchecked_mut(0) = val0;
                                            *o.get_unchecked_mut(1) = val1;
                                            *o.get_unchecked_mut(2) = val2;
                                            *o.get_unchecked_mut(3) = val3;

                                            sum_sq0_int += val0 as i64 * val0 as i64;
                                            sum_sq1_int += val1 as i64 * val1 as i64;
                                            sum_sq2_int += val2 as i64 * val2 as i64;
                                            sum_sq3_int += val3 as i64 * val3 as i64;
                                        }
                                    }
                                } else {
                                    for (((o, s), gx), tx) in out_chunks
                                        .by_ref()
                                        .zip(src_chunks.by_ref())
                                        .zip(gx0_chunks.by_ref())
                                        .zip(tx_chunks.by_ref())
                                    {
                                        unsafe {
                                            for i in 0..4 {
                                                let gx_val = *gx.get_unchecked(i);
                                                let tx_val = *tx.get_unchecked(i);
                                                let bg = v_grid_row[gx_val]
                                                    + tx_val * d_grid_row[gx_val];
                                                let bg_i32 = (bg * 128.0).round() as i32;

                                                let val =
                                                    (*s.get_unchecked(i) as i32 * 128) - bg_i32;
                                                *o.get_unchecked_mut(i) = val;

                                                match i {
                                                    0 => sum_sq0_int += val as i64 * val as i64,
                                                    1 => sum_sq1_int += val as i64 * val as i64,
                                                    2 => sum_sq2_int += val as i64 * val as i64,
                                                    3 => sum_sq3_int += val as i64 * val as i64,
                                                    _ => unreachable!(),
                                                }
                                            }
                                        }
                                    }
                                }

                                row_sq_sum +=
                                    (sum_sq0_int + sum_sq1_int + sum_sq2_int + sum_sq3_int) as f64
                                        / 16384.0;

                                let x = out_chunks.into_remainder().len();
                                if x > 0 {
                                    let rem_x = self.out_width - x;
                                    let mut rem_sq_int = 0i64;
                                    for i in 0..x {
                                        unsafe {
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i);
                                            let tx = *bg_tx.get_unchecked(rem_x + i);
                                            let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];
                                            let bg_i32 = (bg_val * 128.0).round() as i32;

                                            let val = (*src_row.get_unchecked(rem_x + i) as i32
                                                * 128)
                                                - bg_i32;
                                            *out_row.get_unchecked_mut(rem_x + i) = val;
                                            rem_sq_int += val as i64 * val as i64;
                                        }
                                    }
                                    row_sq_sum += rem_sq_int as f64 / 16384.0;
                                }
                                row_sq_sum
                            })
                            .sum()
                    }
                }
            } else {
                self.image_i32
                    .iter_mut()
                    .zip(self.downsampled_u32.iter())
                    .map(|(o, &i)| {
                        let val_f32 = i as f32;
                        *o = (val_f32 * 128.0).round() as i32;
                        (val_f32 * val_f32) as f64
                    })
                    .sum()
            };

            // Calculate local noise floor threshold
            let base_noise = match self.options.sigma_mode {
                FastSigmaMode::GlobalRootSquare => {
                    let mean_sq =
                        (sum_sq_global / (self.out_height * self.out_width) as f64) as f32;
                    mean_sq.max(0.0).sqrt()
                }
                FastSigmaMode::GlobalMedianAbs => {
                    self.median_scratch_i32.clear();
                    self.median_scratch_i32
                        .extend(self.image_i32.iter().map(|&v| v.abs()));
                    let mid = self.median_scratch_i32.len() / 2;
                    let (_, &mut median, _) = self.median_scratch_i32.select_nth_unstable(mid);
                    (median as f32 / 128.0) * 1.48
                }
            };
            base_noise
        } else {
            // =====================================================================================
            // 1x RESOLUTION PATH (Uses `i16` for memory efficiency)
            // =====================================================================================

            // Subtract Background & Accumulate Global RMS Variance (Fused pass)
            let sum_sq_global: f64 = if let Some(bg_mode) = self.options.bg_sub_mode {
                match bg_mode {
                    FastBgSubMode::GlobalMean => {
                        // OPTIMIZATION: Calculate sum and sum-of-squares in one pass.
                        let mut sum = 0u64;
                        let mut sum_sq = 0u64;
                        for &v in src_slice {
                            let v64 = v as u64;
                            sum += v64;
                            sum_sq += v64 * v64;
                        }
                        let n = src_slice.len() as f64;
                        let mean = sum as f64 / n;
                        let total_sum_sq =
                            (sum_sq as f64) - 2.0 * mean * (sum as f64) + n * mean * mean;

                        // OPTIMIZATION: Pure integer hot loop for subtraction and scaling.
                        let scaled_mean = (mean * 128.0).round() as i32;
                        for (o, &i) in self.image_i16.iter_mut().zip(src_slice.iter()) {
                            *o = (i as i32 * 128 - scaled_mean) as i16;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::GlobalMedian => {
                        // OPTIMIZATION: Row-skipping histogram pass is fast for finding the median.
                        let mut hist = [0u32; 256];
                        let mut count = 0;
                        let step = if self.options.approximate_background {
                            4
                        } else {
                            1
                        };
                        for row in src_slice.chunks_exact(self.width).step_by(step) {
                            for &v in row {
                                unsafe {
                                    *hist.get_unchecked_mut(v as usize) += 1;
                                }
                            }
                            count += self.width;
                        }
                        let target = count.div_ceil(2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &c) in hist.iter().enumerate() {
                            accum += c;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        // OPTIMIZATION: Pure integer loop accumulates residual sum-of-squares
                        // mathematically on the fly without inner loop floats.
                        let med_i32 = (med * 128.0).round() as i32;
                        let mut sum_i = 0u64;
                        let mut sum_sq_i = 0u64;

                        for (o, &i) in self.image_i16.iter_mut().zip(src_slice.iter()) {
                            let iv = i as u64;
                            sum_i += iv;
                            sum_sq_i += iv * iv;
                            *o = (i as i32 * 128 - med_i32) as i16;
                        }

                        let med_f64 = med as f64;
                        let n_f64 = src_slice.len() as f64;

                        (sum_sq_i as f64) - 2.0 * med_f64 * (sum_i as f64)
                            + n_f64 * med_f64 * med_f64
                    }
                    FastBgSubMode::LineMedian => {
                        let mut hist = [0u32; 256];
                        let mut total_sum_sq = 0.0;
                        let o_rows = self.image_i16.chunks_exact_mut(self.width);
                        let i_rows = src_slice.chunks_exact(self.width);

                        for (o_row, i_row) in o_rows.zip(i_rows) {
                            hist.fill(0);
                            let mut count = 0;

                            // OPTIMIZATION: Cache-line aligned interleaved sampling.
                            // Reads 64 contiguous pixels, then skips 64. This halves memory
                            // bandwidth while keeping hardware prefetchers perfectly fed.
                            let mut chunks = i_row.chunks_exact(128);
                            for chunk in chunks.by_ref() {
                                for &v in &chunk[0..64] {
                                    unsafe {
                                        *hist.get_unchecked_mut(v as usize) += 1;
                                    }
                                }
                                count += 64;
                            }

                            // Handle the remainder of the row safely
                            let rem = chunks.remainder();
                            let rem_read = rem.len().min(64);
                            if rem_read > 0 {
                                for &v in &rem[0..rem_read] {
                                    unsafe {
                                        *hist.get_unchecked_mut(v as usize) += 1;
                                    }
                                }
                                count += rem_read;
                            }

                            let target = count.div_ceil(2) as u32;
                            let mut accum = 0;
                            let mut med_val = 0u32;
                            for (val, &c) in hist.iter().enumerate() {
                                accum += c;
                                if accum >= target {
                                    med_val = val as u32;
                                    break;
                                }
                            }

                            let med_i32 = med_val as i32;
                            let mut sum_i = 0u64;
                            let mut sum_sq_i = 0u64;

                            for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                let iv = i as u64;
                                sum_i += iv;
                                sum_sq_i += iv * iv;
                                *o = ((i as i32 - med_i32) * 128) as i16;
                            }

                            let med_f64 = med_val as f64;
                            let n_f64 = self.width as f64;
                            let row_sum_sq = (sum_sq_i as f64) - 2.0 * med_f64 * (sum_i as f64)
                                + n_f64 * med_f64 * med_f64;
                            total_sum_sq += row_sum_sq;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = self.width.div_ceil(block_size);
                        let grid_h = self.height.div_ceil(block_size);
                        let mut hists = vec![0u32; grid_w * 256];
                        for gy in 0..grid_h {
                            hists.fill(0);
                            let start_y = gy * block_size;
                            let end_y = (start_y + block_size).min(self.height);
                            // OPTIMIZATION: .step_by(4) row skipping reads only 25% of block pixels
                            let step = if self.options.approximate_background {
                                4
                            } else {
                                1
                            };
                            for y in (start_y..end_y).step_by(step) {
                                let row_start = y * self.width;
                                let row = &src_slice[row_start..row_start + self.width];
                                for gx in 0..grid_w {
                                    let start_x = gx * block_size;
                                    let end_x = (start_x + block_size).min(self.width);
                                    let hist_offset = gx * 256;
                                    for x in start_x..end_x {
                                        unsafe {
                                            let p = *row.get_unchecked(x);
                                            *hists.get_unchecked_mut(hist_offset + p as usize) += 1;
                                        }
                                    }
                                }
                            }
                            let bg_row = &mut self.bg_grid[gy * grid_w..(gy + 1) * grid_w];
                            for gx in 0..grid_w {
                                let hist = &hists[gx * 256..(gx + 1) * 256];
                                let start_x = gx * block_size;
                                let end_x = (start_x + block_size).min(self.width);
                                let step = if self.options.approximate_background {
                                    4
                                } else {
                                    1
                                };
                                let num_rows = (start_y..end_y).step_by(step).count();
                                let count = num_rows * (end_x - start_x);
                                let target = count.div_ceil(2) as u32;
                                let mut accum = 0;
                                for (val, &c) in hist.iter().enumerate() {
                                    accum += c;
                                    if accum >= target {
                                        bg_row[gx] = val as f32;
                                        break;
                                    }
                                }
                            }
                        }
                        let bg_grid = &self.bg_grid;
                        let bg_gx0 = &self.bg_gx0;
                        let bg_tx = &self.bg_tx;
                        let bg_gy0 = &self.bg_gy0;
                        let bg_gy1 = &self.bg_gy1;
                        let bg_ty = &self.bg_ty;

                        self.image_i16
                            .chunks_exact_mut(self.width)
                            .zip(src_slice.chunks_exact(self.width))
                            .enumerate()
                            .map(|(y, (out_row, src_row))| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y];
                                let row1_start = bg_gy1[y];
                                let ty = bg_ty[y];
                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];
                                let mut v_grid_row = [0.0f32; 1024];
                                let mut d_grid_row = [0.0f32; 1024];
                                let active_grid_w = grid_w.min(1024);
                                for gx in 0..active_grid_w {
                                    let v0 = row_v0[gx];
                                    let v1 = row_v1[gx];
                                    v_grid_row[gx] = v0 + ty * (v1 - v0);
                                }
                                for gx in 0..active_grid_w.saturating_sub(1) {
                                    d_grid_row[gx] = v_grid_row[gx + 1] - v_grid_row[gx];
                                }
                                let mut sum_sq0_int = 0i64;
                                let mut sum_sq1_int = 0i64;
                                let mut sum_sq2_int = 0i64;
                                let mut sum_sq3_int = 0i64;

                                let mut out_chunks = out_row.chunks_exact_mut(4);
                                let mut src_chunks = src_row.chunks_exact(4);
                                let mut gx0_chunks = bg_gx0.chunks_exact(4);
                                let mut tx_chunks = bg_tx.chunks_exact(4);

                                // OPTIMIZATION: Piecewise Constant Interpolation using pure integer math.
                                // Instead of interpolating the exact background for all 4 pixels, we evaluate it once
                                // and apply it to the whole chunk. This eliminates 75% of LUT accesses and math.
                                // Manually unrolled loop with multiple independent integer accumulators
                                // improves instruction-level throughput.
                                for (((o, s), gx), tx) in out_chunks
                                    .by_ref()
                                    .zip(src_chunks.by_ref())
                                    .zip(gx0_chunks.by_ref())
                                    .zip(tx_chunks.by_ref())
                                {
                                    unsafe {
                                        let gx_val = *gx.get_unchecked(0);
                                        let tx_val = *tx.get_unchecked(0);
                                        let bg = v_grid_row[gx_val] + tx_val * d_grid_row[gx_val];
                                        let bg_i32 = (bg * 128.0).round() as i32;

                                        let val0 = (*s.get_unchecked(0) as i32 * 128) - bg_i32;
                                        let val1 = (*s.get_unchecked(1) as i32 * 128) - bg_i32;
                                        let val2 = (*s.get_unchecked(2) as i32 * 128) - bg_i32;
                                        let val3 = (*s.get_unchecked(3) as i32 * 128) - bg_i32;

                                        *o.get_unchecked_mut(0) = val0 as i16;
                                        *o.get_unchecked_mut(1) = val1 as i16;
                                        *o.get_unchecked_mut(2) = val2 as i16;
                                        *o.get_unchecked_mut(3) = val3 as i16;

                                        sum_sq0_int += val0 as i64 * val0 as i64;
                                        sum_sq1_int += val1 as i64 * val1 as i64;
                                        sum_sq2_int += val2 as i64 * val2 as i64;
                                        sum_sq3_int += val3 as i64 * val3 as i64;
                                    }
                                }

                                row_sq_sum +=
                                    (sum_sq0_int + sum_sq1_int + sum_sq2_int + sum_sq3_int) as f64
                                        / 16384.0;

                                let x = out_chunks.into_remainder().len();
                                if x > 0 {
                                    let rem_x = self.width - x;
                                    let mut rem_sq_int = 0i64;
                                    for i in 0..x {
                                        unsafe {
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i);
                                            let tx = *bg_tx.get_unchecked(rem_x + i);
                                            let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];
                                            let bg_i32 = (bg_val * 128.0).round() as i32;

                                            let val = (*src_row.get_unchecked(rem_x + i) as i32
                                                * 128)
                                                - bg_i32;
                                            *out_row.get_unchecked_mut(rem_x + i) = val as i16;
                                            rem_sq_int += val as i64 * val as i64;
                                        }
                                    }
                                    row_sq_sum += rem_sq_int as f64 / 16384.0;
                                }
                                row_sq_sum
                            })
                            .sum()
                    }
                }
            } else {
                self.image_i16
                    .iter_mut()
                    .zip(src_slice.iter())
                    .map(|(o, &i)| {
                        let val_f32 = i as f32;
                        *o = (val_f32 * 128.0).round() as i16;
                        (val_f32 * val_f32) as f64
                    })
                    .sum()
            };

            // Calculate local noise floor threshold
            let base_noise = match self.options.sigma_mode {
                FastSigmaMode::GlobalRootSquare => {
                    let mean_sq =
                        (sum_sq_global / (self.out_height * self.out_width) as f64) as f32;
                    mean_sq.max(0.0).sqrt()
                }
                FastSigmaMode::GlobalMedianAbs => {
                    self.median_scratch_i16.clear();
                    self.median_scratch_i16
                        .extend(self.image_i16.iter().map(|&v| v.abs()));
                    let mid = self.median_scratch_i16.len() / 2;
                    let (_, &mut median, _) = self.median_scratch_i16.select_nth_unstable(mid);
                    (median as f32 / 128.0) * 1.48
                }
            };
            base_noise
        };

        // Quick AGC background level estimation
        let mut bg_sum = 0u32;
        let mut bg_count = 0;
        for &v in src_slice.iter().step_by(64) {
            bg_sum += v as u32;
            bg_count += 1;
        }
        let background_level = if bg_count > 0 {
            bg_sum as f32 / bg_count as f32
        } else {
            0.0
        };

        let mut results = Vec::with_capacity(variants.len());
        for variant in variants {
            let mut opt = self.options.clone();
            if let Some(s) = variant.sigma {
                opt.sigma = s;
            }
            if let Some(nf) = variant.noise_filter {
                opt.binary_open = nf;
            }
            if let Some(ma) = variant.min_area {
                opt.min_area = Some(ma);
            }
            if let Some(m_ar) = variant.max_area {
                opt.max_area = Some(m_ar);
            }
            if let Some(ref vc) = variant.virtual_crops {
                opt.virtual_crops = vc.clone();
            }

            let threshold_f32 = base_noise * opt.sigma;

            let mut extracted = if ds > 1 {
                let threshold_scaled = (threshold_f32 * 128.0).round() as i32;
                Self::execute_erosion_and_extraction(
                    &self.image_i32,
                    self.out_width,
                    self.out_height,
                    threshold_scaled,
                    ds,
                    &opt,
                    &mut self.mask,
                    &mut self.stack,
                    &self.cw_wx,
                    &self.cw_wy,
                    &self.cw_strides,
                )
            } else {
                let threshold_scaled = (threshold_f32 * 128.0).round() as i16;
                Self::execute_erosion_and_extraction(
                    &self.image_i16,
                    self.out_width,
                    self.out_height,
                    threshold_scaled,
                    1,
                    &opt,
                    &mut self.mask,
                    &mut self.stack,
                    &self.cw_wx,
                    &self.cw_wy,
                    &self.cw_strides,
                )
            };

            if let Some(_) = self.options.crop {
                let offset_x = (self.orig_width.saturating_sub(self.width)) / 2;
                let offset_y = (self.orig_height.saturating_sub(self.height)) / 2;
                for r in &mut extracted {
                    r.x += offset_x as f64;
                    r.y += offset_y as f64;
                }
            }

            let final_result =
                self.apply_virtual_crops(extracted, &opt.virtual_crops, background_level);
            results.push(final_result);
        }

        results
    }

    /// Generics-driven logic executor to seamlessly support `i16` and `i32` fixed point pipelines.
    /// Extracted into an associated function to decouple borrow checker lifetimes from `self`.
    fn execute_erosion_and_extraction<T>(
        img: &[T],
        width: usize,
        height: usize,
        threshold: T,
        ds: usize,
        options: &FastExtractOptions,
        mask: &mut [bool],
        stack: &mut Vec<usize>,
        cw_wx: &[f64],
        cw_wy: &[f64],
        cw_strides: &[usize],
    ) -> Vec<FastCentroidResult>
    where
        T: Copy + PartialOrd + Into<f64>,
    {
        // 1. Fast binary erosion + threshold
        // Rather than thresholds then eroding in two passes, we perform a fused 3x3 cross
        // morphological evaluation directly off the scalar threshold.
        let mut eroded_pixels = Vec::with_capacity(128);
        if options.binary_open {
            for y in 1..height - 1 {
                let row_offset = y * width;
                let p_prev = img[(y - 1) * width..y * width].as_ptr();
                let p_curr = img[y * width..(y + 1) * width].as_ptr();
                let p_next = img[(y + 1) * width..(y + 2) * width].as_ptr();
                for x in 1..width - 1 {
                    unsafe {
                        if *p_curr.add(x) > threshold
                            && *p_curr.add(x - 1) > threshold
                            && *p_curr.add(x + 1) > threshold
                            && *p_prev.add(x) > threshold
                            && *p_next.add(x) > threshold
                        {
                            eroded_pixels.push(row_offset + x);
                        }
                    }
                }
            }
        } else {
            for y in 0..height {
                let row_offset = y * width;
                let r_curr = &img[row_offset..row_offset + width];
                for x in 0..width {
                    if r_curr[x] > threshold {
                        eroded_pixels.push(row_offset + x);
                    }
                }
            }
        }

        mask.fill(false);
        let mask_ptr = mask.as_mut_ptr();
        for &i in &eroded_pixels {
            unsafe {
                *mask_ptr.add(i) = true;
                if options.binary_open {
                    *mask_ptr.add(i - 1) = true;
                    *mask_ptr.add(i + 1) = true;
                    *mask_ptr.add(i - width) = true;
                    *mask_ptr.add(i + width) = true;
                }
            }
        }

        // 2. Flood Fill & Extract
        // Use an internal stack to trace 4-connected components, evaluating moments
        // mathematically via the parallel axis theorem on the fly.
        let mut extracted = Vec::with_capacity(256);
        let min_a = options.min_area.unwrap_or(0);
        let max_a = options.max_area.unwrap_or(usize::MAX);
        let min_s = options.min_sum.unwrap_or(0.0);
        let max_s = options.max_sum.unwrap_or(f64::MAX);
        let max_ar = options.max_axis_ratio.unwrap_or(f64::MAX);

        for &seed in &eroded_pixels {
            if !mask[seed] {
                continue;
            }
            mask[seed] = false;
            let mut area = 1;
            let val = img[seed].into();
            let mut sum = val;
            let sx = (seed % width) as f64;
            let sy = (seed / width) as f64;
            let mut sum_x = sx * val;
            let mut sum_y = sy * val;
            let mut sum_xx = sx * sx * val;
            let mut sum_yy = sy * sy * val;
            let mut sum_xy = sx * sy * val;

            stack.clear();
            stack.push(seed);
            while let Some(idx) = stack.pop() {
                let cy = idx / width;
                let cx = idx % width;
                let mut check_push = |ni: usize, n_cx: usize, n_cy: usize| unsafe {
                    if *mask.get_unchecked(ni) {
                        *mask.get_unchecked_mut(ni) = false;
                        area += 1;
                        let v = (*img.get_unchecked(ni)).into();
                        sum += v;
                        let nx = n_cx as f64;
                        let ny = n_cy as f64;
                        sum_x += nx * v;
                        sum_y += ny * v;
                        sum_xx += nx * nx * v;
                        sum_yy += ny * ny * v;
                        sum_xy += nx * ny * v;
                        stack.push(ni);
                    }
                };
                if cy > 0 {
                    check_push(idx - width, cx, cy - 1);
                }
                if cy + 1 < height {
                    check_push(idx + width, cx, cy + 1);
                }
                if cx > 0 {
                    check_push(idx - 1, cx - 1, cy);
                }
                if cx + 1 < width {
                    check_push(idx + 1, cx + 1, cy);
                }
            }

            let scaled_sum = sum * (1.0 / 128.0);
            if area < min_a
                || area > max_a
                || scaled_sum < min_s
                || scaled_sum > max_s
                || scaled_sum == 0.0
            {
                continue;
            }
            let inv_sum = 1.0 / sum;
            let m1_x = sum_x * inv_sum;
            let m1_y = sum_y * inv_sum;
            let m2_xx = (sum_xx * inv_sum - m1_x * m1_x).max(0.0);
            let m2_yy = (sum_yy * inv_sum - m1_y * m1_y).max(0.0);
            let m2_xy = sum_xy * inv_sum - m1_x * m1_y;
            let diff = m2_xx - m2_yy;
            let root = (diff * diff + 4.0 * m2_xy * m2_xy).sqrt();
            let major = (2.0 * (m2_xx + m2_yy + root)).sqrt();
            let minor = (2.0 * 0f64.max(m2_xx + m2_yy - root)).sqrt();
            let axis_ratio = major / minor.max(1e-9);
            if axis_ratio > max_ar || minor <= 0.0 {
                continue;
            }

            extracted.push(FastCentroidResult {
                y: m1_y + 0.5,
                x: m1_x + 0.5,
                sum: scaled_sum,
                area,
                axis_ratio,
            });
        }

        // 3. Sort
        extracted.sort_unstable_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap_or(Ordering::Equal));

        // 4. Centroid window
        if let Some(mut window) = options.centroid_window {
            window = window.min(height).min(width);
            for centroid in &mut extracted {
                let c_x = centroid.x.floor() as isize;
                let c_y = centroid.y.floor() as isize;
                let o_x =
                    (c_x - (window as isize) / 2).clamp(0, (width - window) as isize) as usize;
                let o_y =
                    (c_y - (window as isize) / 2).clamp(0, (height - window) as isize) as usize;
                let mut img_sum = 0.0;
                let mut sum_xc = 0.0;
                let mut sum_yc = 0.0;
                for wy in 0..window {
                    unsafe {
                        let row_start = o_y * width + *cw_strides.get_unchecked(wy) + o_x;
                        let row_slice = img.get_unchecked(row_start..row_start + window);
                        let wy_f = *cw_wy.get_unchecked(wy);
                        for (wx, &v) in row_slice.iter().enumerate() {
                            let val = v.into();
                            img_sum += val;
                            sum_xc += val * *cw_wx.get_unchecked(wx);
                            sum_yc += val * wy_f;
                        }
                    }
                }
                if img_sum > 0.0 {
                    let inv_img_sum = 1.0 / img_sum;
                    centroid.x = sum_xc * inv_img_sum + o_x as f64;
                    centroid.y = sum_yc * inv_img_sum + o_y as f64;
                }
            }
        }
        if ds > 1 {
            for centroid in &mut extracted {
                centroid.x *= ds as f64;
                centroid.y *= ds as f64;
            }
        }
        extracted
    }
}
