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
    /// Stores the contiguous raw `u8` input if the provided view is strided or cropped.
    contiguous_u8: Vec<u8>,

    // -------------------------------------------------------------------------
    // Fixed-Point Mathematical Pipelines
    // -------------------------------------------------------------------------
    // We use integers for the main extraction buffers instead of `f32` floats.
    // This halves memory bandwidth on memory-constrained embedded systems (like Raspberry Pi).
    // To preserve fractional precision after background subtraction, we scale the values
    // by 128.0 before storing them as integers, giving us 7 bits of subpixel precision.
    /// Used for 1x resolution. Background-subtracted intensities stored as scaled `i16`.
    image_i16: Vec<i16>,
    /// Used for downsampled data. Stores pooled `u8` sums initially, then in-place
    /// background-subtracted scaled `i32` intensities.
    image_i32: Vec<i32>,

    // -------------------------------------------------------------------------
    // Utility & Scratch Buffers
    // -------------------------------------------------------------------------
    median_scratch_i16: Vec<i16>,
    median_scratch_i32: Vec<i32>,

    // -------------------------------------------------------------------------
    // Look-Up Tables (LUTs) for BlockMedian
    // -------------------------------------------------------------------------
    bg_grid: Vec<u16>,
    bg_gx0: Vec<u32>,
    bg_tx: Vec<u16>,
    bg_gy0: Vec<u32>,
    bg_gy1: Vec<u32>,
    bg_ty: Vec<u16>,

    // Pre-calculated weights for Centroid Window calculation
    cw_wx: Vec<f64>,
    cw_wy: Vec<f64>,
    cw_strides: Vec<usize>,

    // Morphological & Centroiding buffers
    block_hists: Vec<u32>,
    candidate_pixels: Vec<u32>,
    eroded_pixels: Vec<u32>,
    mask: Vec<usize>,
    stack: Vec<u32>,
}

const MASK_WORD_BITS: usize = usize::BITS as usize;
const MASK_WORD_SHIFT: u32 = usize::BITS.trailing_zeros();
const MASK_WORD_MASK: usize = MASK_WORD_BITS - 1;

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
        let mut block_hists = Vec::new();
        let mut bg_gx0 = Vec::new();
        let mut bg_tx = Vec::new();
        let mut bg_gy0 = Vec::new();
        let mut bg_gy1 = Vec::new();
        let mut bg_ty = Vec::new();

        let hist_bins = match ds {
            1 => 256,
            2 => 1024,
            _ => 4096,
        };

        // Pre-compute bilinear interpolation weights and indices (LUTs) to remove math from hot loops
        match options.bg_sub_mode {
            Some(FastBgSubMode::BlockMedian { block_size }) => {
                let grid_w = out_width.div_ceil(block_size);
                let grid_h = out_height.div_ceil(block_size);
                bg_grid.resize(grid_w * grid_h, 0);
                block_hists.resize(grid_w * hist_bins, 0);
                bg_gx0.resize(out_width, 0);
                bg_tx.resize(out_width, 0);
                bg_gy0.resize(out_height, 0);
                bg_gy1.resize(out_height, 0);
                bg_ty.resize(out_height, 0);

                for x in 0..out_width {
                    let cx = (x as f32 / block_size as f32) - 0.5;
                    let gx0 = cx.floor().clamp(0.0, grid_w.saturating_sub(1) as f32) as u32;
                    bg_gx0[x] = gx0;
                    bg_tx[x] = ((cx - cx.floor()) * 256.0).round() as u16;
                }

                for y in 0..out_height {
                    let cy = (y as f32 / block_size as f32) - 0.5;
                    let gy0 = cy.floor().clamp(0.0, grid_h.saturating_sub(1) as f32) as usize;
                    // Pre-multiply by stride to eliminate per-row multiplications in the hot loop
                    bg_gy0[y] = (gy0 * grid_w) as u32;
                    bg_gy1[y] = ((gy0 + 1).min(grid_h.saturating_sub(1)) * grid_w) as u32;
                    bg_ty[y] = ((cy - cy.floor()) * 256.0).round() as u16;
                }
            }
            Some(FastBgSubMode::GlobalMedian) | Some(FastBgSubMode::LineMedian) => {
                block_hists.resize(hist_bins, 0);
            }
            _ => {}
        }

        // Pre-calculate weights for Centroid Window calculation
        let mut cw_wx = Vec::new();
        let mut cw_wy = Vec::new();
        let mut cw_strides = Vec::new();

        if let Some(mut window) = options.centroid_window {
            window = window.min(out_height).min(out_width);
            cw_wx.reserve(window);
            cw_wy.reserve(window);
            cw_strides.reserve(window);
            for w in 0..window {
                cw_wx.push(w as f64 + 0.5);
                cw_wy.push(w as f64 + 0.5);
                cw_strides.push(w * out_width);
            }
        }

        let is_downsampled = ds > 1;
        let needs_median_scratch = options.sigma_mode == FastSigmaMode::GlobalMedianAbs;
        let needs_crop_buf = options.crop.is_some();

        Self {
            orig_width,
            orig_height,
            width,
            height,
            out_width,
            out_height,
            options,
            contiguous_u8: if needs_crop_buf {
                vec![0; total_pixels]
            } else {
                Vec::new()
            },
            image_i16: if !is_downsampled {
                vec![0; total_pixels]
            } else {
                Vec::new()
            },
            image_i32: if is_downsampled {
                vec![0; out_pixels]
            } else {
                Vec::new()
            },
            // Pre-allocate exact capacities for median filtering only when GlobalMedianAbs is enabled
            median_scratch_i16: if !is_downsampled && needs_median_scratch {
                Vec::with_capacity(total_pixels)
            } else {
                Vec::new()
            },
            median_scratch_i32: if is_downsampled && needs_median_scratch {
                Vec::with_capacity(out_pixels)
            } else {
                Vec::new()
            },
            bg_grid,
            bg_gx0,
            bg_tx,
            bg_gy0,
            bg_gy1,
            bg_ty,
            cw_wx,
            cw_wy,
            cw_strides,
            block_hists,
            candidate_pixels: Vec::with_capacity(2048),
            eroded_pixels: Vec::with_capacity(512),
            mask: vec![0usize; out_pixels.div_ceil(MASK_WORD_BITS)],
            stack: Vec::with_capacity(1024),
        }
    }

    /// Extracts star centroids from a raw contiguous row-major `u8` grayscale slice using multiple
    /// sequential configurations concurrently, bypassing `ndarray` view construction.
    pub fn extract_variants_slice(
        &mut self,
        input_slice: &[u8],
        variants: &[FastExtractOptionsUpdate],
    ) -> Vec<FastExtractionResult> {
        if variants.is_empty() {
            return Vec::new();
        }

        debug_assert_eq!(
            input_slice.len(),
            self.orig_height * self.orig_width,
            "Input slice length must match the initialized FastExtractor dimensions."
        );

        if self.options.crop.is_none() {
            self.extract_core(input_slice, variants)
        } else {
            let crop_y = (self.orig_height.saturating_sub(self.height)) / 2;
            let crop_x = (self.orig_width.saturating_sub(self.width)) / 2;
            let needed = self.width * self.height;
            if self.contiguous_u8.len() < needed {
                self.contiguous_u8.resize(needed, 0);
            }
            for (y, out_row) in self.contiguous_u8.chunks_exact_mut(self.width).enumerate() {
                let src_row_start = (crop_y + y) * self.orig_width + crop_x;
                out_row.copy_from_slice(&input_slice[src_row_start..src_row_start + self.width]);
            }
            self.extract_from_internal(variants)
        }
    }

    /// Extracts star centroids from a raw contiguous row-major `u8` grayscale slice.
    pub fn extract_slice(&mut self, input_slice: &[u8]) -> FastExtractionResult {
        self.extract_variants_slice(input_slice, &[FastExtractOptionsUpdate::default()])
            .remove(0)
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

        if input_image.is_standard_layout() {
            let src_slice = input_image.as_slice().unwrap();
            return self.extract_variants_slice(src_slice, variants);
        }

        let (crop_y, crop_x) = if let Some(_) = self.options.crop {
            (
                (self.orig_height.saturating_sub(self.height)) / 2,
                (self.orig_width.saturating_sub(self.width)) / 2,
            )
        } else {
            (0, 0)
        };

        let needed = self.width * self.height;
        if self.contiguous_u8.len() < needed {
            self.contiguous_u8.resize(needed, 0);
        }
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

        let needed = self.width * self.height;
        if self.contiguous_u8.len() < needed {
            self.contiguous_u8.resize(needed, 0);
        }

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
            // DOWNSAMPLED PATH (Uses `i32` in-place for accumulation and processing)
            // =====================================================================================

            // SIMD-friendly unrolled downsampling loops.
            if ds == 2 {
                self.image_i32
                    .chunks_exact_mut(self.out_width)
                    .enumerate()
                    .for_each(|(out_y, row)| {
                        let start_y = out_y * 2;
                        for out_x in 0..self.out_width {
                            let start_x = out_x * 2;
                            unsafe {
                                let r1 = start_y * self.width + start_x;
                                let r2 = (start_y + 1) * self.width + start_x;
                                let sum = *src_slice.get_unchecked(r1) as i32
                                    + *src_slice.get_unchecked(r1 + 1) as i32
                                    + *src_slice.get_unchecked(r2) as i32
                                    + *src_slice.get_unchecked(r2 + 1) as i32;
                                *row.get_unchecked_mut(out_x) = sum;
                            }
                        }
                    });
            } else if ds == 4 {
                self.image_i32
                    .chunks_exact_mut(self.out_width)
                    .enumerate()
                    .for_each(|(out_y, row)| {
                        let start_y = out_y * 4;
                        for out_x in 0..self.out_width {
                            let start_x = out_x * 4;
                            let mut sum = 0i32;
                            unsafe {
                                for dy in 0..4 {
                                    let r = (start_y + dy) * self.width + start_x;
                                    sum += *src_slice.get_unchecked(r) as i32
                                        + *src_slice.get_unchecked(r + 1) as i32
                                        + *src_slice.get_unchecked(r + 2) as i32
                                        + *src_slice.get_unchecked(r + 3) as i32;
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
                        for &v in &self.image_i32 {
                            let v64 = v as u64;
                            sum += v64;
                            sum_sq += v64 * v64;
                        }
                        let n = self.image_i32.len() as f64;
                        let mean = sum as f64 / n;
                        let total_sum_sq =
                            (sum_sq as f64) - 2.0 * mean * (sum as f64) + n * mean * mean;

                        // OPTIMIZATION: Pure integer hot loop for in-place subtraction and scaling.
                        let scaled_mean = (mean * 128.0).round() as i32;
                        for o in self.image_i32.iter_mut() {
                            *o = (*o * 128) - scaled_mean;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::GlobalMedian => {
                        // OPTIMIZATION: Row-skipping histogram pass is fast for finding the median.
                        let hist_bins = if ds == 2 { 1024 } else { 4096 };
                        let hist = &mut self.block_hists[..hist_bins];
                        hist.fill(0);
                        let mut count = 0;
                        let step = if self.options.approximate_background {
                            4
                        } else {
                            1
                        };
                        for row in self.image_i32.chunks_exact(self.out_width).step_by(step) {
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

                        for o in self.image_i32.iter_mut() {
                            let iv = *o as u64;
                            sum_i += iv;
                            sum_sq_i += iv * iv;
                            *o = (*o * 128) - med_i32;
                        }

                        let med_f64 = med as f64;
                        let n_f64 = self.image_i32.len() as f64;

                        (sum_sq_i as f64) - 2.0 * med_f64 * (sum_i as f64)
                            + n_f64 * med_f64 * med_f64
                    }
                    FastBgSubMode::LineMedian => {
                        let hist_bins = if ds == 2 { 1024 } else { 4096 };
                        let hist = &mut self.block_hists[..hist_bins];
                        let mut total_sum_sq = 0.0;

                        for row in self.image_i32.chunks_exact_mut(self.out_width) {
                            hist.fill(0);
                            let mut count = 0;

                            // OPTIMIZATION: Cache-line aligned interleaved sampling.
                            // Reads 64 contiguous pixels, then skips 64. This halves memory
                            // bandwidth while keeping hardware prefetchers perfectly fed.
                            let mut chunks = row.chunks_exact(128);
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

                            // Standard in-place loop vectorizes natively via LLVM
                            for o in row.iter_mut() {
                                let iv = *o as u64;
                                sum_i += iv;
                                sum_sq_i += iv * iv;
                                *o = (*o - med_i32) * 128;
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
                        let hist_bins = if ds == 2 { 1024 } else { 4096 };
                        let hists = &mut self.block_hists[..grid_w * hist_bins];

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
                                let row = &self.image_i32[row_start..row_start + self.out_width];
                                for gx in 0..grid_w {
                                    let start_x = gx * block_size;
                                    let end_x = (start_x + block_size).min(self.out_width);
                                    let hist_offset = gx * hist_bins;
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
                                let hist = &hists[gx * hist_bins..(gx + 1) * hist_bins];
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
                                        bg_row[gx] = val as u16;
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
                        let mut v_grid_row = [0i32; 256];
                        let mut d_grid_row = [0i32; 256];
                        let active_grid_w = grid_w.min(256);

                        self.image_i32
                            .chunks_exact_mut(self.out_width)
                            .enumerate()
                            .map(|(y, out_row)| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y] as usize;
                                let row1_start = bg_gy1[y] as usize;
                                let ty_q8 = bg_ty[y] as i32;
                                let inv_ty_q8 = 256 - ty_q8;
                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];
                                for gx in 0..active_grid_w {
                                    let v0 = row_v0[gx] as i32;
                                    let v1 = row_v1[gx] as i32;
                                    v_grid_row[gx] = v0 * inv_ty_q8 + v1 * ty_q8;
                                }
                                for gx in 0..active_grid_w.saturating_sub(1) {
                                    d_grid_row[gx] = v_grid_row[gx + 1] - v_grid_row[gx];
                                }
                                let mut sum_sq0_int = 0i64;
                                let mut sum_sq1_int = 0i64;
                                let mut sum_sq2_int = 0i64;
                                let mut sum_sq3_int = 0i64;

                                let mut out_chunks = out_row.chunks_exact_mut(4);
                                let mut gx0_chunks = bg_gx0.chunks_exact(4);
                                let mut tx_chunks = bg_tx.chunks_exact(4);

                                // OPTIMIZATION: Piecewise Constant Interpolation using pure integer math.
                                // Instead of interpolating the exact background for all 4 pixels, we evaluate it once
                                // and apply it to the whole chunk. This eliminates 75% of LUT accesses and math.
                                // Manually unrolled loop with multiple independent accumulators
                                // improves instruction-level throughput.
                                if self.options.approximate_background {
                                    for ((o, gx), tx) in out_chunks
                                        .by_ref()
                                        .zip(gx0_chunks.by_ref())
                                        .zip(tx_chunks.by_ref())
                                    {
                                        unsafe {
                                            let gx_val = *gx.get_unchecked(0) as usize;
                                            let tx_q8 = *tx.get_unchecked(0) as i32;
                                            let bg_q16 = (v_grid_row[gx_val] << 8)
                                                + tx_q8 * d_grid_row[gx_val];
                                            let bg_i32 = (bg_q16 + 256) >> 9;

                                            let val0 = (*o.get_unchecked(0) * 128) - bg_i32;
                                            let val1 = (*o.get_unchecked(1) * 128) - bg_i32;
                                            let val2 = (*o.get_unchecked(2) * 128) - bg_i32;
                                            let val3 = (*o.get_unchecked(3) * 128) - bg_i32;

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
                                    for ((o, gx), tx) in out_chunks
                                        .by_ref()
                                        .zip(gx0_chunks.by_ref())
                                        .zip(tx_chunks.by_ref())
                                    {
                                        unsafe {
                                            for i in 0..4 {
                                                let gx_val = *gx.get_unchecked(i) as usize;
                                                let tx_q8 = *tx.get_unchecked(i) as i32;
                                                let bg_q16 = (v_grid_row[gx_val] << 8)
                                                    + tx_q8 * d_grid_row[gx_val];
                                                let bg_i32 = (bg_q16 + 256) >> 9;

                                                let val = (*o.get_unchecked(i) * 128) - bg_i32;
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
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i) as usize;
                                            let tx_q8 = *bg_tx.get_unchecked(rem_x + i) as i32;
                                            let bg_q16 =
                                                (v_grid_row[gx0] << 8) + tx_q8 * d_grid_row[gx0];
                                            let bg_i32 = (bg_q16 + 256) >> 9;

                                            let val =
                                                (*out_row.get_unchecked(rem_x + i) * 128) - bg_i32;
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
                    .map(|o| {
                        let val = *o;
                        *o = val * 128;
                        (val as i64 * val as i64) as f64
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
                        let hist = &mut self.block_hists[..256];
                        hist.fill(0);
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
                        let hist = &mut self.block_hists[..256];
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
                        let hists = &mut self.block_hists[..grid_w * 256];
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
                                        bg_row[gx] = val as u16;
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
                        let mut v_grid_row = [0i32; 256];
                        let mut d_grid_row = [0i32; 256];
                        let active_grid_w = grid_w.min(256);

                        self.image_i16
                            .chunks_exact_mut(self.width)
                            .zip(src_slice.chunks_exact(self.width))
                            .enumerate()
                            .map(|(y, (out_row, src_row))| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y] as usize;
                                let row1_start = bg_gy1[y] as usize;
                                let ty_q8 = bg_ty[y] as i32;
                                let inv_ty_q8 = 256 - ty_q8;
                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];
                                for gx in 0..active_grid_w {
                                    let v0 = row_v0[gx] as i32;
                                    let v1 = row_v1[gx] as i32;
                                    v_grid_row[gx] = v0 * inv_ty_q8 + v1 * ty_q8;
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
                                        let gx_val = *gx.get_unchecked(0) as usize;
                                        let tx_q8 = *tx.get_unchecked(0) as i32;
                                        let bg_q16 =
                                            (v_grid_row[gx_val] << 8) + tx_q8 * d_grid_row[gx_val];
                                        let bg_i32 = (bg_q16 + 256) >> 9;

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
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i) as usize;
                                            let tx_q8 = *bg_tx.get_unchecked(rem_x + i) as i32;
                                            let bg_q16 =
                                                (v_grid_row[gx0] << 8) + tx_q8 * d_grid_row[gx0];
                                            let bg_i32 = (bg_q16 + 256) >> 9;

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
                        let val = i as i16;
                        *o = val * 128;
                        (i as u32 * i as u32) as f64
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

        // Sort variant indices by ascending sigma so the most permissive variant (lowest threshold)
        // always runs first, narrowing candidate_pixels in-place for stricter variants while
        // preserving the caller's original output order.
        let mut variant_order: Vec<(usize, f32)> = variants
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, v.sigma.unwrap_or(self.options.sigma)))
            .collect();
        variant_order.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut results: Vec<FastExtractionResult> = (0..variants.len())
            .map(|_| FastExtractionResult::default())
            .collect();

        for (step_idx, &(orig_idx, sigma)) in variant_order.iter().enumerate() {
            let variant = &variants[orig_idx];
            let is_first_variant = step_idx == 0;

            let opt = FastExtractOptions {
                sigma,
                downsample: self.options.downsample,
                bg_sub_mode: self.options.bg_sub_mode,
                sigma_mode: self.options.sigma_mode,
                binary_open: variant.noise_filter.unwrap_or(self.options.binary_open),
                centroid_window: self.options.centroid_window,
                max_area: variant.max_area.or(self.options.max_area),
                min_area: variant.min_area.or(self.options.min_area),
                max_sum: self.options.max_sum,
                min_sum: self.options.min_sum,
                max_axis_ratio: self.options.max_axis_ratio,
                crop: self.options.crop,
                virtual_crops: None,
                approximate_background: self.options.approximate_background,
            };
            let active_vc = variant
                .virtual_crops
                .as_ref()
                .unwrap_or(&self.options.virtual_crops);

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
                    is_first_variant,
                    &mut self.candidate_pixels,
                    &mut self.eroded_pixels,
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
                    is_first_variant,
                    &mut self.candidate_pixels,
                    &mut self.eroded_pixels,
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

            results[orig_idx] = self.apply_virtual_crops(extracted, active_vc, background_level);
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
        is_first_variant: bool,
        candidate_pixels: &mut Vec<u32>,
        eroded_pixels: &mut Vec<u32>,
        mask: &mut [usize],
        stack: &mut Vec<u32>,
        cw_wx: &[f64],
        cw_wy: &[f64],
        cw_strides: &[usize],
    ) -> Vec<FastCentroidResult>
    where
        T: Copy + PartialOrd + Into<i64>,
    {
        // 1. Fast threshold + optional binary erosion
        // The first (most permissive) variant scans `img` once; subsequent stricter variants
        // narrow `candidate_pixels` in-place without re-scanning the image.
        if is_first_variant {
            candidate_pixels.clear();
            for (idx, &v) in img.iter().enumerate() {
                if v > threshold {
                    candidate_pixels.push(idx as u32);
                }
            }
        } else {
            candidate_pixels.retain(|&idx| unsafe { *img.get_unchecked(idx as usize) > threshold });
        }

        let active_seeds: &[u32] = if options.binary_open {
            eroded_pixels.clear();
            for &idx in candidate_pixels.iter() {
                let idx_u = idx as usize;
                let y = idx_u / width;
                let x = idx_u % width;
                if y > 0 && y + 1 < height && x > 0 && x + 1 < width {
                    unsafe {
                        if *img.get_unchecked(idx_u - 1) > threshold
                            && *img.get_unchecked(idx_u + 1) > threshold
                            && *img.get_unchecked(idx_u - width) > threshold
                            && *img.get_unchecked(idx_u + width) > threshold
                        {
                            eroded_pixels.push(idx);
                        }
                    }
                }
            }
            eroded_pixels.as_slice()
        } else {
            candidate_pixels.as_slice()
        };

        debug_assert!(mask.iter().all(|&w| w == 0));
        let set_bit = |m: &mut [usize], i: usize| unsafe {
            *m.get_unchecked_mut(i >> MASK_WORD_SHIFT) |= 1usize << (i & MASK_WORD_MASK);
        };
        for &i_u32 in active_seeds {
            let i = i_u32 as usize;
            set_bit(mask, i);
            if options.binary_open {
                set_bit(mask, i - 1);
                set_bit(mask, i + 1);
                set_bit(mask, i - width);
                set_bit(mask, i + width);
            }
        }

        // 2. Flood Fill & Extract
        // Use an internal stack to trace 4-connected components, evaluating moments
        // mathematically via relative integer offsets from the seed pixel.
        let mut extracted = Vec::with_capacity(256);
        let min_a = options.min_area.unwrap_or(0);
        let max_a = options.max_area.unwrap_or(usize::MAX);
        let min_s = options.min_sum.unwrap_or(0.0);
        let max_s = options.max_sum.unwrap_or(f64::MAX);
        let max_ar = options.max_axis_ratio.unwrap_or(f64::MAX);

        for &seed_u32 in active_seeds {
            let seed = seed_u32 as usize;
            let seed_word = unsafe { mask.get_unchecked_mut(seed >> MASK_WORD_SHIFT) };
            let seed_bit = 1usize << (seed & MASK_WORD_MASK);
            if (*seed_word & seed_bit) == 0 {
                continue;
            }
            *seed_word &= !seed_bit;
            let mut area = 1;
            let val: i64 = unsafe { *img.get_unchecked(seed) }.into();
            let mut sum_val: i64 = val;
            let sx0 = (seed % width) as i32;
            let sy0 = (seed / width) as i32;
            let mut sum_dx: i64 = 0;
            let mut sum_dy: i64 = 0;
            let mut sum_dxx: i64 = 0;
            let mut sum_dyy: i64 = 0;
            let mut sum_dxy: i64 = 0;

            stack.clear();
            stack.push(seed_u32);
            while let Some(idx_u32) = stack.pop() {
                let idx = idx_u32 as usize;
                let cy = idx / width;
                let cx = idx % width;
                let mut check_push = |ni: usize, n_cx: usize, n_cy: usize| unsafe {
                    let w = mask.get_unchecked_mut(ni >> MASK_WORD_SHIFT);
                    let b = 1usize << (ni & MASK_WORD_MASK);
                    if (*w & b) != 0 {
                        *w &= !b;
                        area += 1;
                        let v: i64 = (*img.get_unchecked(ni)).into();
                        sum_val += v;
                        let dx = n_cx as i32 - sx0;
                        let dy = n_cy as i32 - sy0;
                        sum_dx += dx as i64 * v;
                        sum_dy += dy as i64 * v;
                        sum_dxx += (dx * dx) as i64 * v;
                        sum_dyy += (dy * dy) as i64 * v;
                        sum_dxy += (dx * dy) as i64 * v;
                        stack.push(ni as u32);
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

            let scaled_sum = (sum_val as f64) * (1.0 / 128.0);
            if area < min_a
                || area > max_a
                || scaled_sum < min_s
                || scaled_sum > max_s
                || sum_val <= 0
            {
                continue;
            }
            let inv_sum = 1.0f32 / (sum_val as f32);
            let m1_dx = (sum_dx as f32) * inv_sum;
            let m1_dy = (sum_dy as f32) * inv_sum;
            let m2_xx = ((sum_dxx as f32) * inv_sum - m1_dx * m1_dx).max(0.0);
            let m2_yy = ((sum_dyy as f32) * inv_sum - m1_dy * m1_dy).max(0.0);
            let m2_xy = (sum_dxy as f32) * inv_sum - m1_dx * m1_dy;
            let diff = m2_xx - m2_yy;
            let root = (diff * diff + 4.0 * m2_xy * m2_xy).sqrt();
            let major = (2.0 * (m2_xx + m2_yy + root)).sqrt();
            let minor = (2.0 * 0.0f32.max(m2_xx + m2_yy - root)).sqrt();
            let axis_ratio = (major / minor.max(1e-9)) as f64;
            if axis_ratio > max_ar || minor <= 0.0 {
                continue;
            }

            extracted.push(FastCentroidResult {
                y: (sy0 as f64) + (m1_dy as f64) + 0.5,
                x: (sx0 as f64) + (m1_dx as f64) + 0.5,
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
                            let val = v.into() as f64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_multi_variant_equivalence_and_cw_downsample() {
        let w = 640;
        let h = 480;
        let mut frame = vec![18u8; w * h];

        // Add synthetic stars including one near the bottom edge to exercise cw_strides with X2 downsample
        let stars = [
            (120usize, 80usize, 220u8),
            (310, 210, 140),
            (500, 350, 90),
            (250, 474, 200),
        ];
        for &(sx, sy, peak) in &stars {
            for dy in -3isize..=3 {
                for dx in -3isize..=3 {
                    let dist2 = (dx * dx + dy * dy) as u8;
                    let val = peak.saturating_sub(dist2.saturating_mul(18));
                    let x = (sx as isize + dx) as usize;
                    let y = (sy as isize + dy) as usize;
                    frame[y * w + x] = frame[y * w + x].max(val);
                }
            }
        }
        let img = Array2::from_shape_vec((h, w), frame).unwrap();

        for ds in [FastDownsample::None, FastDownsample::X2, FastDownsample::X4] {
            for bg_sub_mode in [
                Some(FastBgSubMode::BlockMedian { block_size: 32 }),
                Some(FastBgSubMode::LineMedian),
                Some(FastBgSubMode::GlobalMedian),
            ] {
                for cw in [None, Some(5)] {
                    let base_opts = FastExtractOptions {
                        sigma: 3.5,
                        downsample: ds,
                        bg_sub_mode,
                        sigma_mode: FastSigmaMode::GlobalMedianAbs,
                        centroid_window: cw,
                        approximate_background: true,
                        ..Default::default()
                    };

                    // Pass strict variant first (5.0), then permissive (2.5), then intermediate (3.5)
                    let variants = [
                        FastExtractOptionsUpdate {
                            sigma: Some(5.0),
                            min_area: Some(1),
                            max_area: Some(100),
                            ..Default::default()
                        },
                        FastExtractOptionsUpdate {
                            sigma: Some(2.5),
                            min_area: Some(2),
                            max_area: Some(100),
                            ..Default::default()
                        },
                        FastExtractOptionsUpdate {
                            sigma: Some(3.5),
                            min_area: Some(1),
                            max_area: Some(80),
                            ..Default::default()
                        },
                    ];

                    let mut multi_ext = FastExtractor::new(w, h, base_opts.clone());
                    let multi_res = multi_ext.extract_variants(&img, &variants);
                    assert_eq!(multi_res.len(), variants.len());

                    for (v_idx, var) in variants.iter().enumerate() {
                        let mut single_ext = FastExtractor::new(w, h, base_opts.clone());
                        single_ext.update_options(var.clone());
                        let single_res = single_ext.extract(&img);

                        assert_eq!(
                            multi_res[v_idx].background_level.to_bits(),
                            single_res.background_level.to_bits()
                        );
                        assert_eq!(multi_res[v_idx].centroids.len(), single_res.centroids.len());
                        for (mc, sc) in multi_res[v_idx]
                            .centroids
                            .iter()
                            .zip(single_res.centroids.iter())
                        {
                            assert_eq!(mc.x.to_bits(), sc.x.to_bits());
                            assert_eq!(mc.y.to_bits(), sc.y.to_bits());
                            assert_eq!(mc.sum.to_bits(), sc.sum.to_bits());
                            assert_eq!(mc.area, sc.area);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_conditional_buffer_allocations() {
        let w = 1280;
        let h = 720;

        // 1. 1x + GlobalRootSquare + BlockMedian (no crop)
        let ext_1x_rms = FastExtractor::new(
            w,
            h,
            FastExtractOptions {
                bg_sub_mode: Some(FastBgSubMode::BlockMedian { block_size: 64 }),
                ..Default::default()
            },
        );
        assert_eq!(ext_1x_rms.image_i16.len(), w * h);
        assert_eq!(ext_1x_rms.image_i32.capacity(), 0);
        assert_eq!(ext_1x_rms.median_scratch_i16.capacity(), 0);
        assert_eq!(ext_1x_rms.median_scratch_i32.capacity(), 0);
        assert_eq!(ext_1x_rms.contiguous_u8.capacity(), 0);
        assert_eq!(ext_1x_rms.mask.len(), (w * h).div_ceil(MASK_WORD_BITS));
        assert!(!ext_1x_rms.bg_gx0.is_empty());

        // 2. X2 downsample + GlobalRootSquare + LineMedian (no crop)
        let ext_2x_line = FastExtractor::new(
            w,
            h,
            FastExtractOptions {
                downsample: FastDownsample::X2,
                bg_sub_mode: Some(FastBgSubMode::LineMedian),
                sigma_mode: FastSigmaMode::GlobalRootSquare,
                ..Default::default()
            },
        );
        let out_px = (w / 2) * (h / 2);
        assert_eq!(ext_2x_line.image_i16.capacity(), 0);
        assert_eq!(ext_2x_line.image_i32.len(), out_px);
        assert_eq!(ext_2x_line.median_scratch_i16.capacity(), 0);
        assert_eq!(ext_2x_line.median_scratch_i32.capacity(), 0);
        assert_eq!(ext_2x_line.contiguous_u8.capacity(), 0);
        assert_eq!(ext_2x_line.mask.len(), out_px.div_ceil(MASK_WORD_BITS));
        assert_eq!(ext_2x_line.bg_gx0.capacity(), 0);
        assert_eq!(ext_2x_line.bg_tx.capacity(), 0);
        assert_eq!(ext_2x_line.bg_gy0.capacity(), 0);
        assert_eq!(ext_2x_line.bg_gy1.capacity(), 0);
        assert_eq!(ext_2x_line.bg_ty.capacity(), 0);
        assert_eq!(ext_2x_line.bg_grid.capacity(), 0);
        assert_eq!(ext_2x_line.block_hists.len(), 1024);

        // 3. X2 downsample + GlobalMedianAbs + crop
        let mut ext_2x_mad_crop = FastExtractor::new(
            w,
            h,
            FastExtractOptions {
                downsample: FastDownsample::X2,
                sigma_mode: FastSigmaMode::GlobalMedianAbs,
                crop: Some((640, 360)),
                ..Default::default()
            },
        );
        let crop_w = 640;
        let crop_h = 360;
        let crop_out_px = (crop_w / 2) * (crop_h / 2);
        assert_eq!(ext_2x_mad_crop.image_i16.capacity(), 0);
        assert_eq!(ext_2x_mad_crop.image_i32.len(), crop_out_px);
        assert_eq!(ext_2x_mad_crop.median_scratch_i16.capacity(), 0);
        assert_eq!(ext_2x_mad_crop.median_scratch_i32.capacity(), crop_out_px);
        assert_eq!(ext_2x_mad_crop.contiguous_u8.len(), crop_w * crop_h);
        assert_eq!(ext_2x_mad_crop.block_hists.capacity(), 0);

        let img_u8 = Array2::from_elem((h, w), 10u8);
        let _ = ext_2x_mad_crop.extract(&img_u8);
        assert_eq!(ext_2x_mad_crop.median_scratch_i32.len(), crop_out_px);
        assert!(ext_2x_mad_crop.mask.iter().all(|&word| word == 0));

        // 4. Lazy allocation of contiguous_u8 on f32 extraction when uncropped
        let mut ext_lazy = FastExtractor::new(64, 64, FastExtractOptions::default());
        assert_eq!(ext_lazy.contiguous_u8.capacity(), 0);
        let img_f32 = Array2::from_elem((64, 64), 0.1f32);
        let _ = ext_lazy.extract_f32(&img_f32);
        assert_eq!(ext_lazy.contiguous_u8.len(), 64 * 64);

        // 5. Lazy allocation of contiguous_u8 on non-contiguous u8 view when uncropped
        let mut ext_non_contig = FastExtractor::new(32, 32, FastExtractOptions::default());
        assert_eq!(ext_non_contig.contiguous_u8.capacity(), 0);
        let full_u8 = Array2::from_elem((64, 64), 15u8);
        let view = full_u8.slice(ndarray::s![0..32, 0..32]);
        let _ = ext_non_contig.extract(&view);
        assert_eq!(ext_non_contig.contiguous_u8.len(), 32 * 32);
    }

    #[test]
    fn test_all_modes_and_sigma_combinations() {
        let w = 260; // intentionally non-multiple of 16 to exercise remainder loops
        let h = 196;
        let mut frame = vec![4u8; w * h];
        let stars = [
            (40usize, 36usize, 240u8),
            (132, 96, 220),
            (216, 156, 210),
            (84, 160, 230),
        ];
        for &(sx, sy, peak) in &stars {
            for dy in -6isize..=6 {
                for dx in -6isize..=6 {
                    let dist2 = (dx * dx + dy * dy) as u8;
                    let val = peak.saturating_sub(dist2.saturating_mul(5));
                    let x = (sx as isize + dx) as usize;
                    let y = (sy as isize + dy) as usize;
                    frame[y * w + x] = frame[y * w + x].max(val);
                }
            }
        }
        let img = Array2::from_shape_vec((h, w), frame).unwrap();

        for ds in [FastDownsample::None, FastDownsample::X2, FastDownsample::X4] {
            for bg_sub_mode in [
                None,
                Some(FastBgSubMode::GlobalMean),
                Some(FastBgSubMode::GlobalMedian),
                Some(FastBgSubMode::LineMedian),
                Some(FastBgSubMode::BlockMedian { block_size: 32 }),
            ] {
                for sigma_mode in [
                    FastSigmaMode::GlobalRootSquare,
                    FastSigmaMode::GlobalMedianAbs,
                ] {
                    for approx in [false, true] {
                        for binary_open in [false, true] {
                            let opts = FastExtractOptions {
                                sigma: 2.5,
                                downsample: ds,
                                bg_sub_mode,
                                sigma_mode,
                                binary_open,
                                min_area: Some(1),
                                max_area: Some(200),
                                approximate_background: approx,
                                ..Default::default()
                            };
                            let mut ext = FastExtractor::new(w, h, opts);
                            let res1 = ext.extract(&img);
                            let res2 = ext.extract_slice(img.as_slice().unwrap());
                            assert_eq!(res1.centroids.len(), stars.len());
                            assert_eq!(res1.centroids.len(), res2.centroids.len());
                            assert!(ext.mask.iter().all(|&word| word == 0));
                            for (c1, c2) in res1.centroids.iter().zip(res2.centroids.iter()) {
                                assert_eq!(c1.x.to_bits(), c2.x.to_bits());
                                assert_eq!(c1.y.to_bits(), c2.y.to_bits());
                                assert_eq!(c1.sum.to_bits(), c2.sum.to_bits());
                                assert_eq!(c1.area, c2.area);
                                assert_eq!(c1.axis_ratio.to_bits(), c2.axis_ratio.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}
