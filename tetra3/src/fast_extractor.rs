// Required Notice: Copyright (c) 2026 Omair Kamil
//
// This file is a derivative work, inspired from `tetra3.py` of the cedar-solve and
// esa/tetra3 projects. This file has major optimizations of algorithms in those
// works with additional original computational logic.
//
// The original underlying code is licensed under the Apache License, Version 2.0.
// Original Copyright (c) European Space Agency, Steven Rosenthal, and contributors.
//
// This derivative work is licensed under the PolyForm Noncommercial License 1.0.0.
// You may not use this file except in compliance with the PolyForm Noncommercial
// License 1.0.0. A copy of the License is located in the LICENSE.md file in the
// root of this repository.
//
// Commercial use of this software is strictly prohibited without a separate
// commercial license.
//
use ndarray::{ArrayBase, Data, Ix2};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastDownsample {
    None,
    X2,
    X4,
}

impl FastDownsample {
    pub fn factor(&self) -> usize {
        match self {
            FastDownsample::None => 1,
            FastDownsample::X2 => 2,
            FastDownsample::X4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastBgSubMode {
    GlobalMedian,
    GlobalMean,
    BlockMedian { block_size: usize },
    LineMedian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastSigmaMode {
    GlobalMedianAbs,
    GlobalRootSquare,
}

#[derive(Debug, Clone)]
pub struct FastExtractOptions {
    pub sigma: f32,
    pub downsample: FastDownsample,
    pub bg_sub_mode: Option<FastBgSubMode>,
    pub sigma_mode: FastSigmaMode,
    pub binary_open: bool,
    pub centroid_window: Option<usize>,
    pub max_area: Option<usize>,
    pub min_area: Option<usize>,
    pub max_sum: Option<f64>,
    pub min_sum: Option<f64>,
    pub max_axis_ratio: Option<f64>,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct FastCentroidResult {
    pub y: f64,
    pub x: f64,
    pub sum: f64,
    pub area: usize,
    pub axis_ratio: f64,
}

/// FastExtractor maintains pre-allocated global buffers to eliminate OS memory allocations
/// during continuous execution. It utilizes a highly specialized integer-only pipeline
/// optimized specifically for zero-copy u8 inputs and integer downsampling.
pub struct FastExtractor {
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
    /// Creates a fully pre-allocated extractor.
    /// Locks dimensions and options to allow for zero-allocation runtime and LUT generation.
    pub fn new(width: usize, height: usize, options: FastExtractOptions) -> Self {
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
            let grid_w = (out_width + block_size - 1) / block_size;
            let grid_h = (out_height + block_size - 1) / block_size;
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
    pub fn extract<S>(&mut self, input_image: &ArrayBase<S, Ix2>) -> Vec<FastCentroidResult>
    where
        S: Data<Elem = u8>,
    {
        debug_assert_eq!(
            input_image.dim(),
            (self.height, self.width),
            "Input image dimensions must match the initialized FastExtractor dimensions."
        );

        // 1. Enforce Contiguous Memory for SIMD operations
        // Some operations require a flat slice. If the `ndarray` is already contiguous, we use
        // it directly (zero-copy). Otherwise, we flatten it into our pre-allocated vector.
        let src_slice = if let Some(s) = input_image.as_slice() {
            s
        } else {
            for (out_row, in_row) in self
                .contiguous_u8
                .chunks_exact_mut(self.width)
                .zip(input_image.rows())
            {
                out_row.copy_from_slice(in_row.as_slice().unwrap());
            }
            &self.contiguous_u8
        };

        let ds = self.options.downsample.factor();

        if ds > 1 {
            // =====================================================================================
            // DOWNSAMPLED PATH (Uses `u32` for accumulation, `i32` for processing)
            // =====================================================================================

            // SIMD-friendly unrolled downsampling loops.
            if ds == 2 {
                self.downsampled_u32
                    .par_chunks_exact_mut(self.out_width)
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
                    .par_chunks_exact_mut(self.out_width)
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

            // Subtract Background & Accumulate Global RMS Variance (Fused pass)
            let sum_sq_global: f64 = if let Some(bg_mode) = self.options.bg_sub_mode {
                match bg_mode {
                    FastBgSubMode::GlobalMean => {
                        let sum: u64 = self.downsampled_u32.par_iter().map(|&v| v as u64).sum();
                        let mean = (sum as f64 / self.downsampled_u32.len() as f64) as f32;
                        self.image_i32
                            .par_chunks_exact_mut(self.out_width)
                            .zip(self.downsampled_u32.par_chunks_exact(self.out_width))
                            .map(|(o_row, i_row)| {
                                let mut row_sum_sq = 0.0;
                                for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                    let val_f32 = (i as f32) - mean;
                                    *o = (val_f32 * 128.0).round() as i32;
                                    row_sum_sq += (val_f32 * val_f32) as f64;
                                }
                                row_sum_sq
                            })
                            .sum()
                    }
                    FastBgSubMode::GlobalMedian => {
                        // OPTIMIZATION: Parallel histogram computation using large chunk sizes minimizes
                        // thread-local allocations and avoids O(N) memory copies/sorting.
                        let hist = self
                            .downsampled_u32
                            .par_chunks(16384.max(self.out_width))
                            .fold(
                                || vec![0u32; 4096],
                                |mut h, chunk| {
                                    for &v in chunk {
                                        unsafe {
                                            *h.get_unchecked_mut(v as usize) += 1;
                                        }
                                    }
                                    h
                                },
                            )
                            .reduce(
                                || vec![0u32; 4096],
                                |mut h1, h2| {
                                    for (a, b) in h1.iter_mut().zip(h2.iter()) {
                                        *a += b;
                                    }
                                    h1
                                },
                            );
                        let target = ((self.downsampled_u32.len() + 1) / 2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &count) in hist.iter().enumerate() {
                            accum += count;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        self.image_i32
                            .par_chunks_exact_mut(self.out_width)
                            .zip(self.downsampled_u32.par_chunks_exact(self.out_width))
                            .map(|(o_row, i_row)| {
                                let mut row_sum_sq = 0.0;
                                for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                    let val_f32 = (i as f32) - med;
                                    *o = (val_f32 * 128.0).round() as i32;
                                    row_sum_sq += (val_f32 * val_f32) as f64;
                                }
                                row_sum_sq
                            })
                            .sum()
                    }
                    FastBgSubMode::LineMedian => {
                        // OPTIMIZATION: Thread-local histograms completely eliminate reallocation overhead.
                        // Processes row-by-row, resolving the median natively via histograms, and subtracts
                        // it from the row directly while it remains extremely hot in the L1 memory cache.
                        self.image_i32
                            .par_chunks_exact_mut(self.out_width)
                            .zip(self.downsampled_u32.par_chunks_exact(self.out_width))
                            .map_init(
                                || vec![0u32; 4096], // Safely bounds ds=4 max accumulation (4080)
                                |hist, (o_row, i_row)| {
                                    hist.fill(0);
                                    for &i in i_row {
                                        unsafe {
                                            *hist.get_unchecked_mut(i as usize) += 1;
                                        }
                                    }

                                    let target = ((self.out_width + 1) / 2) as u32;
                                    let mut accum = 0;
                                    let mut med = 0.0f32;
                                    for (val, &c) in hist.iter().enumerate() {
                                        accum += c;
                                        if accum >= target {
                                            med = val as f32;
                                            break;
                                        }
                                    }

                                    let mut row_sum_sq = 0.0;
                                    for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                        let val_f32 = (i as f32) - med;
                                        *o = (val_f32 * 128.0).round() as i32;
                                        row_sum_sq += (val_f32 * val_f32) as f64;
                                    }
                                    row_sum_sq
                                },
                            )
                            .sum()
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = (self.out_width + block_size - 1) / block_size;

                        // Compute medians for each block in the grid using histograms.
                        // We process by rows of blocks to read the source image only once.
                        self.bg_grid
                            .par_chunks_exact_mut(grid_w)
                            .enumerate()
                            .for_each_init(
                                || Vec::new(),
                                |hists, (gy, bg_row)| {
                                    let hist_size = 4096;
                                    let total_hists_size = grid_w * hist_size;
                                    if hists.len() < total_hists_size {
                                        hists.resize(total_hists_size, 0u32);
                                    }
                                    hists.fill(0);

                                    let start_y = gy * block_size;
                                    let end_y = (start_y + block_size).min(self.out_height);

                                    for y in start_y..end_y {
                                        let row_start = y * self.out_width;
                                        let row = &self.downsampled_u32
                                            [row_start..row_start + self.out_width];
                                        for gx in 0..grid_w {
                                            let start_x = gx * block_size;
                                            let end_x = (start_x + block_size).min(self.out_width);
                                            let hist_offset = gx * hist_size;
                                            for x in start_x..end_x {
                                                unsafe {
                                                    let p = *row.get_unchecked(x);
                                                    *hists.get_unchecked_mut(
                                                        hist_offset + p as usize,
                                                    ) += 1;
                                                }
                                            }
                                        }
                                    }

                                    for gx in 0..grid_w {
                                        let hist = &hists[gx * hist_size..(gx + 1) * hist_size];
                                        let start_x = gx * block_size;
                                        let end_x = (start_x + block_size).min(self.out_width);
                                        let count = (end_y - start_y) * (end_x - start_x);
                                        let target = ((count + 1) / 2) as u32;
                                        let mut accum = 0;
                                        for (val, &c) in hist.iter().enumerate() {
                                            accum += c;
                                            if accum >= target {
                                                bg_row[gx] = val as f32;
                                                break;
                                            }
                                        }
                                    }
                                },
                            );

                        // Isolate LUTs to satisfy borrow checker
                        let bg_grid = &self.bg_grid;
                        let bg_gx0 = &self.bg_gx0;
                        let bg_tx = &self.bg_tx;
                        let bg_gy0 = &self.bg_gy0;
                        let bg_gy1 = &self.bg_gy1;
                        let bg_ty = &self.bg_ty;

                        // Apply bilinearly interpolated subtraction row-by-row using fast LUTs
                        self.image_i32
                            .par_chunks_exact_mut(self.out_width)
                            .zip(self.downsampled_u32.par_chunks_exact(self.out_width))
                            .enumerate()
                            .map(|(y, (out_row, src_row))| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y];
                                let row1_start = bg_gy1[y];
                                let ty = bg_ty[y];

                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];

                                // Pre-calculate interpolated grid medians for this row to reduce inner loop math.
                                // 1024 is sufficient for images up to 16K with block size 16.
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

                                for x in 0..self.out_width {
                                    unsafe {
                                        let gx0 = *bg_gx0.get_unchecked(x);
                                        let tx = *bg_tx.get_unchecked(x);

                                        // Optimized bilinear interpolation: reduces 3 fmuls + 3 fadds to 1 fmul + 1 fadd
                                        let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];

                                        let val_f32 = (*src_row.get_unchecked(x) as f32) - bg_val;
                                        *out_row.get_unchecked_mut(x) =
                                            (val_f32 * 128.0).round() as i32;
                                        row_sq_sum += (val_f32 * val_f32) as f64;
                                    }
                                }
                                row_sq_sum
                            })
                            .sum()
                    }
                }
            } else {
                self.image_i32
                    .par_iter_mut()
                    .zip(self.downsampled_u32.par_iter())
                    .map(|(o, &i)| {
                        let val_f32 = i as f32;
                        *o = (val_f32 * 128.0).round() as i32;
                        (val_f32 * val_f32) as f64
                    })
                    .sum()
            };

            // Calculate local noise floor threshold
            let threshold_f32 = match self.options.sigma_mode {
                FastSigmaMode::GlobalRootSquare => {
                    let mean_sq =
                        (sum_sq_global / (self.out_height * self.out_width) as f64) as f32;
                    mean_sq.max(0.0).sqrt() * self.options.sigma
                }
                FastSigmaMode::GlobalMedianAbs => {
                    self.median_scratch_i32.clear();
                    self.median_scratch_i32
                        .extend(self.image_i32.iter().map(|&v| v.abs()));
                    let mid = self.median_scratch_i32.len() / 2;
                    let (_, &mut median, _) = self.median_scratch_i32.select_nth_unstable(mid);
                    (median as f32 / 128.0) * 1.48 * self.options.sigma
                }
            };

            let threshold_scaled = (threshold_f32 * 128.0).round() as i32;

            // Execute algorithm using disjoint field borrows to prevent memory reallocation
            Self::execute_erosion_and_extraction(
                &self.image_i32,
                self.out_width,
                self.out_height,
                threshold_scaled,
                ds,
                &self.options,
                &mut self.mask,
                &mut self.stack,
                &self.cw_wx,
                &self.cw_wy,
                &self.cw_strides,
            )
        } else {
            // =====================================================================================
            // 1x RESOLUTION PATH (Uses `i16` for memory efficiency)
            // =====================================================================================

            // Subtract Background & Accumulate Global RMS Variance (Fused pass)
            let sum_sq_global: f64 = if let Some(bg_mode) = self.options.bg_sub_mode {
                match bg_mode {
                    FastBgSubMode::GlobalMean => {
                        let sum: u64 = src_slice.par_iter().map(|&v| v as u64).sum();
                        let mean = (sum as f64 / src_slice.len() as f64) as f32;
                        self.image_i16
                            .par_chunks_exact_mut(self.width)
                            .zip(src_slice.par_chunks_exact(self.width))
                            .map(|(o_row, i_row)| {
                                let mut row_sum_sq = 0.0;
                                for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                    let val_f32 = (i as f32) - mean;
                                    *o = (val_f32 * 128.0).round() as i16;
                                    row_sum_sq += (val_f32 * val_f32) as f64;
                                }
                                row_sum_sq
                            })
                            .sum()
                    }
                    FastBgSubMode::GlobalMedian => {
                        // OPTIMIZATION: Parallel histogram computation using large chunk sizes minimizes
                        // thread-local allocations and avoids O(N) memory copies/sorting.
                        let hist = src_slice
                            .par_chunks(16384.max(self.width))
                            .fold(
                                || vec![0u32; 256],
                                |mut h, chunk| {
                                    for &v in chunk {
                                        unsafe {
                                            *h.get_unchecked_mut(v as usize) += 1;
                                        }
                                    }
                                    h
                                },
                            )
                            .reduce(
                                || vec![0u32; 256],
                                |mut h1, h2| {
                                    for (a, b) in h1.iter_mut().zip(h2.iter()) {
                                        *a += b;
                                    }
                                    h1
                                },
                            );
                        let target = ((src_slice.len() + 1) / 2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &count) in hist.iter().enumerate() {
                            accum += count;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        self.image_i16
                            .par_chunks_exact_mut(self.width)
                            .zip(src_slice.par_chunks_exact(self.width))
                            .map(|(o_row, i_row)| {
                                let mut row_sum_sq = 0.0;
                                for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                    let val_f32 = (i as f32) - med;
                                    *o = (val_f32 * 128.0).round() as i16;
                                    row_sum_sq += (val_f32 * val_f32) as f64;
                                }
                                row_sum_sq
                            })
                            .sum()
                    }
                    FastBgSubMode::LineMedian => {
                        // OPTIMIZATION: Thread-local histograms completely eliminate reallocation overhead.
                        // Processes row-by-row, resolving the median natively via histograms, and subtracts
                        // it from the row directly while it remains extremely hot in the L1 memory cache.
                        self.image_i16
                            .par_chunks_exact_mut(self.width)
                            .zip(src_slice.par_chunks_exact(self.width))
                            .map_init(
                                || vec![0u32; 256],
                                |hist, (o_row, i_row)| {
                                    hist.fill(0);
                                    for &i in i_row {
                                        unsafe {
                                            *hist.get_unchecked_mut(i as usize) += 1;
                                        }
                                    }

                                    let target = ((self.width + 1) / 2) as u32;
                                    let mut accum = 0;
                                    let mut med = 0.0f32;
                                    for (val, &c) in hist.iter().enumerate() {
                                        accum += c;
                                        if accum >= target {
                                            med = val as f32;
                                            break;
                                        }
                                    }

                                    let mut row_sum_sq = 0.0;
                                    for (o, &i) in o_row.iter_mut().zip(i_row.iter()) {
                                        let val_f32 = (i as f32) - med;
                                        *o = (val_f32 * 128.0).round() as i16;
                                        row_sum_sq += (val_f32 * val_f32) as f64;
                                    }
                                    row_sum_sq
                                },
                            )
                            .sum()
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = (self.width + block_size - 1) / block_size;

                        // Compute medians for each block in the grid using histograms.
                        // We process by rows of blocks to read the source image only once.
                        self.bg_grid
                            .par_chunks_exact_mut(grid_w)
                            .enumerate()
                            .for_each_init(
                                || Vec::new(),
                                |hists, (gy, bg_row)| {
                                    let hist_size = 256;
                                    let total_hists_size = grid_w * hist_size;
                                    if hists.len() < total_hists_size {
                                        hists.resize(total_hists_size, 0u32);
                                    }
                                    hists.fill(0);

                                    let start_y = gy * block_size;
                                    let end_y = (start_y + block_size).min(self.height);

                                    for y in start_y..end_y {
                                        let row_start = y * self.width;
                                        let row = &src_slice[row_start..row_start + self.width];
                                        for gx in 0..grid_w {
                                            let start_x = gx * block_size;
                                            let end_x = (start_x + block_size).min(self.width);
                                            let hist_offset = gx * hist_size;
                                            for x in start_x..end_x {
                                                unsafe {
                                                    let p = *row.get_unchecked(x);
                                                    *hists.get_unchecked_mut(
                                                        hist_offset + p as usize,
                                                    ) += 1;
                                                }
                                            }
                                        }
                                    }

                                    for gx in 0..grid_w {
                                        let hist = &hists[gx * hist_size..(gx + 1) * hist_size];
                                        let start_x = gx * block_size;
                                        let end_x = (start_x + block_size).min(self.width);
                                        let count = (end_y - start_y) * (end_x - start_x);
                                        let target = ((count + 1) / 2) as u32;
                                        let mut accum = 0;
                                        for (val, &c) in hist.iter().enumerate() {
                                            accum += c;
                                            if accum >= target {
                                                bg_row[gx] = val as f32;
                                                break;
                                            }
                                        }
                                    }
                                },
                            );

                        // Isolate LUTs to satisfy borrow checker
                        let bg_grid = &self.bg_grid;
                        let bg_gx0 = &self.bg_gx0;
                        let bg_tx = &self.bg_tx;
                        let bg_gy0 = &self.bg_gy0;
                        let bg_gy1 = &self.bg_gy1;
                        let bg_ty = &self.bg_ty;

                        self.image_i16
                            .par_chunks_exact_mut(self.width)
                            .zip(src_slice.par_chunks_exact(self.width))
                            .enumerate()
                            .map(|(y, (out_row, src_row))| {
                                let mut row_sq_sum = 0.0;
                                let row0_start = bg_gy0[y];
                                let row1_start = bg_gy1[y];
                                let ty = bg_ty[y];

                                let row_v0 = &bg_grid[row0_start..row0_start + grid_w];
                                let row_v1 = &bg_grid[row1_start..row1_start + grid_w];

                                // Pre-calculate interpolated grid medians for this row to reduce inner loop math.
                                // 1024 is sufficient for images up to 16K with block size 16.
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

                                for x in 0..self.width {
                                    unsafe {
                                        let gx0 = *bg_gx0.get_unchecked(x);
                                        let tx = *bg_tx.get_unchecked(x);

                                        // Optimized bilinear interpolation: reduces 3 fmuls + 3 fadds to 1 fmul + 1 fadd
                                        let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];

                                        let val_f32 = (*src_row.get_unchecked(x) as f32) - bg_val;
                                        *out_row.get_unchecked_mut(x) =
                                            (val_f32 * 128.0).round() as i16;
                                        row_sq_sum += (val_f32 * val_f32) as f64;
                                    }
                                }
                                row_sq_sum
                            })
                            .sum()
                    }
                }
            } else {
                // If bg subtraction is omitted, directly scale to preserve floating precision
                // u8 pixel * 128 max is 32,640, safely inside i16 MAX limit of 32,767.
                self.image_i16
                    .par_iter_mut()
                    .zip(src_slice.par_iter())
                    .map(|(o, &i)| {
                        let val_f32 = i as f32;
                        *o = (val_f32 * 128.0).round() as i16;
                        (val_f32 * val_f32) as f64
                    })
                    .sum()
            };

            let threshold_f32 = match self.options.sigma_mode {
                FastSigmaMode::GlobalRootSquare => {
                    let mean_sq = (sum_sq_global / (self.height * self.width) as f64) as f32;
                    mean_sq.max(0.0).sqrt() * self.options.sigma
                }
                FastSigmaMode::GlobalMedianAbs => {
                    self.median_scratch_i16.clear();
                    self.median_scratch_i16
                        .extend(self.image_i16.iter().map(|&v| v.abs()));
                    let mid = self.median_scratch_i16.len() / 2;
                    let (_, &mut median, _) = self.median_scratch_i16.select_nth_unstable(mid);
                    (median as f32 / 128.0) * 1.48 * self.options.sigma
                }
            };

            let threshold_scaled = (threshold_f32 * 128.0).round() as i16;

            Self::execute_erosion_and_extraction(
                &self.image_i16,
                self.width,
                self.height,
                threshold_scaled,
                1,
                &self.options,
                &mut self.mask,
                &mut self.stack,
                &self.cw_wx,
                &self.cw_wy,
                &self.cw_strides,
            )
        }
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
        T: Copy + PartialOrd + Send + Sync + Into<f64>,
    {
        // 1. Fast binary erosion + threshold
        // Rather than thresholds then eroding in two passes, we perform a fused 3x3 cross
        // morphological evaluation directly off the scalar threshold.
        let chunk_size = (height / rayon::current_num_threads()).max(64);
        let eroded_pixels: Vec<usize> = if options.binary_open {
            let chunks = height.saturating_sub(2).div_ceil(chunk_size);
            (0..chunks)
                .into_par_iter()
                .fold(
                    || Vec::with_capacity(128),
                    |mut acc, chunk_idx| {
                        let start_y = 1 + chunk_idx * chunk_size;
                        let end_y = (start_y + chunk_size).min(height - 1);
                        for y in start_y..end_y {
                            let row_offset = y * width;
                            // Using raw pointers strips vector bounds checking inside the tight loop
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
                                        acc.push(row_offset + x);
                                    }
                                }
                            }
                        }
                        acc
                    },
                )
                .reduce(Vec::new, |mut a, mut b| {
                    a.append(&mut b);
                    a
                })
        } else {
            let chunks = height.div_ceil(chunk_size);
            (0..chunks)
                .into_par_iter()
                .fold(
                    || Vec::with_capacity(128),
                    |mut acc, chunk_idx| {
                        let start_y = chunk_idx * chunk_size;
                        let end_y = (start_y + chunk_size).min(height);
                        for y in start_y..end_y {
                            let row_offset = y * width;
                            let r_curr = &img[row_offset..row_offset + width];
                            for x in 0..width {
                                if r_curr[x] > threshold {
                                    acc.push(row_offset + x);
                                }
                            }
                        }
                        acc
                    },
                )
                .reduce(Vec::new, |mut a, mut b| {
                    a.append(&mut b);
                    a
                })
        };

        // 2. Binary dilation
        // We write the "true" boolean state to the cross pattern directly in memory based on the
        // recorded eroded indices, functioning as an instant morphological dilation.
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

        // Optimization: Pre-allocate capacity
        let mut extracted = Vec::with_capacity(256);

        // Pre-unwrap configuration parameters
        let min_a = options.min_area.unwrap_or(0);
        let max_a = options.max_area.unwrap_or(usize::MAX);
        let min_s = options.min_sum.unwrap_or(0.0);
        let max_s = options.max_sum.unwrap_or(f64::MAX);
        let max_ar = options.max_axis_ratio.unwrap_or(f64::MAX);

        // 3. Flood Fill & Extract
        // Use an internal stack to trace 4-connected components, evaluating moments
        // mathematically via the parallel axis theorem on the fly.
        for &seed in &eroded_pixels {
            if !mask[seed] {
                continue;
            }

            mask[seed] = false;

            let mut area = 1;
            // Unscale back to floating point bounds
            let val = img[seed].into() * (1.0 / 128.0);
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

                let mut check_push = |ni: usize, nx: f64, ny: f64| unsafe {
                    if *mask.get_unchecked(ni) {
                        *mask.get_unchecked_mut(ni) = false;
                        area += 1;
                        let v = (*img.get_unchecked(ni)).into() * (1.0 / 128.0);
                        sum += v;
                        sum_x += nx * v;
                        sum_y += ny * v;
                        sum_xx += nx * nx * v;
                        sum_yy += ny * ny * v;
                        sum_xy += nx * ny * v;
                        stack.push(ni);
                    }
                };

                if cy > 0 {
                    check_push(idx - width, cx as f64, (cy - 1) as f64);
                }
                if cy + 1 < height {
                    check_push(idx + width, cx as f64, (cy + 1) as f64);
                }
                if cx > 0 {
                    check_push(idx - 1, (cx - 1) as f64, cy as f64);
                }
                if cx + 1 < width {
                    check_push(idx + 1, (cx + 1) as f64, cy as f64);
                }
            }

            // Filtering rules evaluated entirely against un-wrapped variables
            if area < min_a || area > max_a || sum < min_s || sum > max_s || sum == 0.0 {
                continue;
            }

            // Calculate final moments
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
                sum,
                area,
                axis_ratio,
            });
        }

        // Sort descending by sum
        extracted.sort_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap_or(Ordering::Equal));

        // Centroid Window recalculation
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
                        // Optimization: Fetch elided row bounds and precalculated floating weights directly
                        let row_start = o_y * width + *cw_strides.get_unchecked(wy) + o_x;
                        let row_slice = img.get_unchecked(row_start..row_start + window);
                        let wy_f = *cw_wy.get_unchecked(wy);

                        for (wx, &v) in row_slice.iter().enumerate() {
                            let val = v.into() * (1.0 / 128.0);
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

        // Revert downsample coordinate shift
        if ds > 1 {
            for centroid in &mut extracted {
                centroid.x *= ds as f64;
                centroid.y *= ds as f64;
            }
        }

        extracted
    }
}

include!("fast_extractor_seq.rs");
