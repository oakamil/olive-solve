// Required Notice: Copyright (c) 2026 Omair Kamil
//
// This file is a derivative work - a port to Rust with heavy performance
// optimizations from `tetra3.py` of the cedar-solve and esa/tetra3 projects.
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

use ndarray::{Array2, s};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgSubMode {
    LocalMedian,
    LocalMean,
    GlobalMedian,
    GlobalMean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaMode {
    LocalMedianAbs,
    LocalRootSquare,
    GlobalMedianAbs,
    GlobalRootSquare,
}

#[derive(Debug, Clone)]
pub enum Crop {
    /// Scalar: Image is cropped to given fraction (e.g. 2 gives 1/2 size image out).
    Fraction(usize),
    /// 2-tuple: Image is cropped to centered region.
    Center { height: usize, width: usize },
    /// 4-tuple: Image is cropped to region with an offset.
    Region {
        height: usize,
        width: usize,
        offset_y: isize,
        offset_x: isize,
    },
}

#[derive(Debug, Clone)]
pub struct CentroidConfig {
    pub sigma: f32,
    pub image_th: Option<f32>,
    pub crop: Option<Crop>,
    pub downsample: Option<usize>,
    pub filtsize: usize,
    pub bg_sub_mode: Option<BgSubMode>,
    pub sigma_mode: SigmaMode,
    pub binary_open: bool,
    pub centroid_window: Option<usize>,
    pub max_area: Option<usize>,
    pub min_area: Option<usize>,
    pub max_sum: Option<f64>,
    pub min_sum: Option<f64>,
    pub max_axis_ratio: Option<f64>,
    pub max_returned: Option<usize>,
    pub return_images: bool,
}

impl Default for CentroidConfig {
    fn default() -> Self {
        CentroidConfig {
            sigma: 2.0,
            image_th: None,
            crop: None,
            downsample: None,
            filtsize: 25,
            bg_sub_mode: Some(BgSubMode::LocalMean),
            sigma_mode: SigmaMode::GlobalRootSquare,
            binary_open: true,
            centroid_window: None,
            max_area: Some(100),
            min_area: Some(5),
            max_sum: None,
            min_sum: None,
            max_axis_ratio: None,
            max_returned: None,
            return_images: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CentroidResult {
    pub y: f64,
    pub x: f64,
    pub sum: f64,
    pub area: usize,
    pub m2_xx: f64,
    pub m2_yy: f64,
    pub m2_xy: f64,
    pub axis_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct DebugImages {
    pub cropped_and_downsampled: Array2<f32>,
    pub removed_background: Array2<f32>,
    pub binary_mask: Array2<bool>,
}

#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub centroids: Vec<CentroidResult>,
    pub debug_images: Option<DebugImages>,
}

/// Helper: 2D Uniform Filter (Box Blur)
/// Optimized for ARM NEON: Uses simple indexing (no deep iterator zips) to guarantee LLVM auto-vectorization.
/// Note: Reflect-edge padding is computed mathematically in-place, eliminating padded array allocations.
fn fast_box_blur_2d(
    src: &[f32],
    scratch: &mut [f32],
    out: &mut [f32],
    w: usize,
    h: usize,
    size: usize,
) {
    let rad = size / 2;
    let area = (size * size) as f32;

    // Horizontal Pass
    scratch
        .par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(s_row, i_row)| {
            let mut sum = 0.0_f32;
            if w > 2 * rad {
                for x in 0..=rad {
                    sum += i_row[x];
                }
                for x in 1..=rad {
                    sum += i_row[x - 1];
                } // Reflection
                s_row[0] = sum;

                for x in 1..=rad {
                    sum += i_row[x + rad] - i_row[rad - x];
                    s_row[x] = sum;
                }
                for x in (rad + 1)..(w - rad) {
                    // Branchless inner core
                    sum += i_row[x + rad] - i_row[x - rad - 1];
                    s_row[x] = sum;
                }
                for x in (w - rad)..w {
                    let add_px = if x + rad >= w {
                        2 * w - 1 - (x + rad)
                    } else {
                        x + rad
                    };
                    sum += i_row[add_px] - i_row[x - rad - 1];
                    s_row[x] = sum;
                }
            } else {
                // Fallback for extremely narrow images
                let rad_i = rad as isize;
                for x in -rad_i..=rad_i {
                    let px = if x < 0 {
                        (-x - 1) as usize
                    } else if x >= w as isize {
                        (2 * w as isize - 1 - x) as usize
                    } else {
                        x as usize
                    };
                    sum += i_row[px];
                }
                s_row[0] = sum;
                for x in 1..w {
                    let add_x = (x as isize) + rad_i;
                    let add_px = if add_x >= w as isize {
                        (2 * w as isize - 1 - add_x).max(0) as usize
                    } else {
                        add_x as usize
                    };
                    let sub_x = (x as isize) - rad_i - 1;
                    let sub_px = if sub_x < 0 {
                        (-sub_x - 1).max(0) as usize
                    } else {
                        sub_x as usize
                    };
                    sum += i_row[add_px] - i_row[sub_px];
                    s_row[x] = sum;
                }
            }
        });

    // Vertical Pass (Chunked via row-banding for parallelism without memory collisions)
    let chunk_rows = (h + rayon::current_num_threads() - 1) / rayon::current_num_threads();
    let chunk_rows = chunk_rows.max(16);

    out.par_chunks_mut(chunk_rows * w)
        .enumerate()
        .for_each(|(chunk_idx, o_chunk)| {
            let start_y = chunk_idx * chunk_rows;
            let end_y = start_y + (o_chunk.len() / w);
            let mut col_sums = vec![0.0_f32; w];

            for y in (start_y as isize - rad as isize)..=(start_y as isize + rad as isize) {
                let py = if y < 0 {
                    (-y - 1) as usize
                } else if y >= h as isize {
                    (2 * h as isize - 1 - y) as usize
                } else {
                    y as usize
                };
                let s_row = &scratch[py * w..(py + 1) * w];
                for x in 0..w {
                    col_sums[x] += s_row[x];
                }
            }

            let o_row = &mut o_chunk[0..w];
            for x in 0..w {
                o_row[x] = col_sums[x] / area;
            }

            for y in (start_y + 1)..end_y {
                let add_y = (y as isize) + rad as isize;
                let add_py = if add_y >= h as isize {
                    (2 * h as isize - 1 - add_y) as usize
                } else {
                    add_y as usize
                };
                let sub_y = (y as isize) - rad as isize - 1;
                let sub_py = if sub_y < 0 {
                    (-sub_y - 1) as usize
                } else {
                    sub_y as usize
                };

                let add_row = &scratch[add_py * w..(add_py + 1) * w];
                let sub_row = &scratch[sub_py * w..(sub_py + 1) * w];
                let local_y = y - start_y;
                let o_row = &mut o_chunk[local_y * w..(local_y + 1) * w];

                for x in 0..w {
                    col_sums[x] += add_row[x] - sub_row[x];
                    o_row[x] = col_sums[x] / area;
                }
            }
        });
}

/// Helper: 2D Median Filter using raw slices
fn fast_median_filter_2d(src: &[f32], out: &mut [f32], w: usize, h: usize, size: usize) {
    let pad = (size / 2) as isize;
    let mid = (size * size) / 2;

    out.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let y_i = y as isize;
            let mut window = vec![0.0; size * size];
            for x in 0..w {
                let x_i = x as isize;
                let mut idx = 0;
                for wy in -pad..=pad {
                    for wx in -pad..=pad {
                        let mut sy = y_i + wy;
                        if sy < 0 {
                            sy = -sy - 1;
                        } else if sy >= h as isize {
                            sy = 2 * h as isize - 1 - sy;
                        }
                        let mut sx = x_i + wx;
                        if sx < 0 {
                            sx = -sx - 1;
                        } else if sx >= w as isize {
                            sx = 2 * w as isize - 1 - sx;
                        }
                        window[idx] = src[(sy as usize) * w + (sx as usize)];
                        idx += 1;
                    }
                }
                let (_, &mut median, _) = window.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                out_row[x] = median;
            }
        });
}

/// Extractor maintains pre-allocated global buffers to eliminate OS memory allocations
/// during continuous execution, fulfilling the zero-allocation performance pattern.
pub struct Extractor {
    pub config: CentroidConfig,
    image_vec: Vec<f32>,
    scratch: Vec<f32>,
    median_scratch: Vec<f32>,
    std_img: Vec<f32>,
    mask: Vec<bool>,
    stack: Vec<usize>, // Tiny stack size keeps the L1 cache hot on Pi 5 (bandwidth constrained)
}

impl Extractor {
    pub fn new(config: CentroidConfig) -> Self {
        Self {
            config,
            image_vec: Vec::new(),
            scratch: Vec::new(),
            median_scratch: Vec::new(),
            std_img: Vec::new(),
            mask: Vec::new(),
            stack: Vec::with_capacity(1024),
        }
    }

    /// Extract spot centroids from an image and calculate statistics.
    pub fn extract(&mut self, input_image: &Array2<f32>) -> ExtractionResult {
        // 1. Crop and downsample
        // Note: Cropping is applied before downsampling.
        let (full_height, full_width) = input_image.dim();

        let (mut height, mut width, offs_h_isize, offs_w_isize) = match self.config.crop {
            Some(Crop::Fraction(f)) => (full_height / f, full_width / f, 0isize, 0isize),
            Some(Crop::Center {
                height: h,
                width: w,
            }) => (h, w, 0isize, 0isize),
            Some(Crop::Region {
                height: h,
                width: w,
                offset_y,
                offset_x,
            }) => (h, w, offset_y, offset_x),
            None => (full_height, full_width, 0isize, 0isize),
        };

        let final_offs_h;
        let final_offs_w;

        if self.config.crop.is_some() {
            let divisor = self.config.downsample.unwrap_or(2);
            height = ((height as f32 / divisor as f32).ceil() as usize) * divisor;
            width = ((width as f32 / divisor as f32).ceil() as usize) * divisor;
            height = height.min(full_height);
            width = width.min(full_width);

            final_offs_h = (offs_h_isize + (full_height as isize - height as isize) / 2)
                .clamp(0, (full_height - height) as isize) as usize;
            final_offs_w = (offs_w_isize + (full_width as isize - width as isize) / 2)
                .clamp(0, (full_width - width) as isize) as usize;
        } else {
            final_offs_h = 0;
            final_offs_w = 0;
        }

        let cropped = input_image.slice(s![
            final_offs_h..final_offs_h + height,
            final_offs_w..final_offs_w + width
        ]);

        self.image_vec.clear();
        if let Some(ds) = self.config.downsample {
            height /= ds;
            width /= ds;
            self.image_vec.resize(height * width, 0.0);

            self.image_vec
                .par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..width {
                        let mut sum = 0.0;
                        for dy in 0..ds {
                            for dx in 0..ds {
                                sum += cropped[[y * ds + dy, x * ds + dx]];
                            }
                        }
                        // Hardcoded `sum_when_downsample = true` (mathematically identical to Python wrapper)
                        row[x] = sum;
                    }
                });
        } else {
            // Memory Optimization: Eliminate unnecessary zeroing memset
            if let Some(s) = cropped.as_slice() {
                self.image_vec.extend_from_slice(s);
            } else {
                self.image_vec.reserve(height * width);
                for row in cropped.rows() {
                    self.image_vec.extend(row.iter().copied());
                }
            }
        }

        let dbg_cropped = if self.config.return_images {
            Some(Array2::from_shape_vec((height, width), self.image_vec.clone()).unwrap())
        } else {
            None
        };

        // 2. Subtract background
        // Fused background subtraction + RMS calculation (Evaluated as an expression)
        let sum_sq_global: f64 = if let Some(mode) = self.config.bg_sub_mode {
            match mode {
                BgSubMode::LocalMean => {
                    self.scratch.resize(width * height, 0.0);
                    let rad = self.config.filtsize / 2;
                    let area = (self.config.filtsize * self.config.filtsize) as f32;

                    self.scratch
                        .par_chunks_exact_mut(width)
                        .zip(self.image_vec.par_chunks_exact(width))
                        .for_each(|(s_row, i_row)| {
                            let mut sum = 0.0_f32;
                            if width > 2 * rad {
                                for x in 0..=rad {
                                    sum += i_row[x];
                                }
                                for x in 1..=rad {
                                    sum += i_row[x - 1];
                                }
                                s_row[0] = sum;

                                for x in 1..=rad {
                                    sum += i_row[x + rad] - i_row[rad - x];
                                    s_row[x] = sum;
                                }
                                for x in (rad + 1)..(width - rad) {
                                    sum += i_row[x + rad] - i_row[x - rad - 1];
                                    s_row[x] = sum;
                                }
                                for x in (width - rad)..width {
                                    let add_px = if x + rad >= width {
                                        2 * width - 1 - (x + rad)
                                    } else {
                                        x + rad
                                    };
                                    sum += i_row[add_px] - i_row[x - rad - 1];
                                    s_row[x] = sum;
                                }
                            } else {
                                let rad_i = rad as isize;
                                for x in -rad_i..=rad_i {
                                    let px = if x < 0 {
                                        (-x - 1) as usize
                                    } else if x >= width as isize {
                                        (2 * width as isize - 1 - x) as usize
                                    } else {
                                        x as usize
                                    };
                                    sum += i_row[px];
                                }
                                s_row[0] = sum;
                                for x in 1..width {
                                    let add_x = (x as isize) + rad_i;
                                    let add_px = if add_x >= width as isize {
                                        (2 * width as isize - 1 - add_x).max(0) as usize
                                    } else {
                                        add_x as usize
                                    };
                                    let sub_x = (x as isize) - rad_i - 1;
                                    let sub_px = if sub_x < 0 {
                                        (-sub_x - 1).max(0) as usize
                                    } else {
                                        sub_x as usize
                                    };
                                    sum += i_row[add_px] - i_row[sub_px];
                                    s_row[x] = sum;
                                }
                            }
                        });

                    let chunk_rows =
                        (height + rayon::current_num_threads() - 1) / rayon::current_num_threads();
                    let chunk_rows = chunk_rows.max(16);
                    let scratch_ref = &self.scratch;

                    self.image_vec
                        .par_chunks_mut(chunk_rows * width)
                        .enumerate()
                        .map(|(chunk_idx, i_chunk)| {
                            let start_y = chunk_idx * chunk_rows;
                            let end_y = start_y + (i_chunk.len() / width);
                            let mut col_sums = vec![0.0_f32; width];

                            for y in (start_y as isize - rad as isize)
                                ..=(start_y as isize + rad as isize)
                            {
                                let py = if y < 0 {
                                    (-y - 1) as usize
                                } else if y >= height as isize {
                                    (2 * height as isize - 1 - y) as usize
                                } else {
                                    y as usize
                                };
                                let s_row = &scratch_ref[py * width..(py + 1) * width];
                                for x in 0..width {
                                    col_sums[x] += s_row[x];
                                }
                            }

                            let mut local_sq_sum = 0.0_f64;

                            {
                                let i_row = &mut i_chunk[0..width];
                                for x in 0..width {
                                    let val = i_row[x] - (col_sums[x] / area);
                                    i_row[x] = val;
                                    local_sq_sum += (val as f64) * (val as f64);
                                }
                            }

                            for y in (start_y + 1)..end_y {
                                let add_y = (y as isize) + rad as isize;
                                let add_py = if add_y >= height as isize {
                                    (2 * height as isize - 1 - add_y) as usize
                                } else {
                                    add_y as usize
                                };
                                let sub_y = (y as isize) - rad as isize - 1;
                                let sub_py = if sub_y < 0 {
                                    (-sub_y - 1) as usize
                                } else {
                                    sub_y as usize
                                };

                                let add_row = &scratch_ref[add_py * width..(add_py + 1) * width];
                                let sub_row = &scratch_ref[sub_py * width..(sub_py + 1) * width];
                                let local_y = y - start_y;
                                let i_row = &mut i_chunk[local_y * width..(local_y + 1) * width];

                                for x in 0..width {
                                    col_sums[x] += add_row[x] - sub_row[x];
                                    let val = i_row[x] - (col_sums[x] / area);
                                    i_row[x] = val;
                                    local_sq_sum += (val as f64) * (val as f64);
                                }
                            }
                            local_sq_sum
                        })
                        .sum()
                }
                BgSubMode::GlobalMedian => {
                    self.median_scratch.clear();
                    self.median_scratch.extend_from_slice(&self.image_vec);
                    let mid = self.median_scratch.len() / 2;
                    let (_, &mut median, _) =
                        self.median_scratch.select_nth_unstable_by(mid, |a, b| {
                            a.partial_cmp(b).unwrap_or(Ordering::Equal)
                        });

                    self.image_vec
                        .par_iter_mut()
                        .map(|i| {
                            *i -= median;
                            (*i as f64) * (*i as f64)
                        })
                        .sum()
                }
                BgSubMode::GlobalMean => {
                    let sum: f64 = self.image_vec.par_iter().map(|&v| v as f64).sum();
                    let mean = (sum / self.image_vec.len() as f64) as f32;
                    self.image_vec
                        .par_iter_mut()
                        .map(|i| {
                            *i -= mean;
                            (*i as f64) * (*i as f64)
                        })
                        .sum()
                }
                BgSubMode::LocalMedian => {
                    self.scratch.resize(width * height, 0.0);
                    fast_median_filter_2d(
                        &self.image_vec,
                        &mut self.scratch,
                        width,
                        height,
                        self.config.filtsize,
                    );
                    let bg = &self.scratch;
                    self.image_vec
                        .par_iter_mut()
                        .zip(bg.par_iter())
                        .map(|(i, &b)| {
                            *i -= b;
                            (*i as f64) * (*i as f64)
                        })
                        .sum()
                }
            }
        } else {
            self.image_vec
                .par_iter()
                .map(|&i| (i as f64) * (i as f64))
                .sum()
        };

        let dbg_bg_sub = if self.config.return_images {
            Some(Array2::from_shape_vec((height, width), self.image_vec.clone()).unwrap())
        } else {
            None
        };

        // 3. Find noise standard deviation to threshold
        enum Threshold<'a> {
            Scalar(f32),
            Array(&'a [f32]),
        }

        let threshold = if let Some(th) = self.config.image_th {
            Threshold::Scalar(th)
        } else {
            match self.config.sigma_mode {
                SigmaMode::GlobalRootSquare => {
                    let mean_sq = (sum_sq_global / (height * width) as f64) as f32;
                    Threshold::Scalar(mean_sq.max(0.0).sqrt() * self.config.sigma)
                }
                SigmaMode::GlobalMedianAbs => {
                    self.median_scratch.clear();
                    self.median_scratch
                        .extend(self.image_vec.iter().map(|v| v.abs()));
                    let mid = self.median_scratch.len() / 2;
                    let (_, &mut median, _) =
                        self.median_scratch.select_nth_unstable_by(mid, |a, b| {
                            a.partial_cmp(b).unwrap_or(Ordering::Equal)
                        });
                    Threshold::Scalar(median * 1.48 * self.config.sigma)
                }
                SigmaMode::LocalMedianAbs => {
                    self.std_img.resize(width * height, 0.0);
                    self.scratch.clear();
                    self.scratch.extend(self.image_vec.iter().map(|v| v.abs()));
                    fast_median_filter_2d(
                        &self.scratch,
                        &mut self.std_img,
                        width,
                        height,
                        self.config.filtsize,
                    );
                    self.std_img
                        .par_iter_mut()
                        .for_each(|v| *v *= 1.48 * self.config.sigma);
                    Threshold::Array(&self.std_img)
                }
                SigmaMode::LocalRootSquare => {
                    self.std_img.resize(width * height, 0.0);
                    self.scratch.clear();
                    self.scratch.extend(self.image_vec.iter().map(|v| v * v));
                    self.median_scratch.resize(width * height, 0.0);
                    fast_box_blur_2d(
                        &self.scratch,
                        &mut self.median_scratch,
                        &mut self.std_img,
                        width,
                        height,
                        self.config.filtsize,
                    );
                    self.std_img
                        .par_iter_mut()
                        .for_each(|v| *v = v.max(0.0).sqrt() * self.config.sigma);
                    Threshold::Array(&self.std_img)
                }
            }
        };

        // 4. Threshold to find binary mask
        // Fused Fast Extraction: Evaluates Threshold + Binary Erosion (3x3 cross) in a single pass.
        let chunk_size = (height / rayon::current_num_threads()).max(64);

        let eroded_pixels: Vec<usize> = match threshold {
            Threshold::Scalar(th) => {
                if self.config.binary_open {
                    let chunks = (height.saturating_sub(2) + chunk_size - 1) / chunk_size;

                    (0..chunks)
                        .into_par_iter()
                        .fold(
                            || Vec::with_capacity(128),
                            |mut acc, chunk_idx| {
                                let start_y = 1 + chunk_idx * chunk_size;
                                let end_y = (start_y + chunk_size).min(height - 1);
                                for y in start_y..end_y {
                                    let row_offset = y * width;

                                    // Optimization: Extracting exact row slices enables the compiler to mathematically prove
                                    // boundary safety, entirely stripping vector bounds checks from the inner `x` loop.
                                    let r_prev = &self.image_vec[(y - 1) * width..y * width];
                                    let r_curr = &self.image_vec[y * width..(y + 1) * width];
                                    let r_next = &self.image_vec[(y + 1) * width..(y + 2) * width];

                                    for x in 1..width - 1 {
                                        if r_curr[x] > th {
                                            if r_curr[x - 1] > th
                                                && r_curr[x + 1] > th
                                                && r_prev[x] > th
                                                && r_next[x] > th
                                            {
                                                acc.push(row_offset + x);
                                            }
                                        }
                                    }
                                }
                                acc
                            },
                        )
                        .reduce(
                            || Vec::new(),
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        )
                } else {
                    let chunks = (height + chunk_size - 1) / chunk_size;
                    (0..chunks)
                        .into_par_iter()
                        .fold(
                            || Vec::with_capacity(128),
                            |mut acc, chunk_idx| {
                                let start_y = chunk_idx * chunk_size;
                                let end_y = (start_y + chunk_size).min(height);
                                for y in start_y..end_y {
                                    let row_offset = y * width;
                                    let r_curr = &self.image_vec[row_offset..row_offset + width];

                                    for x in 0..width {
                                        if r_curr[x] > th {
                                            acc.push(row_offset + x);
                                        }
                                    }
                                }
                                acc
                            },
                        )
                        .reduce(
                            || Vec::new(),
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        )
                }
            }
            Threshold::Array(arr) => {
                if self.config.binary_open {
                    let chunks = (height.saturating_sub(2) + chunk_size - 1) / chunk_size;

                    (0..chunks)
                        .into_par_iter()
                        .fold(
                            || Vec::with_capacity(128),
                            |mut acc, chunk_idx| {
                                let start_y = 1 + chunk_idx * chunk_size;
                                let end_y = (start_y + chunk_size).min(height - 1);
                                for y in start_y..end_y {
                                    let row_offset = y * width;

                                    // Optimization: Complete bounds-check elimination via exact slice lengths
                                    let r_prev = &self.image_vec[(y - 1) * width..y * width];
                                    let r_curr = &self.image_vec[y * width..(y + 1) * width];
                                    let r_next = &self.image_vec[(y + 1) * width..(y + 2) * width];

                                    let t_prev = &arr[(y - 1) * width..y * width];
                                    let t_curr = &arr[y * width..(y + 1) * width];
                                    let t_next = &arr[(y + 1) * width..(y + 2) * width];

                                    for x in 1..width - 1 {
                                        if r_curr[x] > t_curr[x] {
                                            if r_curr[x - 1] > t_curr[x - 1]
                                                && r_curr[x + 1] > t_curr[x + 1]
                                                && r_prev[x] > t_prev[x]
                                                && r_next[x] > t_next[x]
                                            {
                                                acc.push(row_offset + x);
                                            }
                                        }
                                    }
                                }
                                acc
                            },
                        )
                        .reduce(
                            || Vec::new(),
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        )
                } else {
                    let chunks = (height + chunk_size - 1) / chunk_size;
                    (0..chunks)
                        .into_par_iter()
                        .fold(
                            || Vec::with_capacity(128),
                            |mut acc, chunk_idx| {
                                let start_y = chunk_idx * chunk_size;
                                let end_y = (start_y + chunk_size).min(height);
                                for y in start_y..end_y {
                                    let row_offset = y * width;
                                    let r_curr = &self.image_vec[row_offset..row_offset + width];
                                    let t_curr = &arr[row_offset..row_offset + width];

                                    for x in 0..width {
                                        if r_curr[x] > t_curr[x] {
                                            acc.push(row_offset + x);
                                        }
                                    }
                                }
                                acc
                            },
                        )
                        .reduce(
                            || Vec::new(),
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        )
                }
            }
        };

        // Helper: Binary Dilation (3x3 cross)
        // Instant Dilation Matrix mapped linearly via fast index marking
        self.mask.resize(width * height, false);
        self.mask.fill(false);

        if self.config.binary_open {
            for &i in &eroded_pixels {
                self.mask[i] = true;
                self.mask[i - 1] = true;
                self.mask[i + 1] = true;
                self.mask[i - width] = true;
                self.mask[i + width] = true;
            }
        } else {
            for &i in &eroded_pixels {
                self.mask[i] = true;
            }
        }

        let dbg_mask = if self.config.return_images {
            Some(Array2::from_shape_vec((height, width), self.mask.clone()).unwrap())
        } else {
            None
        };

        let mut extracted = Vec::new();

        // 5. Label regions & 6. Accumulate statistics
        // Helper: 4-Connected Components Labeling & Centered Moments executed in a single pass
        for &seed in &eroded_pixels {
            if !self.mask[seed] {
                continue;
            }

            self.mask[seed] = false;

            let mut area = 1;
            let val = self.image_vec[seed] as f64;
            let mut sum = val;
            let sx = (seed % width) as f64;
            let sy = (seed / width) as f64;

            // Apply Parallel Axis Theorem to accumulate variances in single loop
            let mut sum_x = sx * val;
            let mut sum_y = sy * val;
            let mut sum_xx = sx * sx * val;
            let mut sum_yy = sy * sy * val;
            let mut sum_xy = sx * sy * val;

            self.stack.clear();
            self.stack.push(seed);

            while let Some(idx) = self.stack.pop() {
                let cy = idx / width;
                let cx = idx % width;

                let mut check_push = |ni: usize, nx: f64, ny: f64| {
                    if self.mask[ni] {
                        self.mask[ni] = false;
                        area += 1;
                        let v = self.image_vec[ni] as f64;
                        sum += v;
                        sum_x += nx * v;
                        sum_y += ny * v;
                        sum_xx += nx * nx * v;
                        sum_yy += ny * ny * v;
                        sum_xy += nx * ny * v;
                        self.stack.push(ni);
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

            if let Some(min_a) = self.config.min_area {
                if area < min_a {
                    continue;
                }
            }
            if let Some(max_a) = self.config.max_area {
                if area > max_a {
                    continue;
                }
            }
            if let Some(min_s) = self.config.min_sum {
                if sum < min_s {
                    continue;
                }
            }
            if let Some(max_s) = self.config.max_sum {
                if sum > max_s {
                    continue;
                }
            }
            if sum == 0.0 {
                continue;
            }

            let m1_x = sum_x / sum;
            let m1_y = sum_y / sum;

            let m2_xx = (sum_xx / sum - m1_x * m1_x).max(0.0);
            let m2_yy = (sum_yy / sum - m1_y * m1_y).max(0.0);
            let m2_xy = sum_xy / sum - m1_x * m1_y;

            let diff = m2_xx - m2_yy;
            let root = (diff * diff + 4.0 * m2_xy * m2_xy).sqrt();
            let major = (2.0 * (m2_xx + m2_yy + root)).sqrt();
            let minor = (2.0 * 0f64.max(m2_xx + m2_yy - root)).sqrt();
            let axis_ratio = major / minor.max(1e-9);

            if let Some(max_ar) = self.config.max_axis_ratio {
                if axis_ratio > max_ar || minor <= 0.0 {
                    continue;
                }
            }

            extracted.push(CentroidResult {
                y: m1_y + 0.5,
                x: m1_x + 0.5,
                sum,
                area,
                m2_xx,
                m2_yy,
                m2_xy,
                axis_ratio,
            });
        }

        // 7. Sort
        extracted.sort_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap_or(Ordering::Equal));
        if let Some(max_ret) = self.config.max_returned {
            extracted.truncate(max_ret);
        }

        // 8. Centroid Window
        if let Some(mut window) = self.config.centroid_window {
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
                    let row_start = (o_y + wy) * width + o_x;
                    let row_slice = &self.image_vec[row_start..row_start + window];
                    let wy_f = wy as f64 + 0.5;

                    for (wx, &v) in row_slice.iter().enumerate() {
                        let val = v as f64;
                        img_sum += val;
                        sum_xc += val * (wx as f64 + 0.5);
                        sum_yc += val * wy_f;
                    }
                }

                if img_sum > 0.0 {
                    centroid.x = sum_xc / img_sum + o_x as f64;
                    centroid.y = sum_yc / img_sum + o_y as f64;
                }
            }
        }

        // 9. Revert effects of crop and downsample
        for centroid in &mut extracted {
            if let Some(ds) = self.config.downsample {
                centroid.x *= ds as f64;
                centroid.y *= ds as f64;
            }
            if self.config.crop.is_some() {
                centroid.x += final_offs_w as f64;
                centroid.y += final_offs_h as f64;
            }
        }

        ExtractionResult {
            centroids: extracted,
            debug_images: if self.config.return_images {
                Some(DebugImages {
                    cropped_and_downsampled: dbg_cropped.unwrap(),
                    removed_background: dbg_bg_sub.unwrap(),
                    binary_mask: dbg_mask.unwrap(),
                })
            } else {
                None
            },
        }
    }
}
