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

use ndarray::{Array2, ArrayBase, Data, Ix2, s};
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
pub struct ExtractOptions {
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

impl Default for ExtractOptions {
    fn default() -> Self {
        ExtractOptions {
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

/// Trait to allow our highly optimized spatial filters to transparently operate
/// on either `u8` (zero-copy ingestion) or `f32` (standard/downsampled modes)
pub trait ToF32 {
    fn to_f32(self) -> f32;
}

impl ToF32 for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl ToF32 for u8 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// Helper: 2D uniform filter (box blur) - generic over source type
fn fast_box_blur_2d<T: Copy + ToF32 + Sync + Send>(
    src: &[T],
    scratch: &mut [f32],
    out: &mut [f32],
    w: usize,
    h: usize,
    size: usize,
) {
    let rad = size / 2;
    let area = (size * size) as f32;

    scratch
        .par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(s_row, i_row)| {
            assert!(i_row.len() >= w);
            assert!(s_row.len() >= w);

            let mut sum = 0.0_f32;
            if w > 2 * rad {
                unsafe {
                    for x in 0..=rad {
                        sum += i_row.get_unchecked(x).to_f32();
                    }
                    for x in 1..=rad {
                        sum += i_row.get_unchecked(x - 1).to_f32();
                    }
                    *s_row.get_unchecked_mut(0) = sum;

                    for x in 1..=rad {
                        sum += i_row.get_unchecked(x + rad).to_f32()
                            - i_row.get_unchecked(rad - x).to_f32();
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (rad + 1)..(w - rad) {
                        sum += i_row.get_unchecked(x + rad).to_f32()
                            - i_row.get_unchecked(x - rad - 1).to_f32();
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (w - rad)..w {
                        let add_px = if x + rad >= w {
                            2 * w - 1 - (x + rad)
                        } else {
                            x + rad
                        };
                        sum += i_row.get_unchecked(add_px).to_f32()
                            - i_row.get_unchecked(x - rad - 1).to_f32();
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                }
            } else {
                let rad_i = rad as isize;
                for x in -rad_i..=rad_i {
                    let px = if x < 0 {
                        (-x - 1) as usize
                    } else if x >= w as isize {
                        (2 * w as isize - 1 - x) as usize
                    } else {
                        x as usize
                    };
                    sum += i_row[px].to_f32();
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
                    sum += i_row[add_px].to_f32() - i_row[sub_px].to_f32();
                    s_row[x] = sum;
                }
            }
        });

    let strip_width = 128;
    let num_strips = (w + strip_width - 1) / strip_width;
    let out_ptr = out.as_mut_ptr() as usize;

    (0..num_strips).into_par_iter().for_each(|strip_idx| {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, w * h) };
        let x_start = strip_idx * strip_width;
        let x_end = (x_start + strip_width).min(w);
        let current_strip_width = x_end - x_start;
        let mut col_sums_buf = [0.0_f32; 128];
        let col_sums = &mut col_sums_buf[..current_strip_width];

        for y in (-(rad as isize))..=(rad as isize) {
            let py = if y < 0 {
                (-y - 1) as usize
            } else if y >= h as isize {
                (2 * h as isize - 1 - y) as usize
            } else {
                y as usize
            };
            let s_row = &scratch[py * w + x_start..py * w + x_end];
            for x in 0..current_strip_width {
                col_sums[x] += s_row[x];
            }
        }

        let inv_area = 1.0 / area;
        let out_row = &mut out_slice[x_start..x_start + current_strip_width];
        for (o, c) in out_row.iter_mut().zip(col_sums.iter()) {
            *o = *c * inv_area;
        }

        for y in 1..h {
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

            let add_row = &scratch[add_py * w + x_start..add_py * w + x_end];
            let sub_row = &scratch[sub_py * w + x_start..sub_py * w + x_end];
            let o_row_start = y * w + x_start;
            let out_row = &mut out_slice[o_row_start..o_row_start + current_strip_width];

            for (((o, c), a), s) in out_row
                .iter_mut()
                .zip(col_sums.iter_mut())
                .zip(add_row.iter())
                .zip(sub_row.iter())
            {
                *c += *a - *s;
                *o = *c * inv_area;
            }
        }
    });
}

/// Specialized integer blur: Reads u8, scratches to u16 (50% less bandwidth), outputs scaled subtracted i16 image.
/// Preserves 7 bits of sub-pixel fractional precision by scaling subtracted values by 128.0 and applying proper float rounding before cast.
fn fast_box_blur_2d_u8_to_i16_scaled(
    src: &[u8],
    scratch: &mut [u16],
    out: &mut [i16],
    w: usize,
    h: usize,
    size: usize,
) -> f64 {
    assert!(size <= 255, "Filter size too large for u16 scratch buffer");
    let rad = size / 2;
    let inv_area = 1.0 / (size * size) as f32;

    // Horizontal pass: u8 -> u16
    scratch
        .par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(s_row, i_row)| {
            let mut sum = 0_u16;
            if w > 2 * rad {
                unsafe {
                    for x in 0..=rad {
                        sum += *i_row.get_unchecked(x) as u16;
                    }
                    for x in 1..=rad {
                        sum += *i_row.get_unchecked(x - 1) as u16;
                    }
                    *s_row.get_unchecked_mut(0) = sum;

                    for x in 1..=rad {
                        sum = sum + (*i_row.get_unchecked(x + rad) as u16)
                            - (*i_row.get_unchecked(rad - x) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (rad + 1)..(w - rad) {
                        sum = sum + (*i_row.get_unchecked(x + rad) as u16)
                            - (*i_row.get_unchecked(x - rad - 1) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (w - rad)..w {
                        let add_px = if x + rad >= w {
                            2 * w - 1 - (x + rad)
                        } else {
                            x + rad
                        };
                        sum = sum + (*i_row.get_unchecked(add_px) as u16)
                            - (*i_row.get_unchecked(x - rad - 1) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                }
            } else {
                let rad_i = rad as isize;
                for x in -rad_i..=rad_i {
                    let px = if x < 0 {
                        (-x - 1) as usize
                    } else if x >= w as isize {
                        (2 * w as isize - 1 - x) as usize
                    } else {
                        x as usize
                    };
                    sum += i_row[px] as u16;
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
                    sum = sum + (i_row[add_px] as u16) - (i_row[sub_px] as u16);
                    s_row[x] = sum;
                }
            }
        });

    let strip_width = 128;
    let num_strips = (w + strip_width - 1) / strip_width;
    let out_ptr = out.as_mut_ptr() as usize;

    // Vertical pass: Reads u16 -> Math in f32 -> Scales by 128x to preserve sub-pixel accuracy -> Writes fused i16
    (0..num_strips)
        .into_par_iter()
        .map(|strip_idx| {
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i16, w * h) };
            let x_start = strip_idx * strip_width;
            let x_end = (x_start + strip_width).min(w);
            let current_strip_width = x_end - x_start;
            let mut col_sums = [0_u32; 128];

            for y in (-(rad as isize))..=(rad as isize) {
                let py = if y < 0 {
                    (-y - 1) as usize
                } else if y >= h as isize {
                    (2 * h as isize - 1 - y) as usize
                } else {
                    y as usize
                };
                let s_row = &scratch[py * w + x_start..py * w + x_end];
                for x in 0..current_strip_width {
                    col_sums[x] += s_row[x] as u32;
                }
            }

            let mut thread_sq_sum = 0.0_f64;

            {
                let out_row = &mut out_slice[x_start..x_start + current_strip_width];
                let src_row = &src[x_start..x_start + current_strip_width];
                for (x, o) in out_row.iter_mut().enumerate() {
                    let bg = (col_sums[x] as f32) * inv_area;
                    let val_f32 = (src_row[x] as f32) - bg;
                    // `.round()` better quantizes the value instead of truncating towards zero.
                    // This better preserves fractional sub-pixel intensity for the centroid solver.
                    *o = (val_f32 * 128.0).round() as i16;
                    thread_sq_sum += (val_f32 as f64) * (val_f32 as f64);
                }
            }

            for y in 1..h {
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

                let add_row = &scratch[add_py * w + x_start..add_py * w + x_end];
                let sub_row = &scratch[sub_py * w + x_start..sub_py * w + x_end];
                let o_row_start = y * w + x_start;

                let out_row = &mut out_slice[o_row_start..o_row_start + current_strip_width];
                let src_row = &src[o_row_start..o_row_start + current_strip_width];

                for x in 0..current_strip_width {
                    col_sums[x] = col_sums[x] + (add_row[x] as u32) - (sub_row[x] as u32);
                    let bg = (col_sums[x] as f32) * inv_area;
                    let val_f32 = (src_row[x] as f32) - bg;
                    // `.round()` better quantizes the value instead of truncating towards zero.
                    out_row[x] = (val_f32 * 128.0).round() as i16;
                    thread_sq_sum += (val_f32 as f64) * (val_f32 as f64);
                }
            }
            thread_sq_sum
        })
        .sum()
}

/// Fused fast blur: Reads u8, scratches to u16 (saving memory bandwidth), outputs f32 subtracted image.
/// This removes memory copying overhead of the standard pipeline while preserving mathematical bit-for-bit identity.
fn fast_box_blur_2d_u8_to_f32_subtracted(
    src: &[u8],
    scratch: &mut [u16],
    out: &mut [f32],
    w: usize,
    h: usize,
    size: usize,
) -> f64 {
    assert!(size <= 255, "Filter size too large for u16 scratch buffer");
    let rad = size / 2;
    let inv_area = 1.0 / (size * size) as f32;

    // Horizontal pass: u8 -> u16 (Slashes memory write bandwidth by 50% vs standard f32)
    scratch
        .par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(s_row, i_row)| {
            let mut sum = 0_u16;
            if w > 2 * rad {
                unsafe {
                    for x in 0..=rad {
                        sum += *i_row.get_unchecked(x) as u16;
                    }
                    for x in 1..=rad {
                        sum += *i_row.get_unchecked(x - 1) as u16;
                    }
                    *s_row.get_unchecked_mut(0) = sum;

                    for x in 1..=rad {
                        sum = sum + (*i_row.get_unchecked(x + rad) as u16)
                            - (*i_row.get_unchecked(rad - x) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (rad + 1)..(w - rad) {
                        sum = sum + (*i_row.get_unchecked(x + rad) as u16)
                            - (*i_row.get_unchecked(x - rad - 1) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                    for x in (w - rad)..w {
                        let add_px = if x + rad >= w {
                            2 * w - 1 - (x + rad)
                        } else {
                            x + rad
                        };
                        sum = sum + (*i_row.get_unchecked(add_px) as u16)
                            - (*i_row.get_unchecked(x - rad - 1) as u16);
                        *s_row.get_unchecked_mut(x) = sum;
                    }
                }
            } else {
                let rad_i = rad as isize;
                for x in -rad_i..=rad_i {
                    let px = if x < 0 {
                        (-x - 1) as usize
                    } else if x >= w as isize {
                        (2 * w as isize - 1 - x) as usize
                    } else {
                        x as usize
                    };
                    sum += i_row[px] as u16;
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
                    sum = sum + (i_row[add_px] as u16) - (i_row[sub_px] as u16);
                    s_row[x] = sum;
                }
            }
        });

    let strip_width = 128;
    let num_strips = (w + strip_width - 1) / strip_width;
    let out_ptr = out.as_mut_ptr() as usize;

    // Vertical pass: Reads u16 -> Math in f32 -> Writes f32 subtracted image directly to out_buffer.
    (0..num_strips)
        .into_par_iter()
        .map(|strip_idx| {
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, w * h) };
            let x_start = strip_idx * strip_width;
            let x_end = (x_start + strip_width).min(w);
            let current_strip_width = x_end - x_start;
            let mut col_sums = [0_u32; 128];

            for y in (-(rad as isize))..=(rad as isize) {
                let py = if y < 0 {
                    (-y - 1) as usize
                } else if y >= h as isize {
                    (2 * h as isize - 1 - y) as usize
                } else {
                    y as usize
                };
                let s_row = &scratch[py * w + x_start..py * w + x_end];
                for x in 0..current_strip_width {
                    col_sums[x] += s_row[x] as u32;
                }
            }

            let mut thread_sq_sum = 0.0_f64;

            {
                let out_row = &mut out_slice[x_start..x_start + current_strip_width];
                let src_row = &src[x_start..x_start + current_strip_width];
                for (x, o) in out_row.iter_mut().enumerate() {
                    let bg = (col_sums[x] as f32) * inv_area;
                    let val_f32 = (src_row[x] as f32) - bg;
                    *o = val_f32;
                    thread_sq_sum += (val_f32 as f64) * (val_f32 as f64);
                }
            }

            for y in 1..h {
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

                let add_row = &scratch[add_py * w + x_start..add_py * w + x_end];
                let sub_row = &scratch[sub_py * w + x_start..sub_py * w + x_end];
                let o_row_start = y * w + x_start;

                let out_row = &mut out_slice[o_row_start..o_row_start + current_strip_width];
                let src_row = &src[o_row_start..o_row_start + current_strip_width];

                for x in 0..current_strip_width {
                    col_sums[x] = col_sums[x] + (add_row[x] as u32) - (sub_row[x] as u32);
                    let bg = (col_sums[x] as f32) * inv_area;
                    let val_f32 = (src_row[x] as f32) - bg;
                    out_row[x] = val_f32;
                    thread_sq_sum += (val_f32 as f64) * (val_f32 as f64);
                }
            }
            thread_sq_sum
        })
        .sum()
}

/// Helper: 2D median filter using raw slices - generic over source type
fn fast_median_filter_2d<T: Copy + ToF32 + Sync + Send>(
    src: &[T],
    out: &mut [f32],
    w: usize,
    h: usize,
    size: usize,
) {
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
                        unsafe {
                            window[idx] = src
                                .get_unchecked((sy as usize) * w + (sx as usize))
                                .to_f32();
                        }
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
    // Primary buffers for the standard f32 pipeline
    image_vec: Vec<f32>,
    scratch: Vec<f32>,
    median_scratch: Vec<f32>,
    std_img: Vec<f32>,

    // Fast fused pipeline optimizations
    scratch_u16: Vec<u16>,
    median_scratch_u8: Vec<u8>,
    contiguous_u8: Vec<u8>,

    // i16 integer pipeline buffers
    image_i16: Vec<i16>,

    // Shared state variables
    mask: Vec<bool>,
    stack: Vec<usize>, // Tiny stack size keeps the L1 cache hot on Pi (bandwidth constrained)
}

impl Default for Extractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Extractor {
    pub fn new() -> Self {
        Self {
            image_vec: Vec::new(),
            scratch: Vec::new(),
            median_scratch: Vec::new(),
            std_img: Vec::new(),

            scratch_u16: Vec::new(),
            median_scratch_u8: Vec::new(),
            contiguous_u8: Vec::new(),

            image_i16: Vec::new(),

            mask: Vec::new(),
            stack: Vec::with_capacity(1024),
        }
    }

    /// Primary extraction method for standard `f32` floating-point image inputs.
    pub fn extract<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = f32>,
    {
        // 1. Crop and downsample
        // Note: Cropping is applied before downsampling.
        let (full_height, full_width) = input_image.dim();

        let (mut height, mut width, offs_h_isize, offs_w_isize) = match options.crop {
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

        if options.crop.is_some() {
            let divisor = options.downsample.unwrap_or(2);
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

        if let Some(ds) = options.downsample {
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
                        row[x] = sum;
                    }
                });
        } else {
            self.image_vec.resize(height * width, 0.0);
            if let Some(s) = cropped.as_slice() {
                self.image_vec.copy_from_slice(s);
            } else {
                for (out_row, in_row) in self
                    .image_vec
                    .chunks_exact_mut(width)
                    .zip(cropped.axis_iter(ndarray::Axis(0)))
                {
                    out_row.copy_from_slice(in_row.as_slice().unwrap());
                }
            }
        }

        // Handoff to shared core execution pipeline
        self.extract_f32_pipeline(width, height, final_offs_w, final_offs_h, options)
    }

    /// Standard u8 pipeline: Handles downsampling internally. Utilizes f32 pipeline for 1x resolution. Removes image_vec copy overhead while identically matching f32 math accuracy.
    pub fn extract_u8<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = u8>,
    {
        // Crop and bounds logic (identical mathematical behavior)
        let (full_height, full_width) = input_image.dim();

        let (mut height, mut width, offs_h_isize, offs_w_isize) = match options.crop {
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

        if options.crop.is_some() {
            let divisor = options.downsample.unwrap_or(2);
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

        if let Some(ds) = options.downsample {
            // Late-promotion path (for downsampled sources)
            let out_height = height / ds;
            let out_width = width / ds;

            self.image_vec.resize(out_height * out_width, 0.0);

            if let Some(s) = cropped.as_slice() {
                // Manually unrolled 2x and 4x paths.
                // Removes inner variable loops, allowing the compiler to use direct SIMD load-adds.
                let w = width;
                if ds == 2 {
                    self.image_vec
                        .par_chunks_exact_mut(out_width)
                        .enumerate()
                        .for_each(|(out_y, row)| {
                            let start_y = out_y * 2;
                            for out_x in 0..out_width {
                                let start_x = out_x * 2;
                                unsafe {
                                    let r1 = start_y * w + start_x;
                                    let r2 = (start_y + 1) * w + start_x;
                                    let sum = *s.get_unchecked(r1) as u32
                                        + *s.get_unchecked(r1 + 1) as u32
                                        + *s.get_unchecked(r2) as u32
                                        + *s.get_unchecked(r2 + 1) as u32;
                                    *row.get_unchecked_mut(out_x) = sum as f32;
                                }
                            }
                        });
                } else if ds == 4 {
                    self.image_vec
                        .par_chunks_exact_mut(out_width)
                        .enumerate()
                        .for_each(|(out_y, row)| {
                            let start_y = out_y * 4;
                            for out_x in 0..out_width {
                                let start_x = out_x * 4;
                                let mut sum = 0u32;
                                unsafe {
                                    for dy in 0..4 {
                                        let r = (start_y + dy) * w + start_x;
                                        sum += *s.get_unchecked(r) as u32
                                            + *s.get_unchecked(r + 1) as u32
                                            + *s.get_unchecked(r + 2) as u32
                                            + *s.get_unchecked(r + 3) as u32;
                                    }
                                    *row.get_unchecked_mut(out_x) = sum as f32;
                                }
                            }
                        });
                } else {
                    self.image_vec
                        .par_chunks_exact_mut(out_width)
                        .enumerate()
                        .for_each(|(out_y, row)| {
                            let start_y = out_y * ds;
                            for out_x in 0..out_width {
                                let mut sum: u32 = 0;
                                let start_x = out_x * ds;
                                for dy in 0..ds {
                                    let row_offset = (start_y + dy) * w;
                                    for dx in 0..ds {
                                        unsafe {
                                            sum +=
                                                *s.get_unchecked(row_offset + start_x + dx) as u32;
                                        }
                                    }
                                }
                                row[out_x] = sum as f32;
                            }
                        });
                }
            } else {
                // Fallback for non-contiguous views
                self.image_vec
                    .par_chunks_exact_mut(out_width)
                    .enumerate()
                    .for_each(|(y, row)| {
                        for x in 0..out_width {
                            let mut sum: u32 = 0;
                            for dy in 0..ds {
                                for dx in 0..ds {
                                    sum += cropped[[y * ds + dy, x * ds + dx]] as u32;
                                }
                            }
                            row[x] = sum as f32;
                        }
                    });
            }

            self.extract_f32_pipeline(out_width, out_height, final_offs_w, final_offs_h, options)
        } else {
            // Fast promotion path (1x mode)
            // Route to standard float pipeline for local sigmas/medians that are too complex to fuse
            if matches!(
                options.sigma_mode,
                SigmaMode::LocalMedianAbs | SigmaMode::LocalRootSquare
            ) || matches!(options.bg_sub_mode, Some(BgSubMode::LocalMedian))
            {
                self.image_vec.resize(height * width, 0.0);
                if let Some(s) = cropped.as_slice() {
                    self.image_vec
                        .par_chunks_mut(4096)
                        .zip(s.par_chunks(4096))
                        .for_each(|(out_c, in_c)| {
                            for (o, &i) in out_c.iter_mut().zip(in_c.iter()) {
                                *o = i as f32;
                            }
                        });
                } else {
                    for (out_row, in_row) in self
                        .image_vec
                        .chunks_exact_mut(width)
                        .zip(cropped.axis_iter(ndarray::Axis(0)))
                    {
                        for (out_val, &in_val) in out_row.iter_mut().zip(in_row.iter()) {
                            *out_val = in_val as f32;
                        }
                    }
                }
                self.extract_f32_pipeline(width, height, final_offs_w, final_offs_h, options)
            } else {
                // Execute highly optimized fused f32 pipeline
                if let Some(s) = cropped.as_slice() {
                    self.extract_fused_f32_pipeline(
                        s,
                        width,
                        height,
                        final_offs_w,
                        final_offs_h,
                        options,
                    )
                } else {
                    let mut u8_buffer = std::mem::take(&mut self.contiguous_u8);
                    u8_buffer.resize(width * height, 0);
                    for (out_row, in_row) in u8_buffer
                        .chunks_exact_mut(width)
                        .zip(cropped.axis_iter(ndarray::Axis(0)))
                    {
                        out_row.copy_from_slice(in_row.as_slice().unwrap());
                    }
                    let result = self.extract_fused_f32_pipeline(
                        &u8_buffer,
                        width,
                        height,
                        final_offs_w,
                        final_offs_h,
                        options,
                    );
                    self.contiguous_u8 = u8_buffer;
                    result
                }
            }
        }
    }

    /// Fast 1x pipeline utilizing fixed-point math inside CPU integer registers.
    /// Scaled by 128.0 and rounded to preserve fractional sub-pixel intensity at maximum speed.
    /// Small tradeoff in accuracy for better performance.
    pub fn extract_u8_fast<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = u8>,
    {
        if options.downsample.is_some() {
            // Downsampled loops are already hyper-optimized in extract_u8
            return self.extract_u8(input_image, options);
        }

        // Standard float pipeline is best suited for complex localized median/sigma evaluations
        if matches!(
            options.sigma_mode,
            SigmaMode::LocalMedianAbs | SigmaMode::LocalRootSquare
        ) || matches!(options.bg_sub_mode, Some(BgSubMode::LocalMedian))
        {
            return self.extract_u8(input_image, options);
        }

        let (full_height, full_width) = input_image.dim();
        let (mut height, mut width, offs_h_isize, offs_w_isize) = match options.crop {
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

        if options.crop.is_some() {
            let divisor = options.downsample.unwrap_or(2);
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

        // 1x Integer pipeline (half memory bandwidth route via i16)
        if let Some(s) = cropped.as_slice() {
            // If memory is already contiguous, pass the slice directly
            self.extract_i16_pipeline(s, width, height, final_offs_w, final_offs_h, options)
        } else {
            // Temporarily detach the contiguous_u8 buffer to satisfy the borrow checker
            let mut u8_buffer = std::mem::take(&mut self.contiguous_u8);

            u8_buffer.resize(width * height, 0);
            for (out_row, in_row) in u8_buffer
                .chunks_exact_mut(width)
                .zip(cropped.axis_iter(ndarray::Axis(0)))
            {
                out_row.copy_from_slice(in_row.as_slice().unwrap());
            }

            // Pass the detached buffer so we don't hold an immutable borrow on self
            let result = self.extract_i16_pipeline(
                &u8_buffer,
                width,
                height,
                final_offs_w,
                final_offs_h,
                options,
            );

            // Return the buffer back to the Extractor struct for reuse
            self.contiguous_u8 = u8_buffer;

            result
        }
    }

    /// Dedicated fixed-point pipeline: Halves memory bandwidth using `i16` arrays while
    /// scaling subtracted intensities by `128x` to preserve 7 bits of `f32` sub-pixel precision.
    fn extract_i16_pipeline(
        &mut self,
        src: &[u8],
        width: usize,
        height: usize,
        final_offs_w: usize,
        final_offs_h: usize,
        options: ExtractOptions,
    ) -> ExtractionResult {
        let dbg_cropped = if options.return_images {
            Some(Array2::from_shape_fn((height, width), |(y, x)| {
                src[y * width + x] as f32
            }))
        } else {
            None
        };

        // 2. Subtract background directly into i16 buffer
        self.image_i16.resize(width * height, 0);
        let sum_sq_global: f64 = if let Some(mode) = options.bg_sub_mode {
            match mode {
                BgSubMode::LocalMean => {
                    self.scratch_u16.resize(width * height, 0);
                    fast_box_blur_2d_u8_to_i16_scaled(
                        src,
                        &mut self.scratch_u16,
                        &mut self.image_i16,
                        width,
                        height,
                        options.filtsize,
                    )
                }
                BgSubMode::GlobalMean => {
                    let sum: u64 = src.par_iter().map(|&v| v as u64).sum();
                    let mean = (sum as f64 / src.len() as f64) as f32;
                    self.image_i16
                        .par_iter_mut()
                        .zip(src.par_iter())
                        .map(|(o, &i)| {
                            let val_f32 = (i as f32) - mean;
                            // .round() prevents truncation-towards-zero, preserving floating point statistics
                            *o = (val_f32 * 128.0).round() as i16;
                            (val_f32 as f64) * (val_f32 as f64)
                        })
                        .sum()
                }
                BgSubMode::GlobalMedian => {
                    self.median_scratch_u8.clear();
                    self.median_scratch_u8.extend_from_slice(src);
                    let mid = self.median_scratch_u8.len() / 2;
                    let (_, &mut median, _) = self
                        .median_scratch_u8
                        .select_nth_unstable_by(mid, |a, b| a.cmp(b));
                    let med = median as f32;

                    self.image_i16
                        .par_iter_mut()
                        .zip(src.par_iter())
                        .map(|(o, &i)| {
                            let val_f32 = (i as f32) - med;
                            *o = (val_f32 * 128.0).round() as i16;
                            (val_f32 as f64) * (val_f32 as f64)
                        })
                        .sum()
                }
                BgSubMode::LocalMedian => unreachable!(),
            }
        } else {
            self.image_i16
                .par_iter_mut()
                .zip(src.par_iter())
                .map(|(o, &i)| {
                    *o = (i as i16) * 128; // Safe from overflow (255 * 128 = 32,640)
                    (i as f64) * (i as f64)
                })
                .sum()
        };

        let dbg_bg_sub = if options.return_images {
            Some(Array2::from_shape_fn((height, width), |(y, x)| {
                (self.image_i16[y * width + x] as f32) * (1.0 / 128.0)
            }))
        } else {
            None
        };

        // 3. Find noise standard deviation to threshold
        let threshold: i16 = if let Some(th) = options.image_th {
            (th * 128.0).round() as i16
        } else {
            match options.sigma_mode {
                SigmaMode::GlobalRootSquare => {
                    let mean_sq = (sum_sq_global / (height * width) as f64) as f32;
                    (mean_sq.max(0.0).sqrt() * options.sigma * 128.0).round() as i16
                }
                SigmaMode::GlobalMedianAbs
                | SigmaMode::LocalMedianAbs
                | SigmaMode::LocalRootSquare => {
                    // Fallback handled smoothly at entry block
                    unreachable!()
                }
            }
        };

        // 4. Threshold to find binary mask
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
                            let p_prev = self.image_i16[(y - 1) * width..y * width].as_ptr();
                            let p_curr = self.image_i16[y * width..(y + 1) * width].as_ptr();
                            let p_next = self.image_i16[(y + 1) * width..(y + 2) * width].as_ptr();

                            for x in 1..width - 1 {
                                unsafe {
                                    // The scalar `threshold` is already scaled by 128x allowing pure int branching
                                    if *p_curr.add(x) > threshold {
                                        if *p_curr.add(x - 1) > threshold
                                            && *p_curr.add(x + 1) > threshold
                                            && *p_prev.add(x) > threshold
                                            && *p_next.add(x) > threshold
                                        {
                                            acc.push(row_offset + x);
                                        }
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
                            let r_curr = &self.image_i16[row_offset..row_offset + width];
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

        // Binary dilation
        self.mask.resize(width * height, false);
        self.mask.fill(false);

        if options.binary_open {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                    *mask_ptr.add(i - 1) = true;
                    *mask_ptr.add(i + 1) = true;
                    *mask_ptr.add(i - width) = true;
                    *mask_ptr.add(i + width) = true;
                }
            }
        } else {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                }
            }
        }

        let dbg_mask = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.mask.clone()).unwrap())
        } else {
            None
        };

        let mut extracted = Vec::new();

        // 5. Label regions & 6. Accumulate statistics
        for &seed in &eroded_pixels {
            if !self.mask[seed] {
                continue;
            }

            self.mask[seed] = false;

            let mut area = 1;
            // De-scale the 128x multiplied value back to fractional float bounds
            let val = (self.image_i16[seed] as f64) * (1.0 / 128.0);
            let mut sum = val;
            let sx = (seed % width) as f64;
            let sy = (seed / width) as f64;

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

                let mut check_push = |ni: usize, nx: f64, ny: f64| unsafe {
                    if *self.mask.get_unchecked(ni) {
                        *self.mask.get_unchecked_mut(ni) = false;
                        area += 1;
                        let v = (*self.image_i16.get_unchecked(ni) as f64) * (1.0 / 128.0);
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

            if let Some(min_a) = options.min_area {
                if area < min_a {
                    continue;
                }
            }
            if let Some(max_a) = options.max_area {
                if area > max_a {
                    continue;
                }
            }
            if let Some(min_s) = options.min_sum {
                if sum < min_s {
                    continue;
                }
            }
            if let Some(max_s) = options.max_sum {
                if sum > max_s {
                    continue;
                }
            }
            if sum == 0.0 {
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

            if let Some(max_ar) = options.max_axis_ratio {
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
        if let Some(max_ret) = options.max_returned {
            extracted.truncate(max_ret);
        }

        // 8. Centroid window
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
                    let row_start = (o_y + wy) * width + o_x;
                    let row_slice = &self.image_i16[row_start..row_start + window];
                    let wy_f = wy as f64 + 0.5;

                    for (wx, &v) in row_slice.iter().enumerate() {
                        let val = (v as f64) * (1.0 / 128.0);
                        img_sum += val;
                        sum_xc += val * (wx as f64 + 0.5);
                        sum_yc += val * wy_f;
                    }
                }

                if img_sum > 0.0 {
                    let inv_img_sum = 1.0 / img_sum;
                    centroid.x = sum_xc * inv_img_sum + o_x as f64;
                    centroid.y = sum_yc * inv_img_sum + o_y as f64;
                }
            }
        }

        // 9. Revert effects of crop and downsample
        for centroid in &mut extracted {
            if options.crop.is_some() {
                centroid.x += final_offs_w as f64;
                centroid.y += final_offs_h as f64;
            }
        }

        ExtractionResult {
            centroids: extracted,
            debug_images: if options.return_images {
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

    /// Fused f32 pipeline: Provides 100% exact bit-for-bit math identical to the standard f32 pipeline,
    /// but slashes memory traffic by computing the subtraction on the fly.
    fn extract_fused_f32_pipeline(
        &mut self,
        src: &[u8],
        width: usize,
        height: usize,
        final_offs_w: usize,
        final_offs_h: usize,
        options: ExtractOptions,
    ) -> ExtractionResult {
        let dbg_cropped = if options.return_images {
            Some(Array2::from_shape_fn((height, width), |(y, x)| {
                src[y * width + x] as f32
            }))
        } else {
            None
        };

        // 2. Subtract background directly into the native f32 buffer
        self.image_vec.resize(width * height, 0.0);

        let sum_sq_global: f64 = if let Some(mode) = options.bg_sub_mode {
            match mode {
                BgSubMode::LocalMean => {
                    self.scratch_u16.resize(width * height, 0);
                    // One pass reads u8, blurs using lightweight u16, outputs f32 subtraction.
                    fast_box_blur_2d_u8_to_f32_subtracted(
                        src,
                        &mut self.scratch_u16,
                        &mut self.image_vec,
                        width,
                        height,
                        options.filtsize,
                    )
                }
                BgSubMode::GlobalMean => {
                    let sum: u64 = src.par_iter().map(|&v| v as u64).sum();
                    let mean = (sum as f64 / src.len() as f64) as f32;
                    self.image_vec
                        .par_iter_mut()
                        .zip(src.par_iter())
                        .map(|(o, &i)| {
                            let val_f32 = (i as f32) - mean;
                            *o = val_f32;
                            (val_f32 as f64) * (val_f32 as f64)
                        })
                        .sum()
                }
                BgSubMode::GlobalMedian => {
                    self.median_scratch_u8.clear();
                    self.median_scratch_u8.extend_from_slice(src);
                    let mid = self.median_scratch_u8.len() / 2;
                    let (_, &mut median, _) = self
                        .median_scratch_u8
                        .select_nth_unstable_by(mid, |a, b| a.cmp(b));
                    let med = median as f32;

                    self.image_vec
                        .par_iter_mut()
                        .zip(src.par_iter())
                        .map(|(o, &i)| {
                            let val_f32 = (i as f32) - med;
                            *o = val_f32;
                            (val_f32 as f64) * (val_f32 as f64)
                        })
                        .sum()
                }
                // LocalMedian safely routed back to standard pipeline at entry point.
                BgSubMode::LocalMedian => unreachable!(),
            }
        } else {
            self.image_vec
                .par_iter_mut()
                .zip(src.par_iter())
                .map(|(o, &i)| {
                    let val_f32 = i as f32;
                    *o = val_f32;
                    (val_f32 as f64) * (val_f32 as f64)
                })
                .sum()
        };

        let dbg_bg_sub = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.image_vec.clone()).unwrap())
        } else {
            None
        };

        // 3. Find noise standard deviation to threshold
        let threshold = if let Some(th) = options.image_th {
            th
        } else {
            match options.sigma_mode {
                SigmaMode::GlobalRootSquare => {
                    let mean_sq = (sum_sq_global / (height * width) as f64) as f32;
                    mean_sq.max(0.0).sqrt() * options.sigma
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
                    median * 1.48 * options.sigma
                }
                // Local noise modes safely routed back to standard pipeline at entry point.
                SigmaMode::LocalMedianAbs | SigmaMode::LocalRootSquare => unreachable!(),
            }
        };

        // 4. Threshold to find binary mask
        // Note: Because self.image_vec contains the same floats as the standard pipeline, this loop is bit-for-bit identical to the original logic, guaranteeing matching centroid output.
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
                            let p_prev = self.image_vec[(y - 1) * width..y * width].as_ptr();
                            let p_curr = self.image_vec[y * width..(y + 1) * width].as_ptr();
                            let p_next = self.image_vec[(y + 1) * width..(y + 2) * width].as_ptr();

                            for x in 1..width - 1 {
                                unsafe {
                                    if *p_curr.add(x) > threshold {
                                        if *p_curr.add(x - 1) > threshold
                                            && *p_curr.add(x + 1) > threshold
                                            && *p_prev.add(x) > threshold
                                            && *p_next.add(x) > threshold
                                        {
                                            acc.push(row_offset + x);
                                        }
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
                            let r_curr = &self.image_vec[row_offset..row_offset + width];
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

        // Binary dilation
        self.mask.resize(width * height, false);
        self.mask.fill(false);

        if options.binary_open {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                    *mask_ptr.add(i - 1) = true;
                    *mask_ptr.add(i + 1) = true;
                    *mask_ptr.add(i - width) = true;
                    *mask_ptr.add(i + width) = true;
                }
            }
        } else {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                }
            }
        }

        let dbg_mask = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.mask.clone()).unwrap())
        } else {
            None
        };

        let mut extracted = Vec::new();

        // 5. Label regions & 6. Accumulate statistics
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

                let mut check_push = |ni: usize, nx: f64, ny: f64| unsafe {
                    if *self.mask.get_unchecked(ni) {
                        *self.mask.get_unchecked_mut(ni) = false;
                        area += 1;
                        let v = *self.image_vec.get_unchecked(ni) as f64;
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

            if let Some(min_a) = options.min_area {
                if area < min_a {
                    continue;
                }
            }
            if let Some(max_a) = options.max_area {
                if area > max_a {
                    continue;
                }
            }
            if let Some(min_s) = options.min_sum {
                if sum < min_s {
                    continue;
                }
            }
            if let Some(max_s) = options.max_sum {
                if sum > max_s {
                    continue;
                }
            }
            if sum == 0.0 {
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

            if let Some(max_ar) = options.max_axis_ratio {
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
        if let Some(max_ret) = options.max_returned {
            extracted.truncate(max_ret);
        }

        // 8. Centroid window
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
                    let inv_img_sum = 1.0 / img_sum;
                    centroid.x = sum_xc * inv_img_sum + o_x as f64;
                    centroid.y = sum_yc * inv_img_sum + o_y as f64;
                }
            }
        }

        // 9. Revert effects of crop and downsample
        for centroid in &mut extracted {
            if options.crop.is_some() {
                centroid.x += final_offs_w as f64;
                centroid.y += final_offs_h as f64;
            }
        }

        ExtractionResult {
            centroids: extracted,
            debug_images: if options.return_images {
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

    /// Shared internal pipeline containing Steps 2-9 for standard f32 inputs or late-promotion formats.
    fn extract_f32_pipeline(
        &mut self,
        width: usize,
        height: usize,
        final_offs_w: usize,
        final_offs_h: usize,
        options: ExtractOptions,
    ) -> ExtractionResult {
        let dbg_cropped = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.image_vec.clone()).unwrap())
        } else {
            None
        };

        // 2. Subtract background
        let sum_sq_global: f64 = if let Some(mode) = options.bg_sub_mode {
            match mode {
                BgSubMode::LocalMean => {
                    self.scratch.resize(width * height, 0.0);
                    let area = (options.filtsize * options.filtsize) as f32;

                    self.scratch
                        .par_chunks_exact_mut(width)
                        .zip(self.image_vec.par_chunks_exact(width))
                        .for_each(|(s_row, i_row)| {
                            let rad = options.filtsize / 2;
                            let mut sum = 0.0_f32;
                            if width > 2 * rad {
                                unsafe {
                                    for x in 0..=rad {
                                        sum += *i_row.get_unchecked(x);
                                    }
                                    for x in 1..=rad {
                                        sum += *i_row.get_unchecked(x - 1);
                                    }
                                    *s_row.get_unchecked_mut(0) = sum;

                                    for x in 1..=rad {
                                        sum += *i_row.get_unchecked(x + rad)
                                            - *i_row.get_unchecked(rad - x);
                                        *s_row.get_unchecked_mut(x) = sum;
                                    }
                                    for x in (rad + 1)..(width - rad) {
                                        sum += *i_row.get_unchecked(x + rad)
                                            - *i_row.get_unchecked(x - rad - 1);
                                        *s_row.get_unchecked_mut(x) = sum;
                                    }
                                    for x in (width - rad)..width {
                                        let add_px = if x + rad >= width {
                                            2 * width - 1 - (x + rad)
                                        } else {
                                            x + rad
                                        };
                                        sum += *i_row.get_unchecked(add_px)
                                            - *i_row.get_unchecked(x - rad - 1);
                                        *s_row.get_unchecked_mut(x) = sum;
                                    }
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

                    let chunk_rows = height.div_ceil(rayon::current_num_threads());
                    let chunk_rows = chunk_rows.max(16);
                    let scratch_ref = &self.scratch;
                    let rad = options.filtsize / 2;
                    let inv_area = 1.0 / area;

                    self.image_vec
                        .par_chunks_mut(chunk_rows * width)
                        .enumerate()
                        .map_init(
                            || vec![0.0_f32; width],
                            |col_sums, (chunk_idx, i_chunk)| {
                                col_sums.fill(0.0);
                                let start_y = chunk_idx * chunk_rows;
                                let end_y = start_y + (i_chunk.len() / width);

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
                                    for (i, c) in i_row.iter_mut().zip(col_sums.iter()) {
                                        let val = *i - (*c * inv_area);
                                        *i = val;
                                        local_sq_sum += (val * val) as f64;
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

                                    let add_row =
                                        &scratch_ref[add_py * width..(add_py + 1) * width];
                                    let sub_row =
                                        &scratch_ref[sub_py * width..(sub_py + 1) * width];
                                    let local_y = y - start_y;
                                    let i_row =
                                        &mut i_chunk[local_y * width..(local_y + 1) * width];

                                    for (((i, c), a), s) in i_row
                                        .iter_mut()
                                        .zip(col_sums.iter_mut())
                                        .zip(add_row.iter())
                                        .zip(sub_row.iter())
                                    {
                                        *c += *a - *s;
                                        let val = *i - (*c * inv_area);
                                        *i = val;
                                        local_sq_sum += (val * val) as f64;
                                    }
                                }
                                local_sq_sum
                            },
                        )
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
                            (*i * *i) as f64
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
                            (*i * *i) as f64
                        })
                        .sum()
                }
                BgSubMode::LocalMedian => {
                    self.scratch.resize(width * height, 0.0);
                    fast_median_filter_2d(
                        self.image_vec.as_slice(),
                        &mut self.scratch,
                        width,
                        height,
                        options.filtsize,
                    );
                    let bg = &self.scratch;
                    self.image_vec
                        .par_iter_mut()
                        .zip(bg.par_iter())
                        .map(|(i, &b)| {
                            *i -= b;
                            (*i * *i) as f64
                        })
                        .sum()
                }
            }
        } else {
            self.image_vec.par_iter().map(|&i| (i * i) as f64).sum()
        };

        let dbg_bg_sub = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.image_vec.clone()).unwrap())
        } else {
            None
        };

        // 3. Find noise standard deviation to threshold
        enum Threshold<'a> {
            Scalar(f32),
            Array(&'a [f32]),
        }

        let threshold = if let Some(th) = options.image_th {
            Threshold::Scalar(th)
        } else {
            match options.sigma_mode {
                SigmaMode::GlobalRootSquare => {
                    let mean_sq = (sum_sq_global / (height * width) as f64) as f32;
                    Threshold::Scalar(mean_sq.max(0.0).sqrt() * options.sigma)
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
                    Threshold::Scalar(median * 1.48 * options.sigma)
                }
                SigmaMode::LocalMedianAbs => {
                    self.std_img.resize(width * height, 0.0);
                    self.scratch.clear();
                    self.scratch.extend(self.image_vec.iter().map(|v| v.abs()));
                    fast_median_filter_2d(
                        self.scratch.as_slice(),
                        &mut self.std_img,
                        width,
                        height,
                        options.filtsize,
                    );
                    self.std_img
                        .par_iter_mut()
                        .for_each(|v| *v *= 1.48 * options.sigma);
                    Threshold::Array(&self.std_img)
                }
                SigmaMode::LocalRootSquare => {
                    self.std_img.resize(width * height, 0.0);
                    self.scratch.resize(width * height, 0.0);

                    self.scratch
                        .par_chunks_mut(4096)
                        .zip(self.image_vec.par_chunks(4096))
                        .for_each(|(s_c, i_c)| {
                            for (s, &i) in s_c.iter_mut().zip(i_c.iter()) {
                                *s = i * i;
                            }
                        });

                    self.median_scratch.resize(width * height, 0.0);
                    fast_box_blur_2d(
                        self.scratch.as_slice(),
                        &mut self.median_scratch,
                        &mut self.std_img,
                        width,
                        height,
                        options.filtsize,
                    );
                    self.std_img
                        .par_iter_mut()
                        .for_each(|v| *v = v.max(0.0).sqrt() * options.sigma);
                    Threshold::Array(&self.std_img)
                }
            }
        };

        // 4. Threshold to find binary mask
        let chunk_size = (height / rayon::current_num_threads()).max(64);

        let eroded_pixels: Vec<usize> = match threshold {
            Threshold::Scalar(th) => {
                if options.binary_open {
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

                                    let p_prev =
                                        self.image_vec[(y - 1) * width..y * width].as_ptr();
                                    let p_curr =
                                        self.image_vec[y * width..(y + 1) * width].as_ptr();
                                    let p_next =
                                        self.image_vec[(y + 1) * width..(y + 2) * width].as_ptr();

                                    for x in 1..width - 1 {
                                        unsafe {
                                            if *p_curr.add(x) > th {
                                                if *p_curr.add(x - 1) > th
                                                    && *p_curr.add(x + 1) > th
                                                    && *p_prev.add(x) > th
                                                    && *p_next.add(x) > th
                                                {
                                                    acc.push(row_offset + x);
                                                }
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
                        .reduce(Vec::new, |mut a, mut b| {
                            a.append(&mut b);
                            a
                        })
                }
            }
            Threshold::Array(arr) => {
                if options.binary_open {
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

                                    let p_prev =
                                        self.image_vec[(y - 1) * width..y * width].as_ptr();
                                    let p_curr =
                                        self.image_vec[y * width..(y + 1) * width].as_ptr();
                                    let p_next =
                                        self.image_vec[(y + 1) * width..(y + 2) * width].as_ptr();

                                    let t_prev = arr[(y - 1) * width..y * width].as_ptr();
                                    let t_curr = arr[y * width..(y + 1) * width].as_ptr();
                                    let t_next = arr[(y + 1) * width..(y + 2) * width].as_ptr();

                                    for x in 1..width - 1 {
                                        unsafe {
                                            if *p_curr.add(x) > *t_curr.add(x) {
                                                if *p_curr.add(x - 1) > *t_curr.add(x - 1)
                                                    && *p_curr.add(x + 1) > *t_curr.add(x + 1)
                                                    && *p_prev.add(x) > *t_prev.add(x)
                                                    && *p_next.add(x) > *t_next.add(x)
                                                {
                                                    acc.push(row_offset + x);
                                                }
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
                        .reduce(Vec::new, |mut a, mut b| {
                            a.append(&mut b);
                            a
                        })
                }
            }
        };

        // Binary dilation
        self.mask.resize(width * height, false);
        self.mask.fill(false);

        if options.binary_open {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                    *mask_ptr.add(i - 1) = true;
                    *mask_ptr.add(i + 1) = true;
                    *mask_ptr.add(i - width) = true;
                    *mask_ptr.add(i + width) = true;
                }
            }
        } else {
            let mask_ptr = self.mask.as_mut_ptr();
            for &i in &eroded_pixels {
                unsafe {
                    *mask_ptr.add(i) = true;
                }
            }
        }

        let dbg_mask = if options.return_images {
            Some(Array2::from_shape_vec((height, width), self.mask.clone()).unwrap())
        } else {
            None
        };

        let mut extracted = Vec::new();

        // 5. Label regions & 6. Accumulate statistics
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

                let mut check_push = |ni: usize, nx: f64, ny: f64| unsafe {
                    if *self.mask.get_unchecked(ni) {
                        *self.mask.get_unchecked_mut(ni) = false;
                        area += 1;
                        let v = *self.image_vec.get_unchecked(ni) as f64;
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

            if let Some(min_a) = options.min_area
                && area < min_a
            {
                continue;
            }
            if let Some(max_a) = options.max_area
                && area > max_a
            {
                continue;
            }
            if let Some(min_s) = options.min_sum
                && sum < min_s
            {
                continue;
            }
            if let Some(max_s) = options.max_sum
                && sum > max_s
            {
                continue;
            }
            if sum == 0.0 {
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

            if let Some(max_ar) = options.max_axis_ratio
                && (axis_ratio > max_ar || minor <= 0.0)
            {
                continue;
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
        if let Some(max_ret) = options.max_returned {
            extracted.truncate(max_ret);
        }

        // 8. Centroid window
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
                    let inv_img_sum = 1.0 / img_sum;
                    centroid.x = sum_xc * inv_img_sum + o_x as f64;
                    centroid.y = sum_yc * inv_img_sum + o_y as f64;
                }
            }
        }

        // 9. Revert effects of crop and downsample
        for centroid in &mut extracted {
            if let Some(ds) = options.downsample {
                centroid.x *= ds as f64;
                centroid.y *= ds as f64;
            }
            if options.crop.is_some() {
                centroid.x += final_offs_w as f64;
                centroid.y += final_offs_h as f64;
            }
        }

        ExtractionResult {
            centroids: extracted,
            debug_images: if options.return_images {
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
