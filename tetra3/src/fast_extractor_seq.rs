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
impl FastExtractor {
    /// Sequential version of the extractor that trades some accuract for single-threaded performance.
    pub fn extract_sequential<S>(
        &mut self,
        input_image: &ArrayBase<S, Ix2>,
    ) -> Vec<FastCentroidResult>
    where
        S: Data<Elem = u8>,
    {
        debug_assert_eq!(
            input_image.dim(),
            (self.height, self.width),
            "Input image dimensions must match the initialized FastExtractor dimensions."
        );

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
                        for row in self.downsampled_u32.chunks_exact(self.out_width).step_by(4) {
                            for &v in row {
                                unsafe {
                                    *hist.get_unchecked_mut(v as usize) += 1;
                                }
                            }
                            count += self.out_width;
                        }
                        let target = ((count + 1) / 2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &c) in hist.iter().enumerate() {
                            accum += c;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        // OPTIMIZATION: Manually unrolled second pass accumulates residual sum-of-squares
                        // while performing integer subtraction. Contiguous access allows SIMD.
                        let mut sum_sq0 = 0.0;
                        let mut sum_sq1 = 0.0;
                        let mut sum_sq2 = 0.0;
                        let mut sum_sq3 = 0.0;
                        let mut o_chunks = self.image_i32.chunks_exact_mut(4);
                        let mut i_chunks = self.downsampled_u32.chunks_exact(4);
                        for (o, i) in o_chunks.by_ref().zip(i_chunks.by_ref()) {
                            let v0 = (i[0] as f32) - med;
                            let v1 = (i[1] as f32) - med;
                            let v2 = (i[2] as f32) - med;
                            let v3 = (i[3] as f32) - med;
                            o[0] = (v0 * 128.0).round() as i32;
                            o[1] = (v1 * 128.0).round() as i32;
                            o[2] = (v2 * 128.0).round() as i32;
                            o[3] = (v3 * 128.0).round() as i32;
                            sum_sq0 += (v0 * v0) as f64;
                            sum_sq1 += (v1 * v1) as f64;
                            sum_sq2 += (v2 * v2) as f64;
                            sum_sq3 += (v3 * v3) as f64;
                        }
                        let mut sum_sq = sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;
                        for (o, &i) in o_chunks
                            .into_remainder()
                            .iter_mut()
                            .zip(i_chunks.remainder().iter())
                        {
                            let val_f32 = (i as f32) - med;
                            *o = (val_f32 * 128.0).round() as i32;
                            sum_sq += (val_f32 * val_f32) as f64;
                        }
                        sum_sq
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
                                count += rem_read as usize;
                            }

                            let target = ((count + 1) / 2) as u32;
                            let mut accum = 0;
                            let mut med = 0.0f32;
                            for (val, &c) in hist.iter().enumerate() {
                                accum += c;
                                if accum >= target {
                                    med = val as f32;
                                    break;
                                }
                            }

                            let mut sum_sq0 = 0.0;
                            let mut sum_sq1 = 0.0;
                            let mut sum_sq2 = 0.0;
                            let mut sum_sq3 = 0.0;
                            let mut o_chunks = o_row.chunks_exact_mut(4);
                            let mut i_chunks = i_row.chunks_exact(4);

                            for (o, i) in o_chunks.by_ref().zip(i_chunks.by_ref()) {
                                let v0 = (i[0] as f32) - med;
                                let v1 = (i[1] as f32) - med;
                                let v2 = (i[2] as f32) - med;
                                let v3 = (i[3] as f32) - med;
                                o[0] = (v0 * 128.0).round() as i32;
                                o[1] = (v1 * 128.0).round() as i32;
                                o[2] = (v2 * 128.0).round() as i32;
                                o[3] = (v3 * 128.0).round() as i32;
                                sum_sq0 += (v0 * v0) as f64;
                                sum_sq1 += (v1 * v1) as f64;
                                sum_sq2 += (v2 * v2) as f64;
                                sum_sq3 += (v3 * v3) as f64;
                            }

                            let mut row_sum_sq = sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;
                            for (o, &i) in o_chunks
                                .into_remainder()
                                .iter_mut()
                                .zip(i_chunks.remainder().iter())
                            {
                                let val_f32 = (i as f32) - med;
                                *o = (val_f32 * 128.0).round() as i32;
                                row_sum_sq += (val_f32 * val_f32) as f64;
                            }
                            total_sum_sq += row_sum_sq;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = (self.out_width + block_size - 1) / block_size;
                        let grid_h = (self.out_height + block_size - 1) / block_size;
                        let mut hists = vec![0u32; grid_w * 4096];

                        for gy in 0..grid_h {
                            hists.fill(0);
                            let start_y = gy * block_size;
                            let end_y = (start_y + block_size).min(self.out_height);
                            // OPTIMIZATION: .step_by(4) row skipping reads only 25% of block pixels
                            for y in (start_y..end_y).step_by(4) {
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
                                let num_rows = (start_y..end_y).step_by(4).count();
                                let count = num_rows * (end_x - start_x);
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
                                let mut sum_sq0 = 0.0;
                                let mut sum_sq1 = 0.0;
                                let mut sum_sq2 = 0.0;
                                let mut sum_sq3 = 0.0;

                                let mut out_chunks = out_row.chunks_exact_mut(4);
                                let mut src_chunks = src_row.chunks_exact(4);
                                let mut gx0_chunks = bg_gx0.chunks_exact(4);
                                let mut tx_chunks = bg_tx.chunks_exact(4);

                                // OPTIMIZATION: Piecewise Constant Interpolation.
                                // Instead of interpolating the exact background for all 4 pixels, we evaluate it once
                                // and apply it to the whole chunk. This eliminates 75% of LUT accesses and math.
                                // Manually unrolled loop with multiple independent accumulators
                                // improves instruction-level parallelism.
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

                                        let v0 = (*s.get_unchecked(0) as f32) - bg;
                                        let v1 = (*s.get_unchecked(1) as f32) - bg;
                                        let v2 = (*s.get_unchecked(2) as f32) - bg;
                                        let v3 = (*s.get_unchecked(3) as f32) - bg;

                                        *o.get_unchecked_mut(0) = (v0 * 128.0).round() as i32;
                                        *o.get_unchecked_mut(1) = (v1 * 128.0).round() as i32;
                                        *o.get_unchecked_mut(2) = (v2 * 128.0).round() as i32;
                                        *o.get_unchecked_mut(3) = (v3 * 128.0).round() as i32;

                                        sum_sq0 += (v0 * v0) as f64;
                                        sum_sq1 += (v1 * v1) as f64;
                                        sum_sq2 += (v2 * v2) as f64;
                                        sum_sq3 += (v3 * v3) as f64;
                                    }
                                }

                                row_sq_sum += sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;

                                let x = out_chunks.into_remainder().len();
                                if x > 0 {
                                    let rem_x = self.out_width - x;
                                    for i in 0..x {
                                        unsafe {
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i);
                                            let tx = *bg_tx.get_unchecked(rem_x + i);
                                            let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];
                                            let val_f32 =
                                                (*src_row.get_unchecked(rem_x + i) as f32) - bg_val;
                                            *out_row.get_unchecked_mut(rem_x + i) =
                                                (val_f32 * 128.0).round() as i32;
                                            row_sq_sum += (val_f32 * val_f32) as f64;
                                        }
                                    }
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

            Self::execute_erosion_and_extraction_sequential(
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
                        for row in src_slice.chunks_exact(self.width).step_by(4) {
                            for &v in row {
                                unsafe {
                                    *hist.get_unchecked_mut(v as usize) += 1;
                                }
                            }
                            count += self.width;
                        }
                        let target = ((count + 1) / 2) as u32;
                        let mut accum = 0;
                        let mut med = 0.0f32;
                        for (val, &c) in hist.iter().enumerate() {
                            accum += c;
                            if accum >= target {
                                med = val as f32;
                                break;
                            }
                        }

                        // OPTIMIZATION: Manually unrolled second pass accumulates residual sum-of-squares
                        // while performing integer subtraction. Contiguous access allows SIMD.
                        let mut sum_sq0 = 0.0;
                        let mut sum_sq1 = 0.0;
                        let mut sum_sq2 = 0.0;
                        let mut sum_sq3 = 0.0;
                        let mut o_chunks = self.image_i16.chunks_exact_mut(4);
                        let mut i_chunks = src_slice.chunks_exact(4);
                        for (o, i) in o_chunks.by_ref().zip(i_chunks.by_ref()) {
                            let v0 = (i[0] as f32) - med;
                            let v1 = (i[1] as f32) - med;
                            let v2 = (i[2] as f32) - med;
                            let v3 = (i[3] as f32) - med;
                            o[0] = (v0 * 128.0).round() as i16;
                            o[1] = (v1 * 128.0).round() as i16;
                            o[2] = (v2 * 128.0).round() as i16;
                            o[3] = (v3 * 128.0).round() as i16;
                            sum_sq0 += (v0 * v0) as f64;
                            sum_sq1 += (v1 * v1) as f64;
                            sum_sq2 += (v2 * v2) as f64;
                            sum_sq3 += (v3 * v3) as f64;
                        }
                        let mut sum_sq = sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;
                        for (o, &i) in o_chunks
                            .into_remainder()
                            .iter_mut()
                            .zip(i_chunks.remainder().iter())
                        {
                            let val_f32 = (i as f32) - med;
                            *o = (val_f32 * 128.0).round() as i16;
                            sum_sq += (val_f32 * val_f32) as f64;
                        }
                        sum_sq
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
                                count += rem_read as usize;
                            }

                            let target = ((count + 1) / 2) as u32;
                            let mut accum = 0;
                            let mut med = 0.0f32;
                            for (val, &c) in hist.iter().enumerate() {
                                accum += c;
                                if accum >= target {
                                    med = val as f32;
                                    break;
                                }
                            }

                            let mut sum_sq0 = 0.0;
                            let mut sum_sq1 = 0.0;
                            let mut sum_sq2 = 0.0;
                            let mut sum_sq3 = 0.0;
                            let mut o_chunks = o_row.chunks_exact_mut(4);
                            let mut i_chunks = i_row.chunks_exact(4);

                            for (o, i) in o_chunks.by_ref().zip(i_chunks.by_ref()) {
                                let v0 = (i[0] as f32) - med;
                                let v1 = (i[1] as f32) - med;
                                let v2 = (i[2] as f32) - med;
                                let v3 = (i[3] as f32) - med;
                                o[0] = (v0 * 128.0).round() as i16;
                                o[1] = (v1 * 128.0).round() as i16;
                                o[2] = (v2 * 128.0).round() as i16;
                                o[3] = (v3 * 128.0).round() as i16;
                                sum_sq0 += (v0 * v0) as f64;
                                sum_sq1 += (v1 * v1) as f64;
                                sum_sq2 += (v2 * v2) as f64;
                                sum_sq3 += (v3 * v3) as f64;
                            }

                            let mut row_sum_sq = sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;
                            for (o, &i) in o_chunks
                                .into_remainder()
                                .iter_mut()
                                .zip(i_chunks.remainder().iter())
                            {
                                let val_f32 = (i as f32) - med;
                                *o = (val_f32 * 128.0).round() as i16;
                                row_sum_sq += (val_f32 * val_f32) as f64;
                            }
                            total_sum_sq += row_sum_sq;
                        }
                        total_sum_sq
                    }
                    FastBgSubMode::BlockMedian { block_size } => {
                        let grid_w = (self.width + block_size - 1) / block_size;
                        let grid_h = (self.height + block_size - 1) / block_size;
                        let mut hists = vec![0u32; grid_w * 256];
                        for gy in 0..grid_h {
                            hists.fill(0);
                            let start_y = gy * block_size;
                            let end_y = (start_y + block_size).min(self.height);
                            // OPTIMIZATION: .step_by(4) row skipping reads only 25% of block pixels
                            for y in (start_y..end_y).step_by(4) {
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
                                let num_rows = (start_y..end_y).step_by(4).count();
                                let count = num_rows * (end_x - start_x);
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
                                let mut sum_sq0 = 0.0;
                                let mut sum_sq1 = 0.0;
                                let mut sum_sq2 = 0.0;
                                let mut sum_sq3 = 0.0;

                                let mut out_chunks = out_row.chunks_exact_mut(4);
                                let mut src_chunks = src_row.chunks_exact(4);
                                let mut gx0_chunks = bg_gx0.chunks_exact(4);
                                let mut tx_chunks = bg_tx.chunks_exact(4);

                                // OPTIMIZATION: Piecewise Constant Interpolation.
                                // Instead of interpolating the exact background for all 4 pixels, we evaluate it once
                                // and apply it to the whole chunk. This eliminates 75% of LUT accesses and math.
                                // Manually unrolled loop with multiple independent accumulators
                                // improves instruction-level parallelism.
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

                                        let v0 = (*s.get_unchecked(0) as f32) - bg;
                                        let v1 = (*s.get_unchecked(1) as f32) - bg;
                                        let v2 = (*s.get_unchecked(2) as f32) - bg;
                                        let v3 = (*s.get_unchecked(3) as f32) - bg;

                                        *o.get_unchecked_mut(0) = (v0 * 128.0).round() as i16;
                                        *o.get_unchecked_mut(1) = (v1 * 128.0).round() as i16;
                                        *o.get_unchecked_mut(2) = (v2 * 128.0).round() as i16;
                                        *o.get_unchecked_mut(3) = (v3 * 128.0).round() as i16;

                                        sum_sq0 += (v0 * v0) as f64;
                                        sum_sq1 += (v1 * v1) as f64;
                                        sum_sq2 += (v2 * v2) as f64;
                                        sum_sq3 += (v3 * v3) as f64;
                                    }
                                }

                                row_sq_sum += sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3;

                                let x = out_chunks.into_remainder().len();
                                if x > 0 {
                                    let rem_x = self.width - x;
                                    for i in 0..x {
                                        unsafe {
                                            let gx0 = *bg_gx0.get_unchecked(rem_x + i);
                                            let tx = *bg_tx.get_unchecked(rem_x + i);
                                            let bg_val = v_grid_row[gx0] + tx * d_grid_row[gx0];
                                            let val_f32 =
                                                (*src_row.get_unchecked(rem_x + i) as f32) - bg_val;
                                            *out_row.get_unchecked_mut(rem_x + i) =
                                                (val_f32 * 128.0).round() as i16;
                                            row_sq_sum += (val_f32 * val_f32) as f64;
                                        }
                                    }
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

            Self::execute_erosion_and_extraction_sequential(
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

    fn execute_erosion_and_extraction_sequential<T>(
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

            if area < min_a || area > max_a || sum < min_s || sum > max_s || sum == 0.0 {
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
                sum,
                area,
                axis_ratio,
            });
        }

        // 3. Sort
        extracted.sort_by(|a, b| b.sum.partial_cmp(&a.sum).unwrap_or(Ordering::Equal));

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
        if ds > 1 {
            for centroid in &mut extracted {
                centroid.x *= ds as f64;
                centroid.y *= ds as f64;
            }
        }
        extracted
    }
}
