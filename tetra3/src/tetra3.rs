// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

//! Unified `Tetra3` interface tying together centroid extraction and plate solving.
//!
//! Provides the primary [`Tetra3`] facade struct with lazy loading of the star catalog
//! and shared working extractors.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::extractor::{ExtractOptions, ExtractionResult, Extractor};
use crate::fast_extractor::{FastExtractOptions, FastExtractor};
use crate::solver::{Solution, SolveOptions, Solver};

/// The main Tetra3 instance that centralizes star extraction and plate solving.
/// Holds lazy-initialized instances of the Solver and Extractor to minimize startup
/// overhead and prevent unnecessary memory allocations.
pub struct Tetra3 {
    database_path: PathBuf,
    solver: Option<Solver>,
    extractor: Option<Extractor>,
    fast_extractor: Option<FastExtractor>,
}

impl Tetra3 {
    /// Creates a new Tetra3 instance. The database is not loaded until the first
    /// solve operation is executed.
    pub fn new(database_path: impl AsRef<Path>) -> Self {
        Self {
            database_path: database_path.as_ref().to_path_buf(),
            solver: None,
            extractor: None,
            fast_extractor: None,
        }
    }

    /// Helper to lazy-initialize or retrieve the solver.
    fn get_solver(&mut self) -> Result<&mut Solver, Box<dyn std::error::Error>> {
        if self.solver.is_none() {
            self.solver = Some(Solver::load_database(&self.database_path)?);
        }
        Ok(self.solver.as_mut().unwrap())
    }

    /// Helper to lazy-initialize or retrieve the extractor.
    fn get_extractor(&mut self) -> &mut Extractor {
        if self.extractor.is_none() {
            self.extractor = Some(Extractor::new());
        }
        self.extractor.as_mut().unwrap()
    }

    /// Solves the star pattern from pre-extracted centroids.
    pub fn solve_from_centroids(
        &mut self,
        centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
    ) -> Result<Solution, Box<dyn std::error::Error>> {
        let solver = self.get_solver()?;
        Ok(solver.solve(centroids, size, options))
    }

    /// Extracts star centroids from an image array.
    pub fn get_centroids_from_image<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = f32>,
    {
        let extractor = self.get_extractor();
        extractor.extract(image, options)
    }

    /// Extracts star centroids from a u8 image array.
    pub fn get_centroids_from_image_u8<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = u8>,
    {
        let extractor = self.get_extractor();
        extractor.extract_u8(image, options)
    }

    /// Extracts star centroids using the fast extraction path.
    pub fn get_centroids_from_image_fast<S, T>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        fast_options: FastExtractOptions,
    ) -> crate::fast_extractor::FastExtractionResult
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let (height, width) = image.dim();
        let reinit = match &self.fast_extractor {
            Some(fe) => {
                fe.orig_width() != width
                    || fe.orig_height() != height
                    || fe.options() != &fast_options
            }
            None => true,
        };
        if reinit {
            self.fast_extractor = Some(FastExtractor::new(width, height, fast_options));
        }
        let fe = self.fast_extractor.as_mut().unwrap();
        T::extract_fast(fe, image)
    }

    /// Runs the full pipeline using the fast extraction path.
    pub fn solve_from_image_fast<S, T>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        extract_options: FastExtractOptions,
        solve_options: SolveOptions,
    ) -> Result<(Solution, f64), Box<dyn std::error::Error>>
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let t0 = Instant::now();
        let extract_result = self.get_centroids_from_image_fast(image, extract_options);
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let num_centroids = extract_result.centroids.len();
        let mut centroids_arr = Array2::zeros((num_centroids, 2));
        for (i, c) in extract_result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        let (height, width) = image.dim();
        let solution = self.solve_from_centroids(
            &centroids_arr,
            (height as f64, width as f64),
            solve_options,
        )?;

        Ok((solution, extract_time_ms))
    }

    /// Runs the full pipeline: extracts centroids from the image and immediately solves them.
    /// Returns the Solution alongside the extraction time in milliseconds.
    pub fn solve_from_image<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        extract_options: ExtractOptions,
        solve_options: SolveOptions,
    ) -> Result<(Solution, f64), Box<dyn std::error::Error>>
    where
        S: Data<Elem = f32>,
    {
        let t0 = Instant::now();

        // 1. Extract centroids
        let extract_result = self.get_centroids_from_image(image, extract_options);
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Map Vec<CentroidResult> into the Array2<f64> expected by the solver
        let num_centroids = extract_result.centroids.len();
        let mut centroids_arr = Array2::zeros((num_centroids, 2));
        for (i, c) in extract_result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        // 2. Solve
        let (height, width) = image.dim();
        let solution = self.solve_from_centroids(
            &centroids_arr,
            (height as f64, width as f64),
            solve_options,
        )?;

        Ok((solution, extract_time_ms))
    }
}

/// Internal trait to unify dispatch between f32 and u8 for fast extraction.
pub trait FastPixel: Copy {
    /// Executes the fast extraction path.
    fn extract_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> crate::fast_extractor::FastExtractionResult
    where
        S: Data<Elem = Self>;

    /// Executes the fast extraction path with variants.
    fn extract_variants_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
        variants: &[crate::fast_extractor::FastExtractOptionsUpdate],
    ) -> Vec<crate::fast_extractor::FastExtractionResult>
    where
        S: Data<Elem = Self>;

    /// Executes the standard extraction path.
    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>;
}

impl FastPixel for f32 {
    fn extract_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> crate::fast_extractor::FastExtractionResult
    where
        S: Data<Elem = Self>,
    {
        fe.extract_f32(image)
    }

    fn extract_variants_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
        variants: &[crate::fast_extractor::FastExtractOptionsUpdate],
    ) -> Vec<crate::fast_extractor::FastExtractionResult>
    where
        S: Data<Elem = Self>,
    {
        fe.extract_variants_f32(image, variants)
    }

    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>,
    {
        t3.get_centroids_from_image(image, options)
    }
}

impl FastPixel for u8 {
    fn extract_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> crate::fast_extractor::FastExtractionResult
    where
        S: Data<Elem = Self>,
    {
        fe.extract(image)
    }

    fn extract_variants_fast<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
        variants: &[crate::fast_extractor::FastExtractOptionsUpdate],
    ) -> Vec<crate::fast_extractor::FastExtractionResult>
    where
        S: Data<Elem = Self>,
    {
        fe.extract_variants(image, variants)
    }

    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>,
    {
        t3.get_centroids_from_image_u8(image, options)
    }
}
