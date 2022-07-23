use dashmap::DashMap;
use interp1d::Interp1d;
use itertools::Itertools;
use nabo::dummy_point::P3;
use ndarray_npy::NpzWriter;

use ndarray::Array1;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;
use std::{fs::File, error::Error};

use crate::{knns::NearestNeighbors, cdf::{construct_cdf_interpolator, KNN}};

pub fn output_to_disk<const K: usize>(
    knns: NearestNeighbors<f64, P3>,
    interp_config: InterpConfig,
) -> Result<(), Box<dyn Error>> {

    // Initialize NpzWriter
    let mut npz = NpzWriter::new(File::create(knns.dataset.out())?);

    for run in 1..=15 {

        // Construct CDF interpolator
        let cdf_interpolator: DashMap<KNN, Interp1d<f64, f64>> = construct_cdf_interpolator::<K>(
            std::mem::take(&mut*knns.neighbors.get_mut(&run).unwrap())
        );

        for k in 1..=K {

            // Get interpolator from map
            let interp = cdf_interpolator.get(&(k as u16)).unwrap();

            // Interpolate
            let interpolated_cdf: Vec<f64> = interp_config
                .quantiles
                .par_iter()
                .map(|&quantile|{
                    interp.interpolate(quantile)
                })
                .collect();

            // Convert to array and write to disk
            let interpolated_cdf = Array1::from_vec(interpolated_cdf);
            npz.add_array(format!("run{run}_{k}").as_str(), &interpolated_cdf)?;
        }

        
    }
    npz.finish()?;

    Ok(())
}


#[derive(Debug)]
pub struct InterpConfig {
    quantiles: Vec<f64>,
}

impl InterpConfig {
    pub fn new(ymin_log10: f64, points: usize) -> Result<InterpConfig, &'static str> {

        if points % 2 == 0 { return Err("InterpConfig requires an odd number of points") }

        let ymax_log10 = (0.5_f64).log10();
        
        // Construct left half
        let left_points = points / 2;
        let dlogy = (ymax_log10 - ymin_log10) / left_points as f64;
        let mut left_half = (0..left_points)
            .map(|i| 10.0_f64.powf(ymin_log10 + i as f64 * dlogy))
            .collect_vec();
        
        // Construct right half (equiv to left half w/ one less point and reflected @ 0.5)
        let mut right_half = (0..(left_points-1))
            .map(|i| 1.0 - 10.0_f64.powf(ymin_log10 + i as f64 * dlogy))
            .collect_vec();
        right_half.reverse();

        left_half.append(&mut right_half);
        Ok(InterpConfig {
            quantiles: left_half
        })
    }
}