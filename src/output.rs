use interp1d::Interp1d;
use itertools::Itertools;
use ndarray_npy::NpzWriter;

use ndarray::Array1;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;
use std::{fs::File, error::Error, collections::HashMap};

use log::info;

use crate::{knns::NearestNeighbors, cdf::KNN};

pub fn output_to_disk<const K: usize>(
    knns: NearestNeighbors,
    interp_config: InterpConfig,
) -> Result<(), Box<dyn Error>> {

    // Initialize NpzWriter
    let mut npz = NpzWriter::new(File::create(knns.dataset.out())?);

    // Output data (i.e. cdf y values)
    for run in 1..=15 {

        // Construct CDF interpolator
        let cdf_interpolator: &HashMap<KNN, Interp1d<f64, f64>> = &*knns.interpolators.get(&run).expect("run data should exist");

        for k in 1..=K {

            info!("Interpolating (run, k) = ({run}, {k})");

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

    // Now output percentiles used
    npz.add_array("quantiles", &Array1::from_vec(interp_config.quantiles))?;


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
        let left_points = points / 2 + 1;
        let dlogy = (ymax_log10 - ymin_log10) / left_points as f64;
        let mut left_half = (0..=left_points)
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