use ndarray_npy::NpzWriter;

use ndarray::Array1;
use std::{fs::File, error::Error, collections::HashMap};

use log::info;

use crate::{knns::{NearestNeighbors, InterpolatedCDF}, cdf::KNN, interp::InterpConfig};

pub fn output_to_disk<const K: usize, const DD: bool>(
    mut knns: NearestNeighbors,
    interp_config: &InterpConfig,
) -> Result<(), Box<dyn Error>> {

    // Initialize NpzWriter
    let mut out_file = knns.dataset.out().to_owned();
    if DD { out_file.push_str("_DD"); }
    let mut npz = NpzWriter::new(File::create(out_file)?);

    // Output data (i.e. cdf y values)
    for run in 1..=15 {

        // Get interpolated cdfs
        let interpolated_cdfs: &mut HashMap<KNN, InterpolatedCDF> = knns
            .interpolators
            .get_mut(&run)
            .expect("run data should exist");

        for k in 1..=K as u16 {

            // Interpolated cdf
            let interpolated_cdf: InterpolatedCDF = interpolated_cdfs
                .remove(&k)
                .expect("knn data should exist");


            if DD {
                npz.add_array(format!("run{run}_{k}_DD").as_str(), &Array1::from_vec(interpolated_cdf))?;
            } else {
                npz.add_array(format!("run{run}_{k}").as_str(), &Array1::from_vec(interpolated_cdf))?;
            }
            info!("Appended run = {run}, k = {k} to output file");
        }        
    }

    // Now output percentiles used
    npz.add_array("quantiles", &Array1::from_vec(interp_config.quantiles.clone()))?;

    npz.finish()?;
    Ok(())
}