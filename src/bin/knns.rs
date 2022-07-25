
use std::error::Error;
use log::info;
use env_logger::Builder;
use log::LevelFilter;


use ifpu_knns::load_data::*;
use ifpu_knns::knns::calculate_knns;
use ifpu_knns::output::output_to_disk;
use nabo_pbc::dummy_point::P3;

use ifpu_knns::{
    knns::NearestNeighbors,
    output::InterpConfig
};

fn main() -> Result<(), Box<dyn Error>> {

    // Logging
    let mut builder = Builder::new();
    builder
        .filter_level(LevelFilter::Info)
        .write_style(env_logger::WriteStyle::Auto)
        .format_timestamp_secs()
        .init();

    /// Number of points per subsample
    const SUBSAMPLE: usize = 500;
    /// Total number of queries
    const QUERIES: usize = 100_000_000;
    /// Box length, in Mpc/h
    const LENGTH: usize = 2_000;
    /// Number of neighbors to find
    const K: usize = 4;
    /// Leafsize to use for KDTree
    const LEAFSIZE: u32 = 32;
    /// Minimum value for pcdf
    const LOG_YMIN: f64 = -3.0;
    /// Number of points to use for cdf interpolation. 
    /// Must be odd so that peak @ median is captured
    const NPOINTS: usize = 201;
    
    // Load Galaxies
    let galaxies: Galaxies = load(Dataset::Challenge0Z0)?;

    // Calculate kNNs
    info!("Calculating kNNs");
    let knns: NearestNeighbors<f64, P3> = calculate_knns::
        <SUBSAMPLE, QUERIES, LENGTH, LEAFSIZE, K>(&galaxies);

    // Interpolate knns and output to disk
    let interp_config = InterpConfig::new(LOG_YMIN, NPOINTS)
        .expect("an odd number must be provided");
    output_to_disk::<K>(knns, interp_config)?;

    Ok(())
}