
use std::error::Error;
use log::info;
use env_logger::Builder;
use log::LevelFilter;


use ifpu_knns::load_data::*;
use ifpu_knns::knns::*;
use ifpu_knns::output::output_to_disk;

use ifpu_knns::{
    interp::InterpConfig,
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
    const K: usize = 8;
    /// Leafsize to use for KDTree
    const LEAFSIZE: u32 = 16;
    /// Minimum value for pcdf
    const LOG_YMIN: f64 = -5.0;
    /// Number of points to use for cdf interpolation. 
    /// Must be odd so that peak @ median is captured
    const NPOINTS: usize = 401;
    
    // Load Galaxies
    let galaxies: Galaxies = load(Dataset::Challenge0Z0)?;

    // Initialize common interp parameters
    let interp_config = InterpConfig::new(LOG_YMIN, NPOINTS)
        .expect("an odd number of points must be provided");

    // Calculate and interp dd knns
    let dd_knns = calculate_knns_dd::<SUBSAMPLE, LENGTH, LEAFSIZE, K>(
        &galaxies,
        &interp_config,
    );
    output_to_disk::<K, true>(dd_knns, &interp_config)?;

    // Calculate and interp rd kNNs
    info!("Calculating kNNs");
    let knns = calculate_knns::<SUBSAMPLE, QUERIES, LENGTH, LEAFSIZE, K>(
            &galaxies,
            &interp_config
    );
    output_to_disk::<K, false>(knns, &interp_config)?;


    Ok(())
}
