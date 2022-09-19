use std::collections::HashMap;

use itertools::Itertools;
use nabo_pbc::{KDTree, Neighbour, dummy_point::P3, NotNan};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::cdf::{construct_cdf_interpolator, KNN};
use crate::load_data::Dataset;
use crate::load_data::{Galaxies, Run};
use crate::interp::InterpConfig;
use log::{info, trace};
use rand::Rng;

pub type Neighbours<T, P> = Vec<Neighbour<T, P>>;
pub type InterpolatedCDF = Vec<f64>;

pub fn calculate_knns<
    const SUBSAMPLE: usize,
    const QUERIES: usize,
    const LENGTH: usize,
    const LEAFSIZE: u32,
    const K: usize,
    >(
    galaxies: &Galaxies,
    interp_config: &InterpConfig,
) -> NearestNeighbors {

    // Initialize inner return value
    let mut interpolators = HashMap::new();

    for run in 1..=15 {

        info!("Working on run {run}");

        // Get galaxy positions for this run
        let entry = galaxies
            .position
            .get(&run)
            .expect("entry for run should exist");
        let mut pos: Vec<P3> = entry.value().clone();

        // Calculate num galaxies, samples, queries per sample
        let num_galaxies = pos.len();
        let num_subsamples = num_galaxies / (SUBSAMPLE * (LENGTH / 1000).pow(3));
        let query_per_sample = QUERIES / num_subsamples;

        info!("Run {run} has {num_galaxies} --> {num_subsamples} subsamples");
        info!("Using {query_per_sample} queries per subsample");

        // Construct an array representing the boxsize
        let periodic: [NotNan<f64>; 3] = [NotNan::new(LENGTH as f64).unwrap(); 3];

        // Chunk positions and find knns
        (&mut pos).shuffle(&mut thread_rng());
        let nearest_neighbors: Vec<[f64; K]> = pos
            .par_chunks_exact(SUBSAMPLE * (LENGTH / 1000).pow(3))
            .flat_map(|subsample| {

                // Generate queries for this subsample
                let queries: Vec<P3> = gen_queries::<LENGTH>(query_per_sample);
                let kdtree = KDTree::new_with_bucket_size(subsample, LEAFSIZE);
                
                let neighbors: Vec<[f64; K]> = queries
                    .par_iter()
                    .map_with(&kdtree, |tree, query| {
                        
                        tree
                            .knn_periodic(K as u32, query, &periodic)
                            .into_iter()
                            .map(|neighbor| {
                                neighbor.dist2.sqrt()
                            })
                            .collect_vec()
                            .try_into()
                            .unwrap()

                    }).collect();

                    neighbors
            }).collect();

        // Construct interpolators
        let run_interpolators = construct_cdf_interpolator(nearest_neighbors);

        // Interpolate
        let interpolated_cdfs: HashMap<KNN, InterpolatedCDF> = run_interpolators
            .into_iter()
            .map(|(k, interp)| {
                (k, interp_config
                    .quantiles
                    .par_iter()
                    .map(|&quantile|{
                        interp.interpolate(quantile)
                    }).collect())
                }).collect();
            
        // Add neighbors to map
        interpolators.insert(run, interpolated_cdfs);
    }

    NearestNeighbors {
        interpolators,
        dataset: galaxies.dataset,
    }
}

pub fn calculate_knns_dd<
    const SUBSAMPLE: usize,
    const LENGTH: usize,
    const LEAFSIZE: u32,
    const K: usize,
    >(
    galaxies: &Galaxies,
    interp_config: &InterpConfig,
) -> NearestNeighbors {

    // Initialize inner return value
    let mut interpolators = HashMap::new();

    for run in 1..=15 {

        info!("Working on run {run}");

        // Get galaxy positions for this run
        let entry = galaxies
            .position
            .get(&run)
            .expect("entry for run should exist");
        let mut pos: Vec<P3> = entry.value().clone();

        // Calculate num galaxies, samples, queries per sample
        let num_galaxies = pos.len();
        let num_subsamples = num_galaxies / (SUBSAMPLE * (LENGTH / 1000).pow(3));

        info!("Run {run} has {num_galaxies} --> {num_subsamples} subsamples");
        info!("Using {} dd queries per N={SUBSAMPLE} subsample", num_galaxies - SUBSAMPLE);

        // Construct an array representing the boxsize
        let periodic: [NotNan<f64>; 3] = [NotNan::new(LENGTH as f64).unwrap(); 3];

        // Chunk positions and find knns
        (&mut pos).shuffle(&mut thread_rng());
        let nearest_neighbors: Vec<[f64; K]> = (0..num_subsamples)
            .flat_map(|subsample_id| {

                // Start and end index of this subsample
                let start = subsample_id * num_galaxies / num_subsamples;
                let end = (subsample_id + 1) * num_galaxies / num_subsamples;
                
                // Get subsample and remainder and remainders
                let subsample: &[P3] = &pos[start..end];
                let left_rem: &[P3] = &pos[..start];
                let right_rem: &[P3] = &pos[end..];
                let remainders = left_rem
                    .into_par_iter()
                    .chain(right_rem.into_par_iter());
                    
                trace!("DD: using {} queries", remainders.len());
                
                // Build tree on subsample
                let kdtree = KDTree::new_with_bucket_size(subsample, LEAFSIZE);
                
                // Allocate vec
                let mut neighbors = Vec::with_capacity(left_rem.len() + right_rem.len());

                // Append remainders
                neighbors.append(&mut
                    remainders
                        .map_with(&kdtree, |tree, query| {
                            tree
                                .knn_periodic(K as u32, query, &periodic)
                                .into_iter()
                                .map(|neighbor| {
                                    neighbor.dist2.sqrt()
                                })
                                .collect_vec()
                                .try_into()
                                .unwrap()
                        }).collect());
                neighbors
            }).collect();

        // Construct interpolators
        let run_interpolators = construct_cdf_interpolator(nearest_neighbors);

        // Interpolate
        let interpolated_cdfs: HashMap<KNN, InterpolatedCDF> = run_interpolators
            .into_iter()
            .map(|(k, interp)| {
                (k, interp_config
                    .quantiles
                    .par_iter()
                    .map(|&quantile|{
                        interp.interpolate(quantile)
                    }).collect())
                }).collect();

        // Add neighbors to map
        interpolators.insert(run, interpolated_cdfs);
    }

    NearestNeighbors {
        interpolators,
        dataset: galaxies.dataset,
    }
}

pub struct NearestNeighbors {
    pub interpolators: HashMap<Run, HashMap<KNN, InterpolatedCDF>>,
    pub(crate) dataset: Dataset,
}

fn gen_queries<const LENGTH: usize>(n: usize) -> Vec<P3> {

    // Get reference to thread local rng
    let ref mut rng = thread_rng();

    (0..n)
        .map(|_|{

            // Generate three uniform randoms, and multiply by the box length
            let three_rands: [f64; 3] = rng.gen::<[f64; 3]>().map(|x| x * LENGTH as f64);

            // Construct a P3
            P3::new(three_rands[0], three_rands[1], three_rands[2])
        })
        .collect()
}
