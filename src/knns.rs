use dashmap::DashMap;
use nabo::{KDTree, Neighbour, Scalar, Point, dummy_point::P3, NotNan};
use rand::thread_rng;
use rayon::prelude::*;

use crate::load_data::Dataset;
use crate::load_data::{Galaxies, Run};
use log::info;
use rand::Rng;

pub type Neighbours<T, P> = Vec<Neighbour<T, P>>;

pub fn calculate_knns<
    const SUBSAMPLE: usize,
    const QUERIES: usize,
    const LENGTH: usize,
    const LEAFSIZE: u32,
    const K: usize,
    >(
    galaxies: &Galaxies
) -> NearestNeighbors<f64, P3> {

    // Initialize inner return value
    let neighbors = DashMap::new();

    for run in 1..=15 {

        info!("Working on run {run}");

        // Get galaxy positions for this run
        let entry = galaxies
            .position
            .get(&run)
            .expect("entry for run should exist");
        let pos: &Vec<P3> = entry.value();

        // Calculate num galaxies, samples, queries per sample
        let num_galaxies = pos.len();
        let num_subsamples = num_galaxies / (SUBSAMPLE * (LENGTH / 1000).pow(3));
        let query_per_sample = QUERIES / num_subsamples;

        info!("Run {run} has {num_galaxies} --> {num_subsamples} subsamples");
        info!("Using {query_per_sample} queries per subsample");

        // Construct an array representing the boxsize
        let periodic: [NotNan<f64>; 3] = [NotNan::new(LENGTH as f64).unwrap(); 3];

        // Chunk positions and find knns
        let all_knns: Vec<Neighbours<f64, P3>> = pos
            .par_chunks_exact(SUBSAMPLE * (LENGTH / 1000).pow(3))
            .flat_map(|subsample| {

                // Generate queries for this subsample
                let queries: Vec<P3> = gen_queries::<LENGTH>(query_per_sample);
                let kdtree = KDTree::new_with_bucket_size(subsample, LEAFSIZE);
                
                queries
                    .par_iter()
                    .map_with(&kdtree, |tree, query| {
                        tree.knn_periodic(K as u32, query, &periodic)
                    })
                    .collect::<Vec<Neighbours<f64, P3>>>()
            })
            .collect::<Vec<Neighbours<f64, P3>>>();

        // Add neighbors to map
        neighbors.insert(run, all_knns);
    }

    NearestNeighbors {
        neighbors,
        dataset: galaxies.dataset,
    }
}



pub struct NearestNeighbors<T: Scalar, P: Point<T>> {
    pub neighbors: DashMap<Run, Vec<Neighbours<T, P>>>,
    pub(crate) dataset: Dataset,
}




fn gen_queries<const LENGTH: usize>(n: usize) -> Vec<P3> {

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