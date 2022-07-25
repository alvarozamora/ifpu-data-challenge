use dashmap::DashMap;
use interp1d::*;
use nabo_pbc::dummy_point::P3;
use rayon::prelude::*;
use itertools::Itertools;

use crate::knns::Neighbours;

pub type KNN = u16;
pub type Cdf = f64;
pub type Distance = f64;

pub(crate) fn construct_cdf_interpolator<const K: usize>(neighbors: Vec<Neighbours<f64, P3>>) -> DashMap<KNN, Interp1d<Cdf, Distance>> {


    // Get distance and sort them
    let mut dists: [Vec<f64>; K] = {
        
        let dists: Vec<[f64; K]> = neighbors
            .into_par_iter()
            .map(|neighbors| {

                // Disard indices, neighbor positoins for this query point
                let knn_dists: [f64; K] = neighbors
                    .iter()
                    .map(|neighbor| neighbor.dist2.sqrt())
                    .collect_vec()
                    .try_into()
                    .unwrap();

                knn_dists
            }).collect();

        (0..K).map(|k|{
            let mut knn_dists = dists
                .iter()
                .cloned()
                .map(|x| x[k])
                .collect_vec();
            knn_dists.sort_by(|a, b| a.partial_cmp(&b).expect("all distances should be finite"));
            knn_dists
        })
        .collect_vec()
        .try_into()
        .unwrap()
    };
    // Calculate cdf
    (0..K).map(|k| {

        let size = dists[k].len();
        let cdf = (1..=size)
            .map(|i| i as f64 / size as f64)
            .collect_vec();
        (k as u16 + 1, Interp1d::new_sorted(cdf, std::mem::take(&mut dists[k])).unwrap())
    })
    .collect()

}