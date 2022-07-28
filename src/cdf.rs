use interp1d::*;
use rayon::prelude::*;
use itertools::Itertools;


pub type KNN = u16;
pub type Cdf = f64;
pub type Distance = f64;

pub(crate) fn construct_cdf_interpolator<const K: usize>(dists: Vec<[f64; K]>) -> std::collections::HashMap<KNN, Interp1d<Cdf, Distance>> {


    let size = dists.len();
    let cdf = (1..=size)
            .map(|i| i as f64 / size as f64)
            .collect_vec();

        
    (0..K)
        .map(|k| {
            (k, dists
                .iter()
                .map(|x| x[k])
                .sorted_by(|a, b| a.partial_cmp(&b).expect("all distances should be finite"))
                .collect_vec())
        })
        .par_bridge()
        .into_par_iter()
        .map(|(k, dists)| (k as u16 +1, Interp1d::new_sorted(cdf.clone(), dists).unwrap()))
        .collect()
}