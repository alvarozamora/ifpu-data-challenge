

use std::error::Error;
use nabo::dummy_point::P3;
use ndarray_npy::NpzReader;
use ndarray::{OwnedRepr, Ix2, Axis, Ix1};
use dashmap::DashMap;

pub type Run = u8;
pub type GType = bool;

#[derive(Debug)]
/// This holds the maps from the run index (i.e. 1..=15) to
/// the positions, velocities and galaxy types.
pub struct Galaxies {
    pub position: DashMap<Run, Vec<P3>>,
    pub velocity: DashMap<Run, Vec<P3>>,
    pub gtypes: DashMap<Run, Vec<GType>>,
    pub(crate) dataset: Dataset,
}

/// Given an enum specify the dataset to be used, this loads in the
/// data and returns a map from the run to the positions or velocities
/// of the galaxies for that run.
pub fn load(dataset: Dataset) -> Result<Galaxies, Box<dyn Error>> {
    
    // Initialize inner maps
    let position = DashMap::new();
    let velocity = DashMap::new();
    let gtypes = DashMap::new();

    for run in 1..=15 {

        // Open an NpzReader for this rin
        let mut npz = NpzReader::new(
            std::fs::File::open(
                format!("{}/run{run}.npz", dataset.path())
            )?
        )?;

        // Read the Galaxy positions and velocities
        let pos: Vec<P3> = npz.by_name::<OwnedRepr<f64>, Ix2>("pos.npy")
            .unwrap()
            .map_axis(Axis(1), |pos| P3::new(pos[0], pos[1], pos[2]))
            .to_vec();
        let vel: Vec<P3> = npz.by_name::<OwnedRepr<f64>, Ix2>("vel.npy")
            .unwrap()
            .map_axis(Axis(1), |pos| P3::new(pos[0], pos[1], pos[2]))
            .to_vec();
        let gtp: Vec<GType> = npz.by_name::<OwnedRepr<f64>, Ix1>("gtype.npy")
            .unwrap()
            .map(|&x| {
                if x == 0.0 {
                    false
                } else if x == 1.0 {
                    true
                } else {
                    unreachable!("Only two types were present at time of writing")
                }
            })
            .to_vec();

        // Add these to map
        position.insert(run, pos);
        velocity.insert(run, vel);
        gtypes.insert(run, gtp);
    }

    Ok(Galaxies {
        position,
        velocity,
        gtypes,
        dataset,
    })
}

#[derive(Debug, Copy, Clone)]
/// Used to specify which dataset to use
pub enum Dataset {
    Challenge0Z0
}


impl Dataset {

    /// Returns the corresponding directory for the dataset
    pub fn path(&self) -> &'static str {
        match self {
            Dataset::Challenge0Z0 => "Challenge0_z0"
        }
    }
    /// Returns appropriate npz name for the dataset
    pub fn out(&self) -> &'static str {
        match self {
            Dataset::Challenge0Z0 => "challenge0_z0.npz"
        }
    }
}