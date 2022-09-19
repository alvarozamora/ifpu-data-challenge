use itertools::Itertools;

#[derive(Debug)]
pub struct InterpConfig {
    pub quantiles: Vec<f64>,
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