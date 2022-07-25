use colorous::{Color, SET1, SET2};
use dashmap::DashMap;
use itertools::Itertools;
use ndarray_npy::NpzReader;
use ndarray::Array1;
use plotters::style::Color as PlottersColor;
use std::collections::HashMap;
use std::ops::{DivAssign, AddAssign};
use interp1d::Interp1d;
use std::{fs::File, error::Error};
use plotters::prelude::*;
use log::*;

use ifpu_knns::{
    load_data::{Dataset, Run}, cdf::KNN
};

/// For this script, K must be greater than or equal to 4!
const K: usize = 4;

/// Directory where plots are stored
const PLOTS_DIR: &'static str = "./plots";
/// Name of first plot
const FIRST_PLOT_PREFIX: &'static str = "mean_diff";
/// Name of second plot
const SECOND_PLOT_PREFIX: &'static str = "all_four";

/// Number of points to use for interpolation
const INTERP_POINTS: usize = 200;

fn main() -> Result<(), Box<dyn Error>> {

    // Logging
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .write_style(env_logger::WriteStyle::Auto)
        .format_timestamp_secs()
        .init();
    
    // Load output: interpolated cdfs
    let cdfs = KnnResult::from_output::<K>(Dataset::Challenge0Z0)?;

    cdfs.plot()
}


type InterpolatedCDF = Array1<f64>;
type Distances = Array1<f64>;
type Quantiles = Array1<f64>;
struct KnnResult {
    cdfs: DashMap<(Run, KNN), InterpolatedCDF>,
    quantiles: Quantiles,
    dataset: Dataset,
}


impl KnnResult {
    
    pub fn from_output<const K: usize>(dataset: Dataset) -> Result<KnnResult, Box<dyn Error>> {

        trace!("getting knn outputs");

        // Load dataset
        let mut npz = NpzReader::new(File::open(dataset.out())?)?;

        // Initialize return value
        let cdfs = DashMap::new();

        for run in 1..=15 {
            for k in 1..=K as u16 {
                trace!("getting knn output ({run}, {k})");
                cdfs.insert((run, k), npz
                        .by_name(format!("run{run}_{k}").as_str())
                        .expect(format!("run{run}_{k} should exist").as_str()));
            }
        }

        trace!("getting quantiles");
        let quantiles = npz.by_name("quantiles").expect("quantiles should have been saved");

        Ok(KnnResult {
            cdfs,
            quantiles,
            dataset
        })
    }

    pub fn plot(&self) -> Result<(), Box<dyn Error>> {

        // Ensure directory where plots will be saved exists
        std::fs::create_dir_all(format!("{PLOTS_DIR}/{}", self.dataset))?;

        // Calculate the means of all knns measurements
        trace!("calculating means");
        let means: DashMap<KNN, (Distances, InterpolatedCDF)> = self.calculate_means();

        // Get run colors
        trace!("done calculating mean, getting run colors");
        let run_colors: HashMap<Run, RGBColor> = (1..=15)
            .map(|i| (i, get_color(i)))
            .collect();
        trace!("colors obtained");

        // First plot (one per k, differences from mean)
        trace!("starting first plot");
        for k in 1..=K as u16 {

            trace!("Working on k = {k} for first plot");

            // Path to save to
            let first_plot_file = format!("{PLOTS_DIR}/{}/{FIRST_PLOT_PREFIX}_{k}.svg", self.dataset);

            // Backend
            let root = SVGBackend::new(&first_plot_file, (1080, 720)).into_drawing_area();
            root.fill(&WHITE)?;

            // interpolation grid and means at those distances
            let (distances, k_mean) = &*means.get(&k).unwrap();
            trace!("retrieved mean with shapes ({}, {})", distances.len(), k_mean.len());

            // Initialize chart
            let dmin = distances.iter().sorted_by(|a, b| b.partial_cmp(&a).unwrap()).next_back().unwrap()*0.95;
            let dmax = distances.iter().sorted_by(|a, b| b.partial_cmp(&a).unwrap()).next().unwrap()*1.15;
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(50)
                .margin(5)
                .caption(format!("Difference from {k}NN mean"), ("sans-serif", 50.0).into_font())
                .build_cartesian_2d((dmin..dmax).log_scale(), -1.5e-3..1.5e-3_f64)?;
            trace!("built cartesian grid");
            chart
                .configure_mesh()
                // .disable_x_mesh()
                // .disable_y_mesh()
                .y_desc("Difference from mean")
                .y_label_formatter(&|x| format!("{:.2e}", x))
                .draw()?;
            trace!("configured mesh");

            for run in 1..=15 {

                let run_colors_ref = &run_colors;
                let line_color = run_colors_ref
                    .get(&run)
                    .expect("all colors should be initialized")
                    .stroke_width(3);
                trace!("obtained line color");

                // Retrieve measured cdfs for knns
                let measured_knn = self.cdfs
                    .get(&(run, k))
                    .expect("cdf should exist")
                    .clone();
                trace!("obtained measured knn ({run}, {k})");

                // Interpolator 
                let interpolator = Interp1d::new_sorted(
                    measured_knn.to_vec(),
                    self.quantiles.to_vec(),
                )?;
                
                let interpolated_knn = distances
                        .iter()
                        .cloned()
                        .map(|x| interpolator.interpolate_checked(x))
                        .collect::<Vec<Result<f64, _>>>();
                trace!("interpolated");
                
                trace!("adding line for run {run}, k = {k}");
                chart
                    .draw_series(LineSeries::new(
                        distances
                            .iter()
                            .cloned()
                            .zip(interpolated_knn.into_iter().zip(k_mean))
                            .filter_map(|(x, (y1, &y_mean))| {
                                y1.map(|y_1| (x, y_1-y_mean)).ok()
                            }),
                        line_color,
                    ))?
                    .label(format!("run_{run}"))
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], run_colors_ref.get(&run).unwrap())
                    });
            }

            chart
                .configure_series_labels()
                .background_style(&RGBColor(128, 128, 128))
                .draw()?;
    
            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect(format!("Unable to write {first_plot_file} to file").as_str());
            println!("Plot saved to {}", first_plot_file);
        }

        // plot 2 (one per run, all k shown)
        trace!("starting second plot");
        for run in 1..=15 {

            trace!("working on run {run} for second plot");

            let second_plot_file = format!("{PLOTS_DIR}/{}/{SECOND_PLOT_PREFIX}_{run}.svg", self.dataset);

            // Backend
            let root = SVGBackend::new(&second_plot_file, (1920, 1080)).into_drawing_area();
            root.fill(&WHITE)?;
    
            // Initialize chart
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(40)
                .margin(5)
                .caption("Gaussian Expectations", ("sans-serif", 50.0).into_font())
                .build_cartesian_2d((1.0f64..5e2f64).log_scale(), (self.quantiles[0]..1.0).log_scale())?;
            chart
                .configure_mesh()
                // .disable_x_mesh()
                // .disable_y_mesh()
                .y_desc("Peaked CDF")
                .y_label_formatter(&|x| format!("{:.2e}", x))
                .draw()?;
            
            // Retrieve measured cdfs for knns
            let measured_1nn = self.cdfs.get(&(run, 1))
                .expect("cdf should exist")
                .clone();
            let measured_2nn = self.cdfs.get(&(run, 2))
                .expect("cdf should exist")
                .clone();
            let measured_3nn = self.cdfs.get(&(run, 3))
                .expect("cdf should exist")
                .clone();
            let measured_4nn = self.cdfs.get(&(run, 4))
                .expect("cdf should exist")
                .clone();
    
            // Calculated expected GRF results. 
            // Note: this uses expected_3nn for 4nn, which would be the gaussian result
            // First need to interpolate 1nn and 2nn to the same grid
            let interpolator_1nn = Interp1d::new_unsorted(
                measured_1nn.to_vec(),
                self.quantiles.to_vec(), // this clones, maybe faster than .clone().into_raw_vec()
            )?;
            let interpolator_2nn = Interp1d::new_unsorted(
                measured_2nn.to_vec(),
                self.quantiles.to_vec(), // this clones, maybe faster than .clone().into_raw_vec()
            )?;
            let rmin = measured_4nn
                .iter()
                .sorted_by(|a,b| b.partial_cmp(&a).unwrap())
                .next_back()
                .unwrap();
            let rmax = measured_1nn.iter()
                .sorted_by(|a,b| b.partial_cmp(&a).unwrap())
                .next()
                .unwrap();
            let interp_grid = (0..INTERP_POINTS)
                .map(|n| {
                    rmin + n as f64 * (rmax - rmin) / (INTERP_POINTS - 1).max(1) as f64
                })
                .collect_vec();
            let interp_1nn: InterpolatedCDF = interp_grid
                .iter()
                .cloned()
                .map(|r|{
                    interpolator_1nn.interpolate_checked(r).expect("should be in domain")
                })
                .collect();
            let interp_2nn: InterpolatedCDF = interp_grid
                .iter()
                .cloned()
                .map(|r| {
                    interpolator_2nn.interpolate_checked(r).expect("should be in domain")
                })
                .collect();
            let gaussian_3nn = cdf_3nn_from_prev(
                &interp_1nn,
                &interp_2nn,
            );
            let gaussian_4nn = cdf_4nn_from_prev(
                &interp_1nn,
                &interp_2nn,
                &gaussian_3nn
            );

            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned().map(|x| pcdf(x))
                        .zip((measured_1nn).into_iter())
                        .map(|(a, b)| (b, a)),
                    get_color(1).stroke_width(3),
                ))?
                .label("Measured 1NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(1)));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned().map(|x| pcdf(x))
                        .zip((measured_2nn).into_iter())
                        .map(|(a, b)| (b, a)),
                    get_color(2).stroke_width(3),
                ))?
                .label("Measured 2NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(2)));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned().map(|x| pcdf(x))
                        .zip((measured_3nn).into_iter())
                        .map(|(a, b)| (b, a)),
                    get_color(3).stroke_width(3),
                ))?
                .label("Measured 3NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(3)));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned().map(|x| pcdf(x))
                        .zip((measured_4nn).into_iter())
                        .map(|(a, b)| (b, a)),
                    get_color(4).stroke_width(3),
                ))?
                .label("Measured 4NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(4)));
            chart
                .draw_series(LineSeries::new(
                    interp_grid
                        .iter()
                        .cloned()
                        .zip((gaussian_3nn).into_iter().map(|x| pcdf(x))),
                    get_color(5).stroke_width(3),
                ))?
                .label("GRF 3NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(5)));
            chart
                .draw_series(LineSeries::new(
                    interp_grid
                        .iter()
                        .cloned()
                        .zip((gaussian_4nn).into_iter().map(|x| pcdf(x))),
                    get_color(6).stroke_width(3),
                ))?
                .label("GRF 4NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(6)));
            
            chart
                .configure_series_labels()
                .background_style(&RGBColor(128, 128, 128))
                .draw()?;
    
            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect(format!("Unable to write {second_plot_file} to file").as_str());
            println!("Plot saved to {}", second_plot_file);
        }
        
    
    
        Ok(())
    }

    fn calculate_means(&self) -> DashMap<KNN, (Distances, InterpolatedCDF)> {
        
        let (rmin, rmax) = (1..=15)
            .map(|i| {

                // Get smallest 1NN max
                let one_nn = self.cdfs.get(&(i, 1)).unwrap();
                let max = *one_nn
                    .iter()
                    .sorted_by(|a, b| b.partial_cmp(&a).unwrap())
                    .next()
                    .unwrap();

                // Get largest 4NN min
                let four_nn = self.cdfs.get(&(i, 4)).unwrap();
                let min = *four_nn
                    .iter()
                    .sorted_by(|a, b| b.partial_cmp(&a).unwrap())
                    .next_back()
                    .unwrap();
                trace!("run {i} has min/max {min:.2}/{max:2}");

                (min, max)
            })
            .fold((std::f64::MIN, std::f64::MAX), |mut acc, x| {
                if acc.0 < x.0 {
                    acc.0 = x.0
                } else if acc.1 > x.1 {
                    acc.1 = x.1
                }
                acc
            });
        trace!("overall min/max is {rmin:.2}/{rmax:.2}");

        // Initialize return value
        let means = DashMap::<KNN, (Distances, InterpolatedCDF)>::new();

        for k in 1..=K as u16 {
            for run in 1..=15 {

                // Retrieve measured cdf
                let measured_knn = self.cdfs.get(&(run, k)).expect("cdf should exist");
                
                // means.get_mut(&k).unwrap().add_assign(&*measured_knn);

                // Interpolate to min_max range
                assert!(INTERP_POINTS > 2, "at least two points");
                let interp_grid: Distances = Distances::from_vec(
                    (0..INTERP_POINTS)
                        .map(|n| {
                            rmin + n as f64 * (rmax - rmin) / (INTERP_POINTS - 1) as f64
                        })
                        .collect_vec()
                );
                trace!("constructed interp_grid");
                let interpolator_knn = Interp1d::new_sorted(
                    measured_knn.to_vec(),
                    self.quantiles.to_vec(),
                ).unwrap();
                trace!("constructed Interp1d");
                let interp_knn: (Vec<f64>, Vec<f64>) = interp_grid
                    .iter()
                    .flat_map(|x| {
                        let interp = interpolator_knn.interpolate_checked(*x);
                        if let Ok(value) = interp {
                            trace!("interp: ({x}, {value})");
                            Some((*x, value))
                        } else {
                            None
                        }
                    })
                    .unzip();
                trace!("Interpolate checked");
                let interp_knn = (
                    Array1::from_vec(interp_knn.0),
                    Array1::from_vec(interp_knn.1),
                );

                means.entry(k)
                    .and_modify(|cdf| {
                        trace!("modifying ({run},{k}) with shape {}", cdf.1.len());
                        cdf.1.add_assign(&interp_knn.1); 
                    })
                    .or_insert((interp_knn.0, interp_knn.1));
            }

            // Divide by total to get mean
            means
                .get_mut(&k)
                .unwrap().1
                .div_assign(15.0);
        }

        means
    }
}


#[allow(unused)]
fn cdf_2nn_from_1nn(
    cdf_1nn: &InterpolatedCDF
) -> InterpolatedCDF {
    cdf_1nn + (1.0 - cdf_1nn) * (1.0 - cdf_1nn).mapv(f64::ln)
}


fn cdf_3nn_from_prev(
    cdf_1nn: &InterpolatedCDF,
    cdf_2nn: &InterpolatedCDF,
) -> InterpolatedCDF {
    cdf_2nn + (
        (1.0 - cdf_1nn) * (1.0 - cdf_1nn).mapv(f64::ln)
        + (cdf_1nn - cdf_2nn)
        - 1.0/2.0*(cdf_1nn - cdf_2nn).map(|x| x.powi(2))/(1.0 - cdf_1nn)
    )
}

fn cdf_4nn_from_prev(
    cdf_1nn: &InterpolatedCDF,
    cdf_2nn: &InterpolatedCDF,
    cdf_3nn: &InterpolatedCDF,
) -> InterpolatedCDF {
    cdf_3nn + (cdf_1nn - cdf_2nn)/(1.0 - cdf_1nn) * (
        (1.0 - cdf_1nn)*(1.0 - cdf_1nn).mapv(f64::ln)
        + (cdf_1nn - cdf_2nn)
        - 1.0/6.0 * (cdf_1nn-cdf_2nn).map(|x| x.powi(2))/(1.0 - cdf_1nn)
    )
}





/// Assuming its i <= 15
// fn get_color(i: u8) -> ShapeStyle {
fn get_color(i: u8) -> RGBColor {

    // Zero-based index
    let i: usize = i.checked_sub(1).unwrap().into();
    assert!(i < 15, "This function was made for 15 categories/runs");

    let color: Color = if i < 9 {
        SET1[i]
    } else {
        SET2[i-9]
    };

    // let color: Color = VIRIDIS.eval_rational(i as usize, total as usize);

    RGBColor(color.r, color.g, color.b)

}

#[inline(always)]
fn pcdf(x: f64) -> f64 {
    if x <= 0.5 { x } else { 1.0 - x }
}