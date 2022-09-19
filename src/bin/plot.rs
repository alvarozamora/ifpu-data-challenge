use colorous::{Color, SET2, PAIRED};
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
const K: usize = 8;

/// Directory where plots are stored
const PLOTS_DIR: &'static str = "./plots";
/// Name of first plot
const FIRST_PLOT_PREFIX: &'static str = "mean_diff";
/// Name of second plot
const SECOND_PLOT_PREFIX: &'static str = "all";

/// Number of points to use for interpolation
const INTERP_POINTS: usize = 200;

/// Whether to use DD
const USE_DD: bool = true;

fn main() -> Result<(), Box<dyn Error>> {

    // Logging
    env_logger::builder()
        .filter_level(LevelFilter::Trace)
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
        let mut npz = NpzReader::new(
            if USE_DD {
                File::open(format!("{}_DD", dataset.out()))?
            } else {
                File::open(dataset.out())?
            }
        )?;

        // Initialize return value
        let cdfs = DashMap::new();

        for run in 1..=15 {
            for k in 1..=K as u16 {
                trace!("getting knn output ({run}, {k})");
                cdfs.insert((run, k), npz
                        .by_name(format!("run{run}_{k}{}", if USE_DD { "_DD" } else { "" }).as_str())
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
            let dmin = distances.iter().sorted_by(|a, b| a.partial_cmp(&b).unwrap()).next().unwrap()*0.95;
            let dmax = distances.iter().sorted_by(|a, b| a.partial_cmp(&b).unwrap()).next_back().unwrap()*1.15;
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(40)
                .y_label_area_size(100)
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
        let mut diff_3 = vec![];
        let mut diff_4 = vec![];
        for run in 1..=15 {

            trace!("working on run {run} for second plot");

            let second_plot_file = format!("{PLOTS_DIR}/{}/{SECOND_PLOT_PREFIX}_{run}.svg", self.dataset);

            // Backend
            let root = SVGBackend::new(&second_plot_file, (1920, 1080)).into_drawing_area();
            root.fill(&WHITE)?;
    
            // Initialize chart
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(40)
                .y_label_area_size(100)
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
            let interpolator_3nn = Interp1d::new_unsorted(
                measured_3nn.to_vec(),
                self.quantiles.to_vec(), // this clones, maybe faster than .clone().into_raw_vec()
            )?;
            let interpolator_4nn = Interp1d::new_unsorted(
                measured_4nn.to_vec(),
                self.quantiles.to_vec(), // this clones, maybe faster than .clone().into_raw_vec()
            )?;
            let rmin = measured_4nn
                .iter()
                .sorted_by(|a,b| a.partial_cmp(&b).unwrap())
                .next()
                .unwrap();
            let rmax = measured_1nn.iter()
                .sorted_by(|a,b| a.partial_cmp(&b).unwrap())
                .next_back()
                .unwrap();
            let interp_grid = (1..INTERP_POINTS-1)
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
            let interp_3nn: InterpolatedCDF = interp_grid
                .iter()
                .cloned()
                .map(|r|{
                    interpolator_3nn.interpolate_checked(r).expect("should be in domain")
                })
                .collect();
            let interp_4nn: InterpolatedCDF = interp_grid
                .iter()
                .cloned()
                .map(|r| {
                    interpolator_4nn.interpolate_checked(r).expect("should be in domain")
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
            diff_3.push((interp_grid.clone(), &interp_3nn - &gaussian_3nn));
            diff_4.push((interp_grid.clone(), &interp_4nn - &gaussian_4nn));

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

            if K > 4 {
                for k in 5..=K {

                    let measured_knn = self.cdfs.get(&(run, k as u16))
                        .expect("cdf should exist")
                        .clone();
                    chart
                        .draw_series(LineSeries::new(
                            self.quantiles
                                .iter()
                                .cloned()
                                .map(|x| pcdf(x))
                                .zip((measured_knn).into_iter())
                                .map(|(y, x)| (x, y)),
                            get_color(k as u8 + 2).stroke_width(3),
                        ))?
                        .label(format!("Measured {k}NN"))
                        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(k as u8 + 2)));
                }
            }
            
            chart
                .configure_series_labels()
                .background_style(&RGBColor(128, 128, 128))
                .draw()?;
    
            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect(format!("Unable to write {second_plot_file} to file").as_str());
            println!("Plot saved to {}", second_plot_file);
        }

        trace!("starting third plot");
        let diffs = [(3, diff_3), (4, diff_4)];
        for (k, interp_grid_and_diff) in diffs {

            // Backend
            let third_plot_file = format!("{PLOTS_DIR}/{}/GRF_diff_{k}.svg", self.dataset);
            let root = SVGBackend::new(&third_plot_file, (1920, 1080)).into_drawing_area();
            root.fill(&WHITE)?;

            // Initialize chart
            let (mut ymin, mut ymax) = interp_grid_and_diff
                .iter()
                .fold((std::f64::MAX, std::f64::MIN), |mut acc, (_, diff)| {
                    let mut sorted_iter = diff
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(&b).expect("all values should be finite"));
                    let ymax: f64 = *sorted_iter.next_back().unwrap();
                    let ymin: f64 = *sorted_iter.next().unwrap();
        
                    if acc.0 > ymin { acc.0 = ymin }
                    if acc.1 < ymax { acc.1 = ymax }
                    acc
                });
            if ymin < 0.0 { ymin *= 1.05 } else { ymin *= 0.95 };
            if ymax < 0.0 { ymax *= 0.95 } else { ymax *= 1.05 };
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(40)
                .y_label_area_size(100)
                .margin(5)
                .caption("Gaussian Expectation Diff", ("sans-serif", 50.0).into_font())
                .build_cartesian_2d((2e1f64..3e2f64).log_scale(), ymin..ymax)?;
            chart
                .configure_mesh()
                // .disable_x_mesh()
                // .disable_y_mesh()
                .y_desc("Measured-Expected")
                .y_label_formatter(&|x| format!("{:.2e}", x))
                .draw()?;

            
            for run in 1..=15 {

                // Zero-based index
                let run_idx = run - 1;

                let (interp_grid, diff) = interp_grid_and_diff.get(run_idx).expect("should exist");

                chart
                    .draw_series(LineSeries::new(
                        interp_grid
                            .into_iter()
                            .zip(diff)
                            .map(|(x, y)| {
                                (*x, *y)
                            }),
                        get_color(run as u8 + 2).stroke_width(3),
                    ))?
                    .label(format!("run {run}"))
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(run as u8 + 2)));
            }
            chart
                .configure_series_labels()
                .background_style(&RGBColor(128, 128, 128))
                .draw()?;
    
            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect(format!("Unable to write {third_plot_file} to file").as_str());
            println!("Plot saved to {}", third_plot_file);
        }

        trace!("starting fourth plot");
        let mut mins = vec![];
        let mut maxs = vec![];
        let xy_pairs: Vec<(InterpolatedCDF, InterpolatedCDF)> = (1..=15)
            .map(|run| {

            let measured_2nn = self.cdfs
                .get(&(run, 2))
                .expect("cdf should exist");
            let measured_knn = self.cdfs
                .get(&(run, K as u16))
                .expect("cdf should exist");
            let rmin = *measured_2nn
                .iter()
                .sorted_by(|a,b| a.partial_cmp(&b).unwrap())
                .next()
                .unwrap();
            let rmax = *measured_knn
                .iter()
                .sorted_by(|a,b| a.partial_cmp(&b).unwrap())
                .next()
                .unwrap();
            maxs.push(rmax);
            mins.push(rmin);

            let interp_grid: Distances = Distances::from_vec(
                (0..INTERP_POINTS)
                    .map(|n| {
                        rmin + n as f64 * (rmax - rmin) / (INTERP_POINTS - 1) as f64
                    })
                    .collect_vec()
            );

            let interp_knns: Vec<InterpolatedCDF> = (1..=K)
                .map(|k| {
                    let measured_knn = self.cdfs
                        .get(&(run, k as u16))
                        .expect("cdf should exist")
                        .clone();
                    let interpolator = Interp1d::new_sorted(
                        measured_knn.into_raw_vec(),
                        self.quantiles.clone().into_raw_vec()
                    ).expect("constructing interpolator shouldn't fail");

                    InterpolatedCDF::from_vec(
                        interp_grid
                            .iter()
                            .cloned()
                            .map(|r| {
                                interpolator.interpolate(r)
                            }).collect::<Vec<_>>())
                }).collect();

            let third_cumulant = third_cumulant_approximated_from_first_k(interp_knns);
            (interp_grid, third_cumulant)
        }).collect();

        // Backend
        let fourth_plot_file = format!("{PLOTS_DIR}/{}/third_cumulant.svg", self.dataset);
        let root = SVGBackend::new(&fourth_plot_file, (1080, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        // Initialize chart
        let dmin = *mins.iter().sorted_by(|a, b| a.partial_cmp(&b).expect("all distances should be finite")).next().unwrap();
        let dmax = *maxs.iter().sorted_by(|a, b| a.partial_cmp(&b).expect("all distances should be finite")).next_back().unwrap();
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(40)
            .y_label_area_size(100)
            .margin(5)
            .caption(format!("Third Cumulant"), ("sans-serif", 50.0).into_font())
            .build_cartesian_2d((dmin..dmax).log_scale(), (1e-3..10.0_f64).log_scale())?;
        trace!("built cartesian grid");
        chart
            .configure_mesh()
            // .disable_x_mesh()
            // .disable_y_mesh()
            .y_desc("Third Cumulant")
            .y_label_formatter(&|x| format!("{:.2e}", x))
            .draw()?;
        trace!("configured mesh");

        for (xy_pair, i) in xy_pairs.into_iter().zip(1..) {
            chart
            .draw_series(LineSeries::new(
                xy_pair.0.into_iter().zip(xy_pair.1),
                get_color(i as u8).stroke_width(3),
            ))?
            .label(format!("run {i}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_color(i as u8)));
        }

        chart
            .configure_series_labels()
            .background_style(&RGBColor(128, 128, 128))
            .draw()?;

        root.present().expect(format!("Unable to write {fourth_plot_file} to file").as_str());
        println!("Plot saved to {}", fourth_plot_file);
    
        Ok(())
    }

    fn calculate_means(&self) -> DashMap<KNN, (Distances, InterpolatedCDF)> {
        
        let (rmin, rmax) = (1..=15)
            .map(|i| {

                // Get smallest 1NN max
                let one_nn = self.cdfs.get(&(i, 1)).unwrap();
                let max = *one_nn
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .next_back()
                    .unwrap();

                // Get largest KNN min
                let k_nn = self.cdfs.get(&(i, K as u16)).unwrap();
                let min = *k_nn
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .next()
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

#[allow(unused)]
/// This assumes that all the cdfs are interpolated to a common distance grid
fn third_cumulant_approximated_from_first_k(
    cdf_knns: Vec<InterpolatedCDF>,
) -> InterpolatedCDF {

    // Get num of points
    let num: usize = cdf_knns
        .get(0)
        .unwrap()
        .len();

    // Note: for cumulants, k*Pk = 0 for k = 0 so we don't need it
    let mut pks: Vec<InterpolatedCDF> = cdf_knns
        .windows(2) 
        .map(|x|{

            // Banerjee & Abel 2020 Eq 8
            // Pk = CDF_k - CDF_k+1
            &x[0] - &x[1]

        }).collect_vec();

    // Third cumulant is given by
    // 2 * sum(k Pk)^3 - 3 * sum(k Pk) * sum(k^2 Pk) + sum(k^3 Pk)
    use std::ops::Add;
    let result = {
        let term_1 = 2.0 * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + k as f64 * pk)
            .mapv(|x| x.powi(3));
        let term_2 = 3.0 * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + k as f64 * pk)
            * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + (k as f64).powi(2) * pk);
        let term_3 = pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + (k as f64).powi(3) * pk);

        term_1 - term_2 + term_3
    };

    let _ = pks.pop();
    let result_with_one_less_k = {
        let term_1 = 2.0 * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + k as f64 * pk)
            .mapv(|x| x.powi(3));
        let term_2 = 3.0 * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + k as f64 * pk)
            * pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + (k as f64).powi(2) * pk);
        let term_3 = pks.iter()
            .zip(1..)
            .fold(InterpolatedCDF::zeros(num), |acc, (pk, k)| acc + (k as f64).powi(3) * pk);

        term_1 - term_2 + term_3
    };

    info!(
        "percent_diff of third cumulant with one less k: {:?}",
        (&result - result_with_one_less_k) / &result
    );

    result
}

/// Assuming its i <= 15
// fn get_color(i: u8) -> ShapeStyle {
fn get_color(i: u8) -> RGBColor {

    // Zero-based index
    let mut i: usize = i.checked_sub(1).unwrap().into();
    // Skip yellow due to visibility
    if i >= 10 { i+= 1; }
    assert!(i < 19, "This function was made for 19 categories/runs");

    let color: Color = if i < 12 {
        PAIRED[i]
    } else {
        SET2[i-12]
    };

    // let color: Color = VIRIDIS.eval_rational(i as usize, total as usize);

    RGBColor(color.r, color.g, color.b)

}

#[inline(always)]
fn pcdf(x: f64) -> f64 {
    if x <= 0.5 { x } else { 1.0 - x }
}