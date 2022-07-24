use dashmap::DashMap;
use ndarray_npy::NpzReader;
use ndarray::Array1;
use plotters::style::full_palette::{BLUE_100, ORANGE, GREEN_100, ORANGE_100};
// use std::ops::AddAssign;
use std::ops::DivAssign;

use std::{fs::File, error::Error};
use plotters::prelude::*;

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

fn main() -> Result<(), Box<dyn Error>> {

    // Load output: interpolated cdfs
    let cdfs = KnnResult::from_output::<K>(Dataset::Challenge0Z0)?;

    cdfs.plot()
}


type InterpolatedCDF = Array1<f64>;
type Quantiles = Array1<f64>;
struct KnnResult {
    cdfs: DashMap<(Run, KNN), InterpolatedCDF>,
    quantiles: Quantiles,
    dataset: Dataset,
}


impl KnnResult {
    
    pub fn from_output<const K: usize>(dataset: Dataset) -> Result<KnnResult, Box<dyn Error>> {

        // Load dataset
        let mut npz = NpzReader::new(File::open(dataset.out())?)?;

        // Initialize return value
        let cdfs = DashMap::new();

        for run in 1..15_u8 {
            for k in 1..K as u16 {
                cdfs.insert((run, k), npz.by_name("run{run}_{k}").expect("cdf should exist"));
            }
        }

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
        let means: DashMap<KNN, InterpolatedCDF> = self.calculate_means();

        // First plot (one per k, differences from mean)
        for k in 1..=K as u16 {

            // Path to save to
            let first_plot_file = format!("{PLOTS_DIR}/{}/{FIRST_PLOT_PREFIX}_{k}", self.dataset);

            // Backend
            let root = BitMapBackend::new(&first_plot_file, (1024, 768)).into_drawing_area();
            root.fill(&WHITE)?;

            // Initialize chart
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(40)
                .margin(5)
                .caption("Difference from mean", ("sans-serif", 50.0).into_font())
                .build_cartesian_2d(self.quantiles[0]..1.0, (0.1f64..1e10f64).log_scale())?;
            chart
                .configure_mesh()
                // .disable_x_mesh()
                // .disable_y_mesh()
                .y_desc("Difference from mean")
                .y_label_formatter(&|x| format!("{:e}", x))
                .draw()?;

            let k_mean = &*means.get(&k).unwrap();

            for run in 1..=15 {
            
                // Retrieve measured cdfs for knns
                let measured_knn = self.cdfs.get(&(run, k)).expect("cdf should exist").clone();

                chart
                    .draw_series(LineSeries::new(
                        self.quantiles.iter().cloned()
                            .zip((measured_knn-k_mean).into_iter()),
                        &BLUE,
                    ))?
                    .label("y = 1.02^x^2")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
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
        for run in 1..=15 {

            let second_plot_file = format!("{PLOTS_DIR}{}{SECOND_PLOT_PREFIX}", self.dataset);

            // Backend
            let root = BitMapBackend::new(&second_plot_file, (1024, 768)).into_drawing_area();
            root.fill(&WHITE)?;
    
            // Initialize chart
            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(40)
                .margin(5)
                .caption("Difference from mean", ("sans-serif", 50.0).into_font())
                .build_cartesian_2d(self.quantiles[0]..1.0, (0.1f64..1e10f64).log_scale())?;
            chart
                .configure_mesh()
                // .disable_x_mesh()
                // .disable_y_mesh()
                .y_desc("Difference from mean")
                .y_label_formatter(&|x| format!("{:e}", x))
                .draw()?;
            
            // Retrieve measured cdfs for knns
            let measured_1nn = self.cdfs.get(&(run, 1)).expect("cdf should exist").clone();
            let measured_2nn = self.cdfs.get(&(run, 2)).expect("cdf should exist").clone();
            let measured_3nn = self.cdfs.get(&(run, 3)).expect("cdf should exist").clone();
            let measured_4nn = self.cdfs.get(&(run, 4)).expect("cdf should exist").clone();
    
            // Calculated expected GRF results. 
            // Note: this uses expected_3nn for 4nn, which would be the gaussian result
            let gaussian_3nn = cdf_3nn_from_prev(&measured_1nn, &measured_2nn);
            let gaussian_4nn = cdf_4nn_from_prev(&measured_1nn, &measured_2nn, &gaussian_3nn);

            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((measured_1nn).into_iter()),
                    &BLUE,
                ))?
                .label("Measured 1NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((measured_2nn).into_iter()),
                    &RED,
                ))?
                .label("Measured 2NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((measured_3nn).into_iter()),
                    &GREEN,
                ))?
                .label("Measured 3NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((measured_4nn).into_iter()),
                    &ORANGE,
                ))?
                .label("Measured 4NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((gaussian_3nn).into_iter()),
                    &GREEN_100,
                ))?
                .label("GRF 3NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN_100));
            chart
                .draw_series(LineSeries::new(
                    self.quantiles.iter().cloned()
                        .zip((gaussian_4nn).into_iter()),
                    &ORANGE_100,
                ))?
                .label("GRF 4NN")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE_100));
            
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

    fn calculate_means(&self) -> DashMap<KNN, InterpolatedCDF> {
        
        // Initialize return value
        let means = DashMap::<KNN, InterpolatedCDF>::new();

        for k in 1..=K as u16 {
            for run in 1..=15 {

                // Retrieve measured cdf
                let measured_knn = self.cdfs.get(&(run, 1)).expect("cdf should exist");
                
                // means.get_mut(&k).unwrap().add_assign(&*measured_knn);
                *means.get_mut(&k).unwrap() += &*measured_knn;
            }
            means.get_mut(&k).unwrap().div_assign(15.0);
        }

        means
    }
}



fn cdf_2nn_from_1nn(
    cdf_1nn: &InterpolatedCDF
) -> InterpolatedCDF {
    cdf_1nn + (1.0 - cdf_1nn) * (1.0 - cdf_1nn).mapv(f64::ln)
}


fn cdf_3nn_from_prev(
    cdf_1nn: &InterpolatedCDF,
    cdf_2nn: &InterpolatedCDF,
) -> InterpolatedCDF {
    cdf_2nn + ( (1.0 - cdf_1nn) * (1.0 - cdf_1nn).mapv(f64::ln) + (cdf_1nn - cdf_2nn) - 
                    1.0/2.0*(cdf_1nn - cdf_2nn).map(|x| x.powi(2))/(1.0 - cdf_1nn) )
}

fn cdf_4nn_from_prev(
    cdf_1nn: &InterpolatedCDF,
    cdf_2nn: &InterpolatedCDF,
    cdf_3nn: &InterpolatedCDF,
) -> InterpolatedCDF {
    cdf_3nn + (cdf_1nn - cdf_2nn)/(1.0 - cdf_1nn) * ( (1.0 - cdf_1nn)*(1.0 - cdf_1nn).mapv(f64::ln) + (cdf_1nn - cdf_2nn)
                            - 1.0/6.0 * (cdf_1nn-cdf_2nn).map(|x| x.powi(2))/(1.0 - cdf_1nn))
}
