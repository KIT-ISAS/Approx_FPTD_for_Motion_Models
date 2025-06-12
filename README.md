# Approx FPTD for Motion Models

<img align="right" height="250" src="https://github.com/KIT-ISAS/Approx_FPTD_for_Motion_Models/blob/master/experiments/for_paper/CV_Long_Track_Sw10_denorm/cv_long_track_sw10_denorm_fptd.png">

Code belonging to the paper

*Marcel Reith-Braun, Jakob Thumm, Florian Pfaff, und Uwe D. Hanebeck, "Approximate First Passage Time Distributions for Gaussian Motion and Transportation Models," (submitted to Fusion), 2023.*

The repo contains methods to compute approximate first-passage time distributions (FPTD) for Gaussian processes with an increasing trend, such as motion and transportation models for which it is valid to assume that the increasing trend in the mean is primarily resposible for the first-passage. This may include, e.g., models such as the *constant velocity (CV) model*, and the *constant acceleration (CA) model*.

The repo contains examples for three different process, the *Wiener process with drift*, for which the analytical solution is known, the *CV model* and the *CA model*. The methods are compared with Monte Carlo simulations of the corresponding discrete-time process model.

Note that the code is intended to support two-dimensional processes, where one is also interested in the distribution of *y* at the first-passage time. The *CV* and *CA* models are therefore defined as 2D processes, where the process in *x* is independent of the process in *y*.

## Prerequisites

See `dockerfile/Dockerfile` for a list of required packages. Use

  ```shell script
docker build -t tensorflow/approx_fptd:2.8.0-gpu /path/to/repo/dockerfile
  ```

to build a docker image with all required packages.

### Run with CPU (Docker, Linux)

  ```shell script
docker run -u $(id -u):$(id -g) -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/repo:/mnt tensorflow/approx_fptd:2.8.0-gpu
  ```

### Run with GPU (Docker, Linux)

  ```shell script
docker run -u $(id -u):$(id -g) --gpus all -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/repo:/mnt tensorflow/approx_fptd:2.8.0-gpu
  ```

## Executable scripts 

### `wiener_process_main.py`

- Methods and simulations for a first-passage time problem with a *Wiener process with drift*, for which the analytical solution is known.
### `cv_process_main.py`

- Methods and simulations for a first-passage time problem with a *CV model*.

### `ca_process_main.py`

- Methods and simulations for a first-passage time problem with a *CA model*.

### `cv_experiments_main.py`

- Runs predefined experiments with *CV models*. The experiments are defined via a hard-coded config dictionary in the file.

### `ca_experimets_main.py`

- Runs predefined experiments with *CA models*. The experiments are defined via a hard-coded config dictionary in the file.


### Example Run Commands ###

- Get a list and explanations of all possible flags.


  ```shell script
	 python3 /mnt/cv_process_main.py --help
  ```
  Similar for all other executables.
  
- Run a single experiment with a *CV model* with predefined settings (see the main function of `cv_experiments_main.py`) and save the plots to `/mnt/results/`.

  ```shell script
	 python3 /mnt/cv_process_main.py --save_results --result_dir /mnt/results/
  ```
  Similar for `ca_process_main.py` and `wiener_process_main.py`.
  
- Run a single experiment with a *CV model* with predefined settings (see the main function of `cv_experiments.py`), save the plots to `/mnt/results/`, and save stdout to `/mnt/results/stdout_cv_model.txt`.

  ```shell script
	 python3 /mnt/cv_process_main.py --save_results --result_dir /mnt/results/ > /mnt/results/stdout_cv_model.txt
  ```
  Similar for `ca_process_main.py` and `wiener_process_main.py`.
  
- Run predefined experiments with *CV models*, save the plots to `/mnt/results/cv_experiments/`, and save stdout to `/mnt/results/cv_experiments/stdout_cv_models.txt`, but do not show the plots.

  ```shell script
	 python3 /mnt/cv_experiments_main.py --save_results --result_dir /mnt/results/cv_experimenents/ --no_show > /mnt/results/cv_experimenents/stdout_cv_model.txt
  ```
  
  Similar for `ca_experiments_main.py`. Experiment configs and chosen experiments can be adjusted in `cv_experiments_main.py` and `ca_experiments_main.py`, respectively.

- Run a single experiment with a *CV model* with predefined settings, measure the computational times (averaged over ten runs) and do not save or show the plots (measured times will be printed to stdout).

  ```shell script
	 python3 /mnt/cv_process_main.py --measure_computational_times --no_show
  ```
  Similar for `ca_process_main.py` and `wiener_process_main.py`.

## Import the Repo as a Module & Build the Distributions

The distributions can be imported from the module `Approx_FPDT_for_Motion_Models` (for this, make sure you added
`/path/to/repo` to your `$PYTHONPATH`).


_Example:_

  ```shell script
from Approx_FPDT_for_Motion_Models import GaussTaylorCVHittingTimeDistribution

# define hyperparameters for the distribution...
S_w = 10  # power spectral density (PSD)
x_L = np.array([0.3, 6.2, 0.5, 0.2])  # initial state (x, x_dot, y, y_dot)
C_L = np.array([[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]])  # initial covariance
x_predTo = 0.6458623971412047  # boundary to hit (x-coordinate)
t_L = 0  # initial time
cv_temporal_point_predictor = lambda pos_l, v_l, x_predTo: (x_predTo - pos_l[..., 0]) / v_l[..., 0]  
# constant-velocity prediction
    
# build the distribution...
htd = GaussTaylorCVHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L,
                                           point_predictor=cv_temporal_point_predictor)
# evaluate the distribution, e.g., 
htd.cdf(0.05)   
# or, for a larger sample_size, 
htd.cdf(np.array([0.04, 0.05, 0.06, 0.07]))       
  ```

The distributions support a batch size. 

_Example:_

  ```shell script
from Approx_FPDT_for_Motion_Models import GaussTaylorCVHittingTimeDistribution

# define hyperparameters for the distribution...
S_w = np.stack([S_w, 10**2 * S_w])   # the process at the second batch has a PSD of 100 times the PSD of the first batch 
x_L = np.stack([x_L, x_L])
C_L = np.stack([C_L, C_L])
x_predTo = np.stack([x_predTo, x_predTo])
    
# build the distribution...
htd = GaussTaylorCVHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L,
                                           point_predictor=cv_temporal_point_predictor)
# evaluate the distribution, e.g., 
htd.cdf(0.05) >> array([0.07154484, 0.44075421])
# or, for a larger sample_size, 
cdf_values = htd.cdf(np.array([[0.04], [0.05], [0.06], [0.07]]))     
cdf_values.shape >> (4, 2)      
cdf_values = htd.cdf(np.array([[0.04, 0.01], [0.05, 0.04], [0.06, 0.07], [0.07, 0.1]]))    
cdf_values.shape >> (4, 2)      
  ```


## Example Results

**First Passage Time**

*CV model* with a power spectral density of 93 640 mm<sup>2</sup>s<sup>-3</sup> (experiment *Long_Track_Sw10_denorm* in `cv_experiments.py`).
 
![alt text](https://github.com/KIT-ISAS/Approx_FPTD_for_Motion_Models/blob/master/experiments/for_paper/CV_Long_Track_Sw10_denorm/cv_long_track_sw10_denorm_fptd.png)

*Wiener process with drift* with sigma of 10 (default of `wiener_process.py`).

![alt text](https://github.com/KIT-ISAS/Approx_FPTD_for_Motion_Models/blob/master/experiments/for_paper/Wiener_process_Sigma_10/wiener_process_sigma_10_fptd.png)

**Example Tracks**

Some example tracks for the *CV model* with a power spectral density of 93 640 mm<sup>2</sup>s<sup>-3</sup>.

![alt text](https://github.com/KIT-ISAS/Approx_FPTD_for_Motion_Models/blob/master/experiments/for_paper/CV_Long_Track_Sw10_denorm/cv_long_track_sw10_denorm_mean_and_stddev_over_time.png)

## Additional Notes

Additional plot options can be set in `evaluators.hitting_evaluator.py`, e.g., the size of the figures or whether to omit the headers of the plot (`--for_paper`-mode). 

## Cite as

Marcel Reith-Braun, Jakob Thumm, Florian Pfaff, and Uwe D. Hanebeck, "Approximate First Passage Time Distributions for Gaussian Motion and Transportation Models," 2023.

TODO: Adjust citation and provide bibtex.

