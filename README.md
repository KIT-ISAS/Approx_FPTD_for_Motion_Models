# Approx FPTD for Motion Models

<img align="right" height="250" src="https://github.com/KIT-ISAS/Approx_FPTD_for_Motion_Models/blob/master/experiments/for_paper/CV_Long_Track_Sw10_denorm/cv_long_track_sw10_denorm_fptd.png">

Code belonging to the paper

*Marcel Reith-Braun, Jakob Thumm, Florian Pfaff, und Uwe D. Hanebeck, "Approximate First Passage Time Distributions for Gaussian Motion and Transportation Models," (submitted to Fusion), 2023.*

The repo contains methods to compute approximate first passage time distributions (FPTD) for Gaussian processes with an increasing trend, such as motion and transportation models for which it is valid to assume that the increasing trend in the mean is primarily resposible for the first passage. This may include, e.g., models such as the *constant velocity (CV) model*, and the *constant acceleration (CA) model*.

The repo contains examples for three different process, the *Wiener process with drift*, for which the analytical solution is known, the *CV model* and the *CA model*. The methods are compared with Monte Carlo simulations of the corresponding discrete-time process model.

Note that the code is intended to support two-dimensional processes, where one is also interested in the distribution of *y* at the first passage time. The *CV* and *CA* models are therefore defined as 2D processes, where the process in *x* is independent of the process in *y*.

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

### `wiener_process.py`

- Methods and simulations for a first passage time problem with a *Wiener process with drift*, for which the analytical solution is known.
### `cv_process.py`

- Methods and simulations for a first passage time problem with a *CV model*.

### `ca_process.py`

- Methods and simulations for a first passage time problem with a *CA model*.

### `cv_experiments.py`

- Runs predefined experiments with *CV models*. The experiments are defined via a hard-coded config dictionary in the file.

### `ca_experimets.py`

- Runs predefined experiments with *CA models*. The experiments are defined via a hard-coded config dictionary in the file.


### Example Run Commands ###

- Get a list and explanations of all possible flags.


  ```shell script
	 python3 /mnt/cv_process.py --help
  ```
  Similar for all other executables.
  
- Run a single experiment with a *CV model* with predefined settings (see the main function of `cv_experiments.py`) and save the plots to `/mnt/results/`.

  ```shell script
	 python3 /mnt/cv_process.py --save_results --result_dir /mnt/results/
  ```
  Similar for `ca_process.py` and `wiener_process.py`.
  
- Run a single experiment with a *CV model* with predefined settings (see the main function of `cv_experiments.py`), save the plots to `/mnt/results/`, and save stdout to `/mnt/results/stdout_cv_model.txt`.

  ```shell script
	 python3 /mnt/cv_process.py --save_results --result_dir /mnt/results/ > /mnt/results/stdout_cv_model.txt
  ```
  Similar for `ca_process.py` and `wiener_process.py`.
  
- Run predefined experiments with *CV models*, save the plots to `/mnt/results/cv_experiments/`, and save stdout to `/mnt/results/cv_experiments/stdout_cv_models.txt`, but do not show the plots.

  ```shell script
	 python3 /mnt/cv_experiments.py --save_results --result_dir /mnt/results/cv_experimenents/ --no_show > /mnt/results/cv_experimenents/stdout_cv_model.txt
  ```
  
  Similar for `ca_experiments.py`. Experiment configs and chosen experiments can be adjusted in `cv_experiments.py` and `ca_experiments.py`, respectively.

- Run a single experiment with a *CV model* with predefined settings, measure the computational times (averaged over ten runs) and do not save or show the plots (measured times will be printed to stdout).

  ```shell script
	 python3 /mnt/cv_process.py --measure_computational_times --no_show
  ```
  Similar for `ca_process.py` and `wiener_process.py`.

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

Additional plot options can be set in `hitting_time_uncertainty_utils.py`, e.g., the size of the figures or whether to omit the headers of the plot (`--for_paper`-mode). 

## Cite as

Marcel Reith-Braun, Jakob Thumm, Florian Pfaff, and Uwe D. Hanebeck, "Approximate First Passage Time Distributions for Gaussian Motion and Transportation Models," 2023.

TODO: Adjust citation and provide bibtex.

