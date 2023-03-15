#!/bin/bash

python3 /mnt/ca_experiments.py --save_results --result_dir /mnt/experiments/for_paper/ --for_paper --no_show --measure_computational_times &> >(tee /mnt/experiments/ca_experiments_stddout)

python3 /mnt/cv_experiments.py --save_results --result_dir /mnt/experiments/for_paper/ --for_paper --no_show --measure_computational_times &> >(tee /mnt/experiments/cv_experiments_stddout)

python3 /mnt/wiener_process.py --save_results --result_dir /mnt/experiments/for_paper/Wiener_process_Sigma_10 --for_paper --no_show --measure_computational_times &> >(tee /mnt/experiments/wiener10_stddout)

python3 /mnt/wiener_process.py --save_results --result_dir /mnt/experiments/for_paper/Wiener_process_Sigma_50 --for_paper --no_show --sw_fifty &> >(tee /mnt/experiments/wiener50_stddout)

