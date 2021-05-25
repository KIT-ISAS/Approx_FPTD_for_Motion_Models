#!/bin/bash


python3 /mnt/wiener_process.py --save_results --result_dir /mnt/experiments/for_paper/Wiener_process_Sigma_10 --no_show > /mnt/experiments/wiener10_stddout

python3 /mnt/wiener_process.py --save_results --result_dir /mnt/experiments/for_paper/Wiener_process_Sigma_50 --no_show --sw_fifty > /mnt/experiments/wiener50_stddout

python3 /mnt/cv_experiments.py --save_results --result_dir /mnt/experiments/for_paper/ --no_show > /mnt/experiments/cv_experiments_stddout

python3 /mnt/ca_experiments.py --save_results --result_dir /mnt/experiments/for_paper/ --no_show > /mnt/experiments/ca_experiments_stddout

