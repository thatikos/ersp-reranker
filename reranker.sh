#!/bin/bash
#SBATCH -c 4  
#SBATCH --mem=375G
#SBATCH -p gypsum-2080ti 
#SBATCH --gres=gpu:8
#SBATCH --nodes=1  
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.error


python reranker/training.py \