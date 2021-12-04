#!/bin/bash
#SBATCH -n 1 # Number of cores, pero no funciona
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /home/mtancak/conv-mixer/Code # working directory
#SBATCH -t 0-00:05 # Runtime in D-HH:MM
#SBATCH -p high # Partition to submit to
#SBATCH --mem 19200 # MB solicitados
#SBATCH --gres gpu:1 # GPU's solicitadas gpu: o gpu:TitanXp: o gpu:1080Ti:
#SBATCH -o ouput_log_%j.out # File to which STDOUT will be written
#SBATCH -e error_log_%j.err # File to which STDERR will be written

python3 train.py
