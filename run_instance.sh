#!/bin/bash

# Function to run a single instance
run_instance() {
    local gpu=$1
    local code=$2
    local nlvl=$3
    local ver="FK_${code}"

    export CUDA_VISIBLE_DEVICES=$gpu
    export ODIL_BACKEND=tf
    export ODIL_JIT=1

    WM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_wm.nii.gz"
    GM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_gm.nii.gz"
    CSF_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_csf.nii.gz"
    SEGM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/segm.nii.gz"
    PET_FILE_PATH="/home/michal/Combined_full/data/data_${code}/FET.nii.gz"


    OUT_FILE_PATH=$ver

    python PhysRegTumor.py --Nx 72 --Ny 72 --Nz 72 --Nt 128 \
    --nlvl $nlvl --save_last_timestep_solution \
    --wmfile $WM_FILE_PATH --gmfile $GM_FILE_PATH --csffile $CSF_FILE_PATH --segmfile $SEGM_FILE_PATH \
    --output_dir $OUT_FILE_PATH --petfile $PET_FILE_PATH
}

# Run N instances in parallel
run_instance 0 001 3 &

