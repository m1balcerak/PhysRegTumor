#!/bin/bash

# Function to run a single instance
run_instance() {
    local gpu=$1
    local code=$2
    local nlvl=$3
    local ver="FK_${nlvl}_${code}_noPET_fixBug?"

    export CUDA_VISIBLE_DEVICES=$gpu
    export ODIL_BACKEND=tf
    export ODIL_JIT=1

    WM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_wm.nii.gz"
    GM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_gm.nii.gz"
    CSF_FILE_PATH="/home/michal/Combined_full/data/data_${code}/t1_csf.nii.gz"
    SEGM_FILE_PATH="/home/michal/Combined_full/data/data_${code}/segm.nii.gz"
    PET_FILE_PATH="/home/michal/Combined_full/data/data_${code}/FET.nii.gz"


    OUT_FILE_PATH=$ver

    python PhysRegTumor.py --Nx 64 --Ny 64 --Nz 64 --Nt 128 \
    --nlvl $nlvl --save_full_solution \
    --wmfile $WM_FILE_PATH --gmfile $GM_FILE_PATH --csffile $CSF_FILE_PATH --segmfile $SEGM_FILE_PATH \
    --output_dir $OUT_FILE_PATH --Initial --petfile $PET_FILE_PATH
}

# Run N instances in parallel

#run_instance 0 701 1 &
#run_instance 1 701 2 &
run_instance 2 701 3 &

