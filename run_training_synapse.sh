#!/bin/sh

DATASET_PATH=DATASET_Synapse

export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse
export p_network_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export p_network_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python p_network/run/run_training.py 3d_fullres PE_trainer_synapse 2 0