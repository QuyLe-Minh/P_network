#!/bin/sh
DATASET_PATH=DATASET_Synapse
export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse
export p_network_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export p_network_raw_data_base="$DATASET_PATH"/unetr_pp_raw
python3 p_network/rt_inference/running_scripts_rt.py p_network/rt_inference/model.trt p_network/rt_inference/ -p p_network/rt_inference/unetr_pp_Plansv2.1_plans_3D.pkl
