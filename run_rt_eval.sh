#!/bin/sh
DATASET_PATH=DATASET_Synapse
export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw
python3 unetr_pp/rt_inference/running_scripts_rt.py unetr_pp/rt_inference/model.trt unetr_pp/rt_inference/ -p unetr_pp/rt_inference/unetr_pp_Plansv2.1_plans_3D.pkl
