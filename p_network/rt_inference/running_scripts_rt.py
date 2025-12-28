import argparse
import os

from p_network.rt_inference.rt_wrapper import InferenceEngine


# to improve the efficiency set the last two true

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trt", "--tensorRT_path", help="use this to pass the tensorRT file path in", type=str, required=True)
    parser.add_argument("i", "--input_folder", help="use this to run validation on test folder", type=str, required=True)
    parser.add_argument("-p", "--plans_file_path", help="use this to run validation on test folder", type=str, default=None)
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")

    args = parser.parse_args()

    validation_only = args.validation_only
    Tsfolder = args.i
    trt_path = args.trt
    plans_file = args.plans_file_path

    # plans_file = "C:/Users/Admin/OneDrive - hcmut.edu.vn/A.I. references/ComVis/Research/Coding/framework/p_network/rt_inference/unetr_pp_Plansv2.1_plans_3D.pkl"
    
    trainer = InferenceEngine(trt_path, plans_file)

    trainer.initialize(not validation_only)

    # predict validation
    output_folder = 'inference_output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in os.listdir(Tsfolder):
        if file.endswith(".nii.gz"):
            path = os.path.join(Tsfolder, file)
            output_file = os.path.join(output_folder, file)
            print(file)
            trainer.preprocess_predict_nifti(input_files = [path], output_file=output_file, softmax_ouput_file=None, mixed_precision=False)
            return


if __name__ == "__main__":
    main()
