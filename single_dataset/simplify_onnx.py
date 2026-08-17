import sys
import os
import argparse

def simplify_onnx_model(input_path: str, output_path: str):
    """
    Simplifies an ONNX model graph by performing constant folding and dead code elimination
    using `onnxsim` (ONNX Simplifier).
    
    Why is this needed?
    - When PyTorch/Keras/TensorFlow models are exported to ONNX, they often contain dynamic 
      shape calculation subgraphs (nodes like Shape, Gather, Split, ScatterElements, Unsqueeze).
      These nodes dynamically calculate dimensions during runtime.
    - MATLAB's deep learning network importer (`importNetworkFromONNX` or Deep Network Designer)
      has strict constraints and often fails to parse these dynamic shape subgraphs, yielding
      indexing errors (0-based vs 1-based indexing crashes) or unsupported operator errors.
    - By running ONNX Simplifier (`onnxsim`), we perform constant folding. This means the shapes 
      are calculated ahead of time and replaced with static constant integer values inside the 
      ONNX graph, eliminating the dynamic shape nodes entirely.
    - The resulting simplified graph only contains standard layers (Conv2D, Add, Relu, etc.), 
      making it 100% compatible with MATLAB.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    # Automatically install onnxsim if it is not already installed in the environment
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("[ONNX Sim] Installing onnxsim package...")
        import subprocess
        # Run pip install onnxsim using the current active Python executable
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxsim"])
        import onnx
        from onnxsim import simplify

    print(f"Loading ONNX model for simplification: {input_path}")
    model = onnx.load(input_path)

    print("Simplifying ONNX graph (constant folding shape nodes)...")
    # simplify() returns the simplified model and a validation status boolean
    model_simp, check = simplify(model)

    if not check:
        print("Warning: Simplified ONNX model validation check failed!")
    else:
        print("Validation check passed successfully!")

    # Ensure output parent directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the simplified ONNX model
    onnx.save(model_simp, output_path)
    print(f"Successfully saved simplified ONNX model to:\n  {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify ONNX model graph for MATLAB import using ONNX Simplifier.")
    parser.add_argument("--input", type=str, help="Path to input .onnx file")
    parser.add_argument("--output", type=str, help="Path to save output simplified .onnx file")
    args = parser.parse_args()

    # Default paths (useful for running straight from the IDE run button)
    default_input = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\trained_models\SNR_10dB_li_ssim_decay_s0_95_e0_05\final_net.onnx"
    default_output = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\trained_models\SNR_10dB_li_ssim_decay_s0_95_e0_05\final_net_sim.onnx"

    input_file = args.input or default_input
    output_file = args.output or (input_file.replace(".onnx", "_sim.onnx") if args.input else default_output)

    simplify_onnx_model(input_file, output_file)
