import sys
import os
import argparse
import onnx
from onnx import TensorProto, helper

def convert_onnx_nhwc_to_nchw(input_onnx_path, output_onnx_path):
    """
    Converts an ONNX model from NHWC (channels-last) to NCHW (channels-first) input format.
    
    Why is this needed?
    - Python/TensorFlow models are typically trained using the NHWC format [Batch, Height, Width, Channels].
    - MATLAB's deep learning network importer (e.g., importNetworkFromONNX) expects NCHW format
      [Batch, Channels, Height, Width] for 2D image/grid-like inputs.
    - If the input format is NHWC, MATLAB will import the network but will fail to load or run
      correctly because it assumes channels-first input.
      
    This function modifies the ONNX model graph structure to make it NCHW-compatible:
    1. It defines a new NCHW input tensor of shape [1, 2, 132, 14] (where C=2 represents real/imag).
    2. It either:
       - Detects and removes the redundant tf2onnx-generated Transpose node (converting it from NCHW to NHWC).
       - Or prepends a new Transpose node that maps the new NCHW input back to NHWC so the downstream
         layers (which expect NHWC) continue to work correctly without retraining.
    """
    print(f"Loading ONNX model: {input_onnx_path}")
    model = onnx.load(input_onnx_path)
    graph = model.graph

    # Get the original input node and name
    orig_input = graph.input[0]
    orig_input_name = orig_input.name
    new_input_name = "input_channel_nchw"

    # Try to detect if tf2onnx already inserted an NHWC -> NCHW transpose node at the input.
    # When exporting TensorFlow models to ONNX, tf2onnx often inserts a Transpose node at the entry.
    first_transpose_node = None
    for node in graph.node:
        if node.op_type == "Transpose" and orig_input_name in node.input:
            # Check if this node permutes NHWC (0, 1, 2, 3) to NCHW (0, 3, 1, 2)
            perm_attr = [attr for attr in node.attribute if attr.name == "perm"]
            if perm_attr and list(perm_attr[0].ints) == [0, 3, 1, 2]:
                first_transpose_node = node
                break

    if first_transpose_node is not None:
        # Optimization: if tf2onnx already had a transpose node at the start, we can optimize it out!
        # By removing this node and connecting the NCHW input directly, we simplify the graph.
        print("Found initial tf2onnx NHWC -> NCHW Transpose node. Optimizing it out completely...")
        transpose_output_name = first_transpose_node.output[0]
        
        # Point all downstream nodes that read from the Transpose output to read directly from the new NCHW input
        for node in graph.node:
            for idx, inp_name in enumerate(node.input):
                if inp_name == transpose_output_name:
                    node.input[idx] = new_input_name
                    
        # Delete the redundant transpose node from the graph
        graph.node.remove(first_transpose_node)
    else:
        # Prepend: If there is no transpose node, we insert a new Transpose node to convert
        # the user's NCHW input [1, 2, 132, 14] to the network's internal NHWC format [1, 132, 14, 2].
        print("No initial Transpose node found. Prepending NCHW -> NHWC Transpose node...")
        internal_input_name = orig_input_name + "_transposed_internal"

        # Create a Transpose node: NCHW -> NHWC (axis permutation: 0, 2, 3, 1)
        transpose_node = helper.make_node(
            "Transpose",
            inputs=[new_input_name],
            outputs=[internal_input_name],
            perm=[0, 2, 3, 1],  # (N, C, H, W) -> (N, H, W, C)
            name="nchw_to_nhwc_transpose"
        )

        # Redirect downstream nodes to read from the output of the new transpose node
        for node in graph.node:
            for idx, inp_name in enumerate(node.input):
                if inp_name == orig_input_name:
                    node.input[idx] = internal_input_name

        # Insert the transpose node at the very beginning of the node list (index 0)
        graph.node.insert(0, transpose_node)

    # Define the new NCHW input tensor value info: [1, 2, 132, 14] (FLOAT type)
    nchw_input = helper.make_tensor_value_info(
        new_input_name, TensorProto.FLOAT, [1, 2, 132, 14]
    )

    # Remove the old NHWC input and extend the graph with the new NCHW input
    graph.input.remove(orig_input)
    graph.input.extend([nchw_input])

    # Run the ONNX verification checker to ensure the modified model is valid
    onnx.checker.check_model(model)
    onnx.save(model, output_onnx_path)
    print(f"Successfully saved NCHW model to: {output_onnx_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NHWC ONNX model to NCHW input format for MATLAB.")
    parser.add_argument("--input", type=str, help="Path to input NHWC .onnx file")
    parser.add_argument("--output", type=str, help="Path to save output NCHW .onnx file")
    args = parser.parse_args()

    # Default fallback files (useful for direct execution in the IDE)
    input_file = args.input or r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\single_source_trained_model\Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_-5\results\best_net.onnx"
    output_file = args.output or r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\single_source_trained_model\Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_-5\results\best_net_nchw.onnx"

    if not os.path.exists(input_file):
        # Fallback to local workspace model path
        alt_path = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_-5\results\best_net.onnx"
        if os.path.exists(alt_path):
            input_file = alt_path
            output_file = os.path.join(os.path.dirname(alt_path), "best_net_nchw.onnx")
        else:
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)

    convert_onnx_nhwc_to_nchw(input_file, output_file)
