import torch
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set file path', add_help=False)
    parser.add_argument('--file_path', default="./amed/mae_weights/checkpoint-799.pth", type=str)
    return parser


def visualize_pth_file(file_path):
    """
    Visualize the content of a .pth file.

    Args:
        file_path (str): Path to the .pth file.
    """
    # Load the .pth file
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading .pth file: {e}")
        return
    
    print("\n--- Keys in the .pth file ---")
    if isinstance(data, dict):
        for key in data.keys():
            print(f"Key: {key}")
            
            # If the key contains a model's state_dict
            if isinstance(data[key], dict):
                print("  Contains a nested dictionary (possibly a state_dict).")
                print("  Keys in this dictionary:")
                for subkey in data[key].keys():
                    print(f"    {subkey} (shape: {tuple(data[key][subkey].shape) if hasattr(data[key][subkey], 'shape') else 'scalar'})")
    elif isinstance(data, list):
        print("The file contains a list. Here are the first few elements:")
        print(data[:5])
    else:
        print("The file contains a single object:")
        print(type(data))
        print(data)

# Usage example (replace 'model.pth' with your file path)
# visualize_pth_file('model.pth')

def main(args):
    visualize_pth_file(args.file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('File path', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
