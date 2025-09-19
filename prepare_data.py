"""
Convert ARC JSON files to ASP format for ClingARC. 
"""
import argparse
from utils import create_asp_files, load_json_files

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert ARC JSON files to ASP format for ClingARC")
    parser.add_argument(
        "--arc-data", 
        type=str, 
        default="../ARC-AGI/data/training",
        help="Directory containing ARC JSON files (default: ../ARC-AGI/data/training)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="instances",
        help="Directory to save ASP files (default: instances)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f"Processing data in directory: {data_dir}")

    # Load ARC JSON files
    training_data = load_json_files(data_dir)

    print(f"Loaded {len(training_data)} training instances.")
    create_asp_files(training_data, output_dir=output_dir)
    print(f"ASP files created in: {output_dir}")

if __name__ == "__main__":
    main()