"""
Convert ARC JSON files to ASP format for ClingARC. 
"""
from utils import create_asp_files, load_json_files

def main():

    # Define the directory containing ARC JSON files
    data_dir = "../ARC-AGI/data/training"

    # Where to save ASP files
    output_dir = "instances"

    print(f"Processing data in directory: {data_dir}")

    # Load ARC JSON files
    training_data = load_json_files(data_dir)

    print(f"Loaded {len(training_data)} training instances.")
    create_asp_files(training_data, output_dir=output_dir)
    print(f"ASP files created.")

if __name__ == "__main__":
    main()