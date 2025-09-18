"""Utility functions for ASP+LLM pipeline for ARC tasks."""

import yaml
import clingo
import os
import json

with open("primitive_search_2.lp", "r") as f:
    ASP_PROGRAM = f.read()


def run_clingo(asp_program=ASP_PROGRAM, facts=""):
    """
    Runs Clingo on the provided ASP program string and returns the shown symbols.
    """
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], asp_program)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    models = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            # Get only shown symbols, sorted for stable readable output
            syms = sorted(model.symbols(shown=True), key=str)
            models.append(syms)
    return models

def return_models(models, do_print=False):
    output = []
    for i, syms in enumerate(models, start=1):
        input_atoms = [str(s) for s in syms if "input" in str(s)]
        output_atoms = [str(s) for s in syms if "output" in str(s)]
        if do_print:
            print("Input grid:")
            print(" " + " ".join(input_atoms))
            print("\nOutput grid:")
            print(" " + " ".join(output_atoms))
            print("\n" + "="*40 + "\n")

        output.append("Input grid:")
        output.append(" " + " ".join(input_atoms))
        output.append("\nOutput grid:")
        output.append(" " + " ".join(output_atoms))
        output.append("\n" + "="*40 + "\n")
    return "\n".join(output)

def load_prompts(filepath):
    with open(filepath, 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts

def read(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

def wholesale_solve(instance): 
    "Solves primitives on all training examples except test input."
    all_results = []
    for filename in os.listdir(instance):
        if filename != 'grid_test.lp':
            file_path = os.path.join(instance, filename)
            facts = read(file_path)
            models = run_clingo(ASP_PROGRAM, facts=facts)
            result = return_models(models)
            all_results.append(f"Results for {filename}:\n{result}")

    return "\n".join(all_results)

def test_primitives(filename: str) -> str:
    "Executes primitive search on test input"
    facts = read(filename)
    filtered_facts = "\n".join([line for line in facts.split("\n") if "output" not in line])
    models = run_clingo(ASP_PROGRAM, facts=filtered_facts)
    result = return_models(models, do_print=False)
    return result


def iterative_solve(instance):
    "Solves primitives on all training examples except test input. Then iteratively predicts rules and test output."
    all_results = []
    for filename in os.listdir(instance):
        if filename != 'grid_test.lp':
            file_path = os.path.join(instance, filename)
            facts = read(file_path)
            models = run_clingo(ASP_PROGRAM, facts=facts)
            result = return_models(models)
            all_results.append(f"Results for {filename}:\n{result}")

    return all_results


def is_executable_asp(asp_code: str) -> bool:
    """
    Checks if the given ASP code is valid and executable by Clingo.
    Returns True if valid, False otherwise.
    """
    try:
        models = run_clingo(asp_program=asp_code)
        # If Clingo runs without exception, consider it valid
        return True
    except Exception as e:
        return str(e)
    

def load_json_files(training_dir):
    training_data = []

    def numeric_prefix(filename):
        return int(filename.split('_')[0])

    filenames = sorted(
        [f for f in os.listdir(training_dir) if f.endswith(".json")],
        key=numeric_prefix
    )

    for idx, filename in enumerate(filenames):
        filepath = os.path.join(training_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            data['index'] = idx + 1
            training_data.append(data)

    return training_data


def create_asp_files(ARC_data, output_dir="instances"):
    """
    Loads ARC data and creates ASP files for each instance. 
    Each instance will have its own directory with training and test grid files.
    Grids are converted to ASP facts.
    """
    # Create the main instances directory
    instances_dir = output_dir
    if not os.path.exists(instances_dir):
        os.makedirs(instances_dir)

    # Iterate through each instance in ARC_data
    for instance_idx, instance_data in enumerate(ARC_data):
        # Create directory for this instance (1-indexed)
        instance_dir = os.path.join(instances_dir, str(instance_idx + 1))
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        
        # Process training pairs
        for grid_idx, train_pair in enumerate(instance_data['train']):
            grid_file_path = os.path.join(instance_dir, f"grid_{grid_idx}.lp")
            
            with open(grid_file_path, 'w') as f:
                # Write input grid facts
                input_grid = train_pair['input']
                input_height = len(input_grid)
                input_width = len(input_grid[0]) if input_height > 0 else 0
                
                f.write(f"grid({grid_idx},train,input,({input_height},{input_width})).\n")
                for y in range(input_height):
                    for x in range(input_width):
                        color = input_grid[y][x]
                        f.write(f"cell({grid_idx},train,input,(({y},{x}),{color})).\n")
                
                # Write output grid facts
                output_grid = train_pair['output']
                output_height = len(output_grid)
                output_width = len(output_grid[0]) if output_height > 0 else 0
                
                f.write(f"grid({grid_idx},train,output,({output_height},{output_width})).\n")
                for y in range(output_height):
                    for x in range(output_width):
                        color = output_grid[y][x]
                        f.write(f"cell({grid_idx},train,output,(({y},{x}),{color})).\n")
        
        # Process test pairs
        test_file_path = os.path.join(instance_dir, "grid_test.lp")
        with open(test_file_path, 'w') as f:
            for test_idx, test_pair in enumerate(instance_data['test']):
                # Write test input grid facts
                input_grid = test_pair['input']
                input_height = len(input_grid)
                input_width = len(input_grid[0]) if input_height > 0 else 0
                
                f.write(f"grid({test_idx},test,input,({input_height},{input_width})).\n")
                for y in range(input_height):
                    for x in range(input_width):
                        color = input_grid[y][x]
                        f.write(f"cell({test_idx},test,input,(({y},{x}),{color})).\n")
                
                # Write test output grid facts
                output_grid = test_pair['output']
                output_height = len(output_grid)
                output_width = len(output_grid[0]) if output_height > 0 else 0
                
                f.write(f"grid({test_idx},test,output,({output_height},{output_width})).\n")
                for y in range(output_height):
                    for x in range(output_width):
                        color = output_grid[y][x]
                        f.write(f"cell({test_idx},test,output,(({y},{x}),{color})).\n")


def build_and_eval(program="build_and_eval.lp", instance="instances/1/grid_test.lp", prediction=None):
    asp_program = read(program)
    facts = read(instance)
    if prediction:
        facts += "\n" + prediction
    models = run_clingo(asp_program, facts=facts)
    syms = [str(s) for s in models[0]]
    result = " ".join(syms)
    return result

