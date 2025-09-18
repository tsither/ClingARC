"""Main file for ClingARC project."""

from utils import wholesale_solve, iterative_solve, test_primitives, is_executable_asp, build_and_eval
from llm import LLM

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="ClingARC Project CLI")
    parser.add_argument('--mode', choices=['wholesale','iterative'], default='wholesale', help='Prompting mode to run the project pipeline')
    parser.add_argument('--print_prompts', action='store_true', help='Print prompts and exit')
    parser.add_argument('--instance', type=str, default="train_instances/1", help='Path to training instance directory')
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    llm = LLM(
        model="gpt-5-mini",
        api_key=api_key)
    
    print(f'LLM Model {llm.model}')

    ### Solve Mode ###
    # Choose between wholesale or iterative prompting modes, difference in how examples are fed to LLM for rule prediction
    # Result is found in variable 'rule_explanation' which is the predicted transformation rules in natural language for input-output examples

    if args.mode == 'wholesale':
        primitives_text = wholesale_solve(args.instance) #finds primitives in examples (except test input)

        print(f"Output from wholesale solve: {primitives_text}\n **** ")

        #LLM call to predict transformation rules in examples
        rule_explanation = llm.call("wholesale_pass", primitives=primitives_text, track_usage=False) 
        print(f'Transformation rules predicted for examples:{rule_explanation}\n')
       
    elif args.mode == 'iterative':
        ## Iteratively predict transformation rules based on each successive example ##

        #finds primitives in examples (except test input), returns list of found primitives for each input-outputexample
        primitives_list = iterative_solve(args.instance) 
        print(f"Output from iterative solve: {primitives_list}\n **** ")
        
        ### Iteratively refine transformation rules based on each example ###
        for idx, model in enumerate(primitives_list):

            print(f'--- Iteration {idx+1} of {len(primitives_list)} ---')
            n_examples = len(primitives_list)
            seen_primitives = str(primitives_list[:idx])
            previous_rule_explanation = rule_explanation if idx > 0 else "N/A"

            if idx == 0: #use first_pass prompt for first example
                rule_explanation = llm.call("first_pass", n_examples=n_examples, primitives=model, track_usage=False) 

            else: #otherwise use iterative_pass prompt to predict rules based on previous examples, rule explanations, and current primitives
                rule_explanation = llm.call("iterative_pass", 
                                            previous_primitives=seen_primitives, 
                                            previous_rule_explanation=previous_rule_explanation, 
                                            idx=idx,
                                            n_examples=n_examples,
                                            primitives=model,
                                            track_usage=False) #LLM call to predict transformation rules in examples
            print(f'Transformation rules predicted for example:{rule_explanation}\n')

    ### Generate Test Output grid ###
    # Find the test input grid
    test_instance_path = os.path.join(args.instance, "grid_test.lp") 

    #finds primitives in test input
    test_input_primitives = test_primitives(test_instance_path) 
    # print('Primitives found in test input.')

    # final LLM call to predict test output from test input primitives and predicted rules
    test_pass = llm.call("test_pass", rule_explanation=rule_explanation, test_input_primitives=test_input_primitives, track_usage=False)
    print(f'\n-- Test output predicted -- \n{test_pass}\n')


    ### Translation ###
    # Translate predicted test output into output grid facts
    output_grid_primitives = llm.call("translate_to_asp", 
                            test_input_primitives=test_input_primitives,
                            test_output_prediction=test_pass, 
                            track_usage=False) #LLM call to predict transformation rules in examples
    
    ### Check if ASP code is executable and revise, if necessary ###
    # Check if the generated ASP code is executable, if not return the Clingo error message
    outcome = is_executable_asp(output_grid_primitives)

    if outcome == True:
        print("The ASP code is executable and produces output.")
    else:
        print("Not executable:")
        revised_asp = llm.call("fix_asp_code", error_message=outcome, 
                            test_input_primitives=test_input_primitives,
                            test_output_prediction=test_pass,
                            track_usage=False)

        print(f'Revised ASP code:\n{revised_asp}')
        if is_executable_asp(revised_asp) == True:
            print("The revised ASP code is now executable and produces output.")
        else:
            print("The revised ASP code is still not executable.")
            print(f"Here are the attempted primitives:\n{revised_asp}")
        output_grid_primitives = revised_asp


    ### Build & evaluate output grid from primitives ###
    # Compare predicted output grid to actual output grid in test input file

    result = build_and_eval(output_grid_primitives, test_instance_path)

    if result == 'correct_grid':
        print("The predicted output grid matches the actual output grid. Success!")
    else:
        print("The predicted output grid does not match the actual output grid. Failure.")
        print(f"Here are the attempted primitives:\n{output_grid_primitives}")
        print(f"Here are the errors in the output:\n{result}")


if __name__ == "__main__":
    main()