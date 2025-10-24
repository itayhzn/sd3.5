import os
from datetime import datetime
import argparse
import random

def read_file(file_path):
    """
    Read a text file and return its content as a list of lines.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="The name of the experiment to run",)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=read_file("pqpp_prompts.txt"),
        help="The text prompts to generate images or videos from",)
    parser.add_argument(
        "--prompts_fraction_start",
        type=int,
        default=0,
        help="The fraction of prompts to start with (0-1)",)
    parser.add_argument(
        "--prompts_fraction_end",
        type=int,
        default=1,
        help="The fraction of prompts to end with (0-1)",)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[],
        help="The random seeds to use for generation",)
    parser.add_argument(
        "--num_random_seeds_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate for each prompt. If seeds is given, this will be ignored.",)
    
    args = parser.parse_args()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # redirect output to a file
    with open(f"jobs-out-err/{datetime_str}_{args.experiment_name}.out", "w") as f:
        os.dup2(f.fileno(), 1)
    # redirect error output to a file
    with open(f"jobs-out-err/{datetime_str}_{args.experiment_name}.err", "w") as f:
        os.dup2(f.fileno(), 2)
    
    os.system('nvidia-smi')

    args.prompts = args.prompts[
        int(len(args.prompts) * args.prompts_fraction_start) : int(
            len(args.prompts) * args.prompts_fraction_end
        )
    ]

    for prompt in args.prompts:
        if args.seeds == [] and args.num_random_seeds_per_prompt > 0:
            args.seeds = random.sample(range(0, 10000), args.num_random_seeds_per_prompt)
        for seed in args.seeds:
            command = f"""conda run -n myenv python3 ./sd3_infer.py --prompt \"{prompt}\" --seed {seed} --verbose"""

            if args.experiment_name is not None:
                command += f' --experiment_name "{args.experiment_name}"'
            
            print("\nRunning command:", command)
            status_code = os.system(command)

            if status_code != 0:
                print("\tCommand:\n\t", command, "\nfailed with status code", status_code)
                continue