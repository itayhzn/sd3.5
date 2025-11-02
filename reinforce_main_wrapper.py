import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning for SD3.5"
    )
    parser.add_argument(
        "--state_alphas",
        type=float,
        nargs="*",
        default=[0.0],
        help="The state alpha values to use during training",
    )
    parser.add_argument(
        "--action_alphas",
        type=float,
        nargs="*",
        default=[0.0],
        help="The action alpha values to use during training",
    )

    args = parser.parse_args()

    for state_alpha in args.state_alphas:
        for action_alpha in args.action_alphas:
            command = f"""conda run -n myenv python3 ./reinforce_main_wrapper.py --state_alpha {state_alpha} --action_alpha {action_alpha} --experiment_name sa_{state_alpha}_aa_{action_alpha} --num_epochs 20"""

            print("\nRunning command:", command)
            status_code = os.system(command)

            if status_code != 0:
                print(f"Command failed with status code {status_code}")
    