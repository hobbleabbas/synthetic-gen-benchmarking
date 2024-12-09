from .classes import FilePair, FullyScoredProblem
from pathlib import Path
import yaml
from typing import List, Dict
from .constants import PRICING_DATA_PER_MILLION_TOKENS
from tabulate import tabulate
import textwrap
import csv

def calculate_price(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing_dict = PRICING_DATA_PER_MILLION_TOKENS[model_name]
    input_price, output_price = pricing_dict["input"], pricing_dict["output"]
    return (input_tokens * input_price + output_tokens * output_price) / 1e6

def highest_cosine_filepair_selector(file_pairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]

    return selected_file_pair

def parse_yaml():
    current_dir = Path.cwd()
    config_path = current_dir.parent / "config" / "default.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print('found config', config)
    return config


def save_to_csv(data, file_path="solutions.csv"):
    """
    Save or append the given data to a CSV file.

    :param data: A list of rows where each row is a list of values.
    :param file_path: The path to the CSV file.
    """
    headers = [
        "Repo",
        "Problem",
        "Model",
        "Solution Patch",
        "Output Score",
        "Miner $",
        "Validator $",
        "Duration (s)",
        "Miner $/min"
    ]

    # Check if file exists
    file_exists = Path(file_path).is_file()

    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers if the file does not exist
        if not file_exists:
            writer.writerow(headers)

        # Write the data rows
        writer.writerows(data)

def flatten_and_display_solutions(solutions: Dict[str, List[FullyScoredProblem]]):
    # Helper to wrap text for better display
    def wrap_text(text, width=50):
        return "\n".join(textwrap.wrap(text, width=width))

    # Flatten the solutions dictionary
    flat_data = []
    for repo, problems in solutions.items():
        for problem in problems:
            validator_cost = problem.generated_problem_statement.model_stats.cost
            miner_cost = problem.miner_solution.model_stats.total_cost
            duration_s = problem.miner_solution.model_stats.duration_s
            miner_cost_per_min = miner_cost / duration_s * 60.

            flat_data.append([
                # Repo
                wrap_text(repo, width=30),
                # Problem
                wrap_text(problem.generated_problem_statement.problem_statement[:100] + "...", width=50),
                # Model
                problem.miner_llm,
                # Solution Patch
                wrap_text(problem.miner_solution.patch[:100] + "...", width=50),
                # Output Score
                wrap_text(str(problem.miner_solution_score.total_score), width=50),
                # Synthetic Test Passed
                problem.miner_solution_score.test_results.synthetic_test_passed,
                f"{miner_cost:.2f}",
                f"{validator_cost:.2f}",
                f"{duration_s:.2f}",
                f"{miner_cost_per_min:.2f}",
            ])

    # Define headers
    headers = [
        "Repo",
        "Problem",
        "Model",
        "Solution Patch",
        "Output Score",
        "Synthetic Test Passed"
        "Miner $",
        "Validator $",
        "Duration (s)",
        "Miner $/min"
    ]

    save_to_csv(flat_data)

    print(tabulate(flat_data, headers=headers, tablefmt="fancy_grid", stralign="left"))

def create_container_with_repo(repo_path: Path):
    pass

def run_tests_for_repo(repo_path: Path, patch: str):
    pass