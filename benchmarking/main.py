import csv
import itertools
import shutil
import statistics
import textwrap
from pathlib import Path
from typing import List, Dict

import yaml
from git import Repo
from jinja2 import Template
from tabulate import tabulate

from classes import IngestionHeuristics, GeneratedProblemStatement, ProblemGeneratorParameters, FilePair, \
    FullyScoredProblem, ValidatorModelStats, MinerOutputScore
from generate_problem import generate_problem_statements
from grade_output import grade_miner_solution
from ingest import get_all_filepairs
from miner_utils import generate_code_patch, UnsolvedIssue

PROBLEM_STATEMENT_TEMPLATE = Template(
    """
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files. You will generate a list of these problems, in the generated_problems array response.
    
    Further, once you have a problem statement, generate a checklist of points to consider and things that should be present in the solution (for example, are the correct Github API calls made if its a function that interfaces with the api). Generate several of these into dynamic_checklist field.
    Some additional guidelines are:
    - Do not output anything other than software engineering problem
    - The problem description should be very detailed and meticulous. It should contain sufficient context such that someone equipped with the codebase and your problem statement will have enough information to implement
    - The problem should be solvable by an autonomous SWE which can do things like installing PyPi packages, but cannot do things like make cloud provider accounts and register for other services manually.
    - The problem should not be overly difficult to implement, and should be fairly easy and not take too many LLM calls. 
    - Do not disclose which files would need to be modified to solve the problem.
    
    Here are the files:
    {% for file in files %}
    Filename: {{ file.path }}
    ```python3
    {{ file.contents }}
    ```
    {% endfor %}
    ```
    """
)

GRADER_SYSTEM_PROMPT = """
    Instructions:
    You are tasked with evaluating a code patch to determine how well it addresses a specific problem. Please follow these steps:
    Read the Problem Statement to understand the issue that needs to be resolved.
    Review the Git Diff to see the changes introduced by the patch.
    Examine the Affected Files to understand the context of the changes.
    Your Task:
    Assess the patch for correctness, completeness, and effectiveness in solving the problem.
    Fill out each field (addresses problem in statement, whether its a logical or dumb solution, brevity and how clean the code is, and how likely it is to introduce other bugs)
    Consider any potential side effects or issues introduced by the patch.
    Grade a concise solution higher than a lengthy one assuming both are correct and complete.
    Provide a percentage numerical score between 0 and 1 representing how well the patch solves the problem:
    1 means the patch perfectly and completely solves the problem.
    0 means the patch does not address the problem at all.
    If you do not know for sure that the patch perfectly and completely solved the problem, do not give it 1. Instead, give it some value between 0 and 1. Be harshly critical of the submissions you receive, think carefully to find ways in which they may have issues, and make sure the score is reduced appropriately. You will be penalized more harshly if you give scores that are too high than scores that are too low, so bias on the side of giving lower scores.
"""

# Helper to sort filepairs
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


def flatten_and_display_solutions(solutions):
    # Helper to wrap text for better display
    def wrap_text(text, width=50):
        return "\n".join(textwrap.wrap(text, width=width))

    def compute_overall_score(miner_output_score: MinerOutputScore) -> float:
        DYNAMIC_CHECKLIST_WEIGHT = 0.2
        ADDRESSES_PROBLEM_WEIGHT = 0.3
        LOGICAL_SOLUTION_WEIGHT = 0.25
        BREVITY_WEIGHT = 0.05
        POTENTIAL_BUGS_WEIGHT = 0.2
        
        return DYNAMIC_CHECKLIST_WEIGHT * statistics.mean(vars(miner_output_score.dynamic_checklist_scores).values()) + \
            ADDRESSES_PROBLEM_WEIGHT * miner_output_score.addresses_problem_in_statement + \
            LOGICAL_SOLUTION_WEIGHT * miner_output_score.logical_solution + \
            BREVITY_WEIGHT * miner_output_score.brevity_and_cleanliness_of_code + \
            POTENTIAL_BUGS_WEIGHT * miner_output_score.potential_bugs_generated

    # Flatten the solutions dictionary
    flat_data = []
    for repo, problems in solutions.items():
        for problem in problems:
            overall_score = compute_overall_score(problem.miner_output_score)

            validator_cost = problem.generated_problem_statement.model_stats.cost
            miner_cost = problem.miner_solution.model_stats.total_cost
            duration_s = problem.miner_solution.model_stats.duration_s
            miner_cost_per_min = miner_cost / duration_s * 60.

            flat_data.append([
                wrap_text(repo, width=30),
                wrap_text(problem.generated_problem_statement.problem_statement[:100] + "...", width=50),
                problem.miner_llm,
                wrap_text(problem.miner_solution.patch[:100] + "...", width=50),
                wrap_text(str(overall_score), width=50),
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
        "Miner $",
        "Validator $",
        "Duration (s)",
        "Miner $/min"
    ]

    save_to_csv(flat_data)

    print(tabulate(flat_data, headers=headers, tablefmt="fancy_grid", stralign="left"))

def create_problem_statements(config, repo, repo_path, problems, ingestion_heuristics) -> List[GeneratedProblemStatement]:
    if isinstance(problems, int):
        problem_generator_params = ProblemGeneratorParameters(
            filepair_selection_logic=highest_cosine_filepair_selector,
            prompt_template=PROBLEM_STATEMENT_TEMPLATE,
            num_problems_to_gen=config[repo]["problems"],
            problem_gen_model=config[repo]["validator_llm"]
        )

        problem_statements: List[GeneratedProblemStatement] = generate_problems_for_single_repo(
            repo_path=repo_path,
            ingestion_heuristics=ingestion_heuristics,
            problem_generation_params=problem_generator_params
        )

    elif isinstance(problems, list) and all(isinstance(text, str) for text in problems):
        problem_statements: List[GeneratedProblemStatement] = [
            GeneratedProblemStatement(
                prompt="N/A",
                model="N/A",
                problem_statement=text,
                dynamic_checklist=[],
                model_stats=ValidatorModelStats(
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                )
            ) for text in problems
        ]

        if "repeat" in config[repo] and config[repo]["repeat"] is not None:
            num_repeats = int(config[repo]["repeat"])
            temp_problem_statements = list(itertools.chain.from_iterable(
                [problem_statements[:] for _ in range(num_repeats)]
            ))
            problem_statements = temp_problem_statements

    else:
        raise ValueError(
            f"config[{repo}]['problems'] must be a list of strings or an integer. "
            f"Current value of `{config[repo]['problems']}` is invalid"
        )
    return problem_statements

def clone_repo(author_name: str, repo_name: str, base_path: Path) -> Path:
    """
    Clone a GitHub repository to a specified directory under 'repos' and return the path.

    :param author_name: GitHub username or organization name.
    :param repo_name: Repository name.
    :param base_path: Base path where the 'repos' directory will be created.
    :return: Path to the cloned repository.
    """
    try:
        repo_url = f"https://github.com/{author_name}/{repo_name}.git"

        repos_dir = base_path / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)

        clone_to_path = repos_dir / repo_name
        if clone_to_path.exists() and clone_to_path.is_dir():
            shutil.rmtree(clone_to_path)
            print(f"Directory {clone_to_path} has been removed.")

        Repo.clone_from(repo_url, clone_to_path)
        print(f"Repository cloned to {clone_to_path}")
        return clone_to_path
    except Exception as e:
        print(f"Failed to clone repository: {e}")
        raise

def main():
    config = parse_yaml()

    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )

    current_dir = Path.cwd()
    repos = config.keys()

    solutions: Dict[str, List[FullyScoredProblem]] = {}

    for repo in repos:
        author_name, repo_name = repo.split("/")
        print(f"Cloning repo {repo}...")
        local_repo_dir = clone_repo(author_name, repo_name, current_dir.parent)
        print(f"Finished cloning repo {repo}")

        problems = config[repo]["problems"]
        problem_statements: List[GeneratedProblemStatement] = create_problem_statements(
            config, repo, local_repo_dir, problems, ingestion_heuristics
        )

        print(f"Created problem statements: \n {problem_statements}",)
        solutionset_for_repo: List[FullyScoredProblem] = []
        for problem, llm in itertools.product(problem_statements, config[repo]["agent_llm"]):
            print(f"Generating code patch with LLM {llm} and problem '{problem.problem_statement[:20]}'...")

            # Run miner to generate a solution, then score the solution and create a FullyScoredProblem object, with the problem statement, solution diff, and generated grade
            try:
                solution = generate_code_patch(
                    model_name=llm,
                    unsolved_issue=UnsolvedIssue(
                        desc=problem.problem_statement,
                        local_code_path=local_repo_dir
                    )
                )

                score_for_solution = grade_miner_solution(
                    grader_system_prompt=GRADER_SYSTEM_PROMPT,
                    generated_problem_statement=problem,
                    miner_solution=solution,
                )

                solutionset_for_repo.append(FullyScoredProblem(
                    generated_problem_statement=problem,
                    miner_solution=solution,
                    miner_llm=llm,
                    miner_output_score=score_for_solution
                ))
            except Exception as e:
                print(f"Encountered error, skipping SWE-agent run. Error: {repr(e)} Problem: {problem}, llm: {llm}")


        solutions[repo] = solutionset_for_repo

        print("Obtained solutions. Displaying them in a table...")
        flatten_and_display_solutions(solutions)
        print("Finished displaying solutions in table")

def generate_problems_for_single_repo(
    repo_path: Path,
    ingestion_heuristics: IngestionHeuristics,
    problem_generation_params: ProblemGeneratorParameters
) -> List[GeneratedProblemStatement]:
    file_pairs = get_all_filepairs(
        repo_path,
        heuristics=ingestion_heuristics,
        refresh=False
    )

    # Generate one problem statement, with prompt and model to benchmark
    problem_statements_list = generate_problem_statements(
        filepairs=file_pairs,
        parameters=problem_generation_params
    )
    return problem_statements_list


if __name__ == "__main__":
    main()