from pathlib import Path
from typing import List
from jinja2 import Template
import yaml

from miner_utils import generate_code_patch, UnsolvedIssue
from classes import IngestionHeuristics, GeneratedProblemStatement, ProblemGeneratorParameters, FilePair, FullyScoredProblem

from ingest import get_all_filepairs
from generate_problem import generate_problem_statement
from grade_output import grade_miner_solution

PROBLEM_STATEMENT_TEMPLATE = Template(
    """
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files. You will generate a list of these problems, in the generated_problems array response.
    
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
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config" / "default.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def main():
    config = parse_yaml()

    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )

    current_dir = Path(__file__).parent
    sample_repo = current_dir.parent / "seaborn"
    print(config)
    repos = config.keys()

    solutions = {}

    for repo in repos:
        problem_generator_params = ProblemGeneratorParameters(
            filepair_selection_logic=highest_cosine_filepair_selector,
            prompt_template=PROBLEM_STATEMENT_TEMPLATE,
            num_problems_to_gen=config[repo]["problems"],
            problem_gen_model=config[repo]["agent_llm"]
        )
        
        generated_problem_statements = generate_problems_for_single_repo(
            repo_path=sample_repo,
            ingestion_heuristics=ingestion_heuristics,
            problem_generation_params=problem_generator_params
        )

        solutionset_for_repo = []

        for problem in generated_problem_statements:
            # Run miner to generate a solution, then score the solution and create a FullyScoredProblem object, with the problem statement, solution diff, and generated grade
            solution = generate_code_patch(
                model_name=config[repo]["agent_llm"][0],
                unsolved_issue=UnsolvedIssue(
                    desc=problem.problem_statement,
                    local_code_path=sample_repo
                )
            )

            score_for_solution = grade_miner_solution(
                grader_system_prompt=GRADER_SYSTEM_PROMPT,
                generated_problem_statement=problem,
                miner_solution=solution
            )

            solutionset_for_repo.append(FullyScoredProblem(
                generated_problem_statement=problem,
                miner_solution_patch=solution,
                miner_output_score=score_for_solution
            ))


        solutions[repo] = solutionset_for_repo
    

def generate_problems_for_single_repo(
    repo_path: Path,
    ingestion_heuristics: IngestionHeuristics,
    problem_generation_params: ProblemGeneratorParameters
):
    file_pairs = get_all_filepairs(
        repo_path,
        heuristics=ingestion_heuristics,
        refresh=False
    )

    # Generate one problem statement, with prompt and model to benchmark
    problem_statements: List[GeneratedProblemStatement] = generate_problem_statement(
        filepairs=file_pairs,
        parameters=problem_generation_params
    )

    return problem_statements


if __name__ == "__main__":
    main()