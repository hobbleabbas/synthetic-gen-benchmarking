from pathlib import Path
from typing import List
from jinja2 import Template
import yaml

from classes import IngestionHeuristics, GeneratedProblemStatement, ProblemGeneratorParameters, FilePair

from ingest import get_all_filepairs
from gen_problem import generate_problem_statement
from clients import logger

SAMPLE_TEMPLATE = Template(
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

def parse_yaml():
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config" / "default.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def benchmark():
    config = parse_yaml()

    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )

    problem_generator_params = ProblemGeneratorParameters(
        filepair_selection_logic=highest_cosine_filepair_selector,
        prompt_template=SAMPLE_TEMPLATE,
    )

    current_dir = Path(__file__).parent
    sample_repo = current_dir.parent / "sample-repo"
    print(config)
    repos = config.keys()

    repo_to_problem_statement = {}

    for repo in repos:
        generated_problem_statements = benchmark_single_respository(
            repo_path=sample_repo,
            ingestion_heuristics=ingestion_heuristics,
            problem_generation_params=ProblemGeneratorParameters(
                **problem_generator_params, 
                num_problems_to_gen=config[repo]["problems"],
                problem_gen_model=config[repo]["agent_llm"]
            )
        )

        repo_to_problem_statement[repo] = generated_problem_statements

def benchmark_single_respository(
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

def highest_cosine_filepair_selector(file_pairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]

    return selected_file_pair

if __name__ == "__main__":
    benchmark()
    # config = parse_yaml()
    # print(config)
    # full_loop()
