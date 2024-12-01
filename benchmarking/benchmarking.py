from pathlib import Path
from typing import List
from jinja2 import Template

from classes import IngestionHeuristics, GeneratedProblemStatement, ProblemGeneratorParameters, FilePair

from ingest import get_all_filepairs
from gen_problem import generate_problem_statement

SAMPLE_TEMPLATE = Template(
    """
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files.
    
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

def highest_cosine_filepair_selector(filepairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]

    return selected_file_pair

if __name__ == "__main__":
    ## Ingest the repo and generate/retreive filepairs ranked by cosine similarity
    current_dir = Path(__file__).parent
    sample_repo = current_dir.parent / "sample-repo"
    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )
    file_pairs = get_all_filepairs(
        sample_repo,
        heuristics=ingestion_heuristics
        refresh=False
    )
    
    # Set up parameters for how to generate the problem statement
    problem_statement_generator_params = ProblemGeneratorParameters(
        filepair_selection_logic=highest_cosine_filepair_selector,
        prompt_template=SAMPLE_TEMPLATE,
    )

    # Generate one problem statement, with prompt and model to benchmark
    problem_statement: GeneratedProblemStatement = generate_problem_statement(
        filepairs=file_pairs,
        parameters=problem_statement_generator_params
    )

    print(problem_statement)