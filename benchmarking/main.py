import itertools
import shutil
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from typing import List, Dict, Optional, DefaultDict, Union, Final

from dotenv import load_dotenv
from git import Repo
from jinja2 import Template

from validator.generate_problem import generate_problem_statements
from miner.generate_solution import generate_code_patch
from validator.grade_output import grade_miner_solution
from helpers.classes import IngestionHeuristics, GeneratedProblemStatement, ProblemGeneratorParameters, \
    FullyScoredProblem, ValidatorModelStats, IssueSolution, UnsolvedIssue
from helpers.classes import MinerOutputScore
from helpers.clients import logger
from helpers.helpers import parse_yaml, highest_cosine_filepair_selector, flatten_and_display_solutions, \
    SENTINEL_STRING_FAILURE_VALUE, SENTINEL_INT_FAILURE_VALUE, SENTINEL_FLOAT_FAILURE_VALUE, repeat_list
from validator.ingest import get_all_filepairs

load_dotenv()

PROBLEM_STATEMENT_TEMPLATE: Final[Template] = Template(
    dedent("""
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
    """)
)

GRADER_SYSTEM_PROMPT: Final[str] = """
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


def create_problem_statements(
    config: Dict,
    repo: str,
    local_repo_dir: Path,
    problems: Union[int, List[str]],
    ingestion_heuristics: IngestionHeuristics
) -> List[GeneratedProblemStatement]:
    if isinstance(problems, int):
        problem_generator_params = ProblemGeneratorParameters(
            filepair_selection_logic=highest_cosine_filepair_selector,
            prompt_template=PROBLEM_STATEMENT_TEMPLATE,
            num_problems_to_gen=config[repo]["problems"],
            problem_gen_model=config[repo]["validator_llm"]
        )

        problem_statements: List[GeneratedProblemStatement] = generate_problems_for_single_repo(
            repo_path=local_repo_dir,
            ingestion_heuristics=ingestion_heuristics,
            problem_generation_params=problem_generator_params
        )

    elif isinstance(problems, list) and all(isinstance(text, str) for text in problems):
        problem_statements: List[GeneratedProblemStatement] = [
            GeneratedProblemStatement(
                prompt=SENTINEL_STRING_FAILURE_VALUE,
                model=SENTINEL_STRING_FAILURE_VALUE,
                problem_statement=text,
                dynamic_checklist=[],
                model_stats=ValidatorModelStats(
                    input_tokens=SENTINEL_INT_FAILURE_VALUE,
                    output_tokens=SENTINEL_INT_FAILURE_VALUE,
                    cost=SENTINEL_FLOAT_FAILURE_VALUE,
                )
            ) for text in problems
        ]

        if "repeat" in config[repo] and config[repo]["repeat"] is not None:
            num_repeats = int(config[repo]["repeat"])
            problem_statements = repeat_list(problem_statements, num_repeats)

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
        repos_dir = base_path / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)

        clone_to_path = repos_dir / repo_name
        if clone_to_path.exists() and clone_to_path.is_dir():
            shutil.rmtree(clone_to_path)
            logger.info(f"Directory {clone_to_path} has been removed.")

        Repo.clone_from(f"https://github.com/{author_name}/{repo_name}.git", clone_to_path)
        logger.info(f"Repository cloned to {clone_to_path}")
        return clone_to_path
    except Exception:
        logger.exception(f"Failed to clone repository")
        raise


def main(config: Dict) -> None:
    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )

    current_dir = Path.cwd()
    repos = config.keys()

    solutions: DefaultDict[str, List[FullyScoredProblem]] = defaultdict(list)

    for repo in repos:
        author_name, repo_name = repo.split("/")

        logger.info(f"Cloning repo {repo}...")
        local_repo_dir = clone_repo(author_name, repo_name, current_dir.parent)
        logger.info(f"Finished cloning repo {repo}")

        problems = config[repo]["problems"]
        problem_statements: List[GeneratedProblemStatement] = create_problem_statements(
            config, repo, local_repo_dir, problems, ingestion_heuristics
        )

        logger.info(f"Created problem statements: \n {pformat(problem_statements)}", )

        for problem, llm in itertools.product(problem_statements, config[repo]["agent_llm"]):
            logger.info(f"Generating code patch with LLM {llm} and problem '{problem.problem_statement[:20]}'...")

            # Run miner to generate a solution, then score the solution and create a FullyScoredProblem object, with the problem statement, solution diff, and generated grade
            start_time = time.time()

            solution: Optional[IssueSolution] = None
            try:
                solution = generate_code_patch(
                    model_name=llm,
                    unsolved_issue=UnsolvedIssue(
                        desc=problem.problem_statement,
                        local_code_path=local_repo_dir
                    )
                )
            except Exception:
                logger.exception(f"Encountered error, skipping SWE-agent run. Problem: {pformat(problem)}, llm: {llm}")
            finally:
                time_to_solve_s = time.time() - start_time


            miner_output_score: Optional[MinerOutputScore] = None
            try:
                if solution is not None:
                    miner_output_score = grade_miner_solution(
                        generated_problem_statement=problem,
                        miner_solution=solution,
                    )
            except Exception:
                logger.exception(f"Scoring solution failed")

            solutions[repo].append(FullyScoredProblem(
                generated_problem_statement=problem,
                miner_solution=solution,
                miner_llm=llm,
                miner_output_score=miner_output_score,
                time_to_solve_s=time_to_solve_s,
            ))

            logger.info(f"Obtained solution from model {llm}. Table so far...")
            flatten_and_display_solutions(solutions, should_save_data=False)
            logger.info(f"Finished displaying solutions, current model {llm}")

        logger.info("Obtained solutions. Displaying them in a table...")
        flatten_and_display_solutions(solutions)
        logger.info("Finished displaying solutions in table")


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
    current_dir = Path.cwd()
    config_path = current_dir.parent / "config" / "default.yaml"
    config = parse_yaml(config_path)

    main(config)