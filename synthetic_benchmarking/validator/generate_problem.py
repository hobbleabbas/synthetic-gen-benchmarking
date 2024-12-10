from typing import List, Dict
from pathlib import Path
import difflib
import tempfile

from synthetic_benchmarking.helpers.classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement, \
    ListOfGeneratedProblems, ValidatorModelStats
from synthetic_benchmarking.helpers.clients import OPENAI_CLIENT, logger
from synthetic_benchmarking.helpers.helpers import calculate_price
from synthetic_benchmarking.helpers.prompts import GENERATE_SYNTHETIC_TEST_PROMPT, GENERATE_FUNCTION_SPEC_PROMPT

import pytest

def create_file_patch(code: str, filename: str) -> str:
    """Creates a patch for adding a new file with the given code."""
    new_lines = code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        [],  # Empty list for original content
        new_lines,
        fromfile=f'a/{filename}',
        tofile=f'b/{filename}',
        lineterm=''
    )
    return ''.join(diff)

def compare_test_results(test_results: Dict[str, str]) -> list[int, int]:
    """Compare test results before and after patches are applied. For now we just increment, and don't record which tests failed or passed"""
    pass_before = 0
    fail_before = 0

    for _, status in test_results.items():
        if status == "passed":
            pass_before += 1
        elif status == "failed":
            fail_before += 1

    return [pass_before, fail_before]


def verify_synthetic_test(test_contents: str) -> bool:
    try:
        # Create a custom pytest.ini content to ignore import errors
        pytest_ini = """
[pytest]
addopts = --continue-on-collection-errors
"""
        # Collect tests without running them
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            
            # Write the test file
            test_file = tmp_dir / "test_synthetic.py"
            test_file.write_text(test_contents)
            
            # Write pytest.ini
            (tmp_dir / "pytest.ini").write_text(pytest_ini)
            
            # Run collection
            pytest.main(['--collect-only', str(test_file)])

        return True
    except Exception:
        logger.exception(f"Test validation failed")
        return False

def extract_python_from_llm_generation(raw_generation: str) -> str:
    if "```python" not in raw_generation:
        raise Exception(f"Improper code generation: {raw_generation}")
    
    # Extract only the content between ```python and ``` markers
    start_marker = raw_generation.find("```python") + len("```python")
    end_marker = raw_generation.rfind("```")
    extracted_code = raw_generation[start_marker:end_marker].strip()

    if extracted_code.startswith("python"):
        extracted_code = extracted_code[len("python"):]

    return extracted_code

def generate_spec_and_test_for_problem_statement(
    problem_statement: str,
    filepairs_selected: FilePair,
    dynamic_checklist: List[str] = None,
    model: str = "gpt-4",
) -> list[str, str]: 

    generate_spec_context = f"""
    Problem Statement: {problem_statement}

    Files used to generate the problem statement (consider as context for adding parameters to the spec): 
        FILE 1: {filepairs_selected.files[0].contents}
        FILE 2: {filepairs_selected.files[1].contents}

    {
        f'Checklist of things for solution to consider: {dynamic_checklist}' 
        if dynamic_checklist else 
        ''
    }
    """

    spec_response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATE_FUNCTION_SPEC_PROMPT},
            {"role": "user", "content": generate_spec_context}
        ]
    )

    generated_spec = extract_python_from_llm_generation(spec_response.choices[0].message.content)
    
    logger.info(f"Generated Solution Spec: {generated_spec}")

    # Generate a pytest with the same extraction
    generate_test_context = f"""
    Function spec to generate a test for: {generated_spec}
    Problem Statement: {problem_statement}

    Files used to generate the problem statement (consider as context): 
        FILE 1: {filepairs_selected.files[0].contents}
        FILE 2: {filepairs_selected.files[1].contents}

    {
        f'Checklist of things for solution to consider: {dynamic_checklist}' 
        if dynamic_checklist else 
        ''
    }
    """

    test_response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATE_SYNTHETIC_TEST_PROMPT},
            {"role": "user", "content": generate_test_context}
        ]
    )

    generated_pytest = extract_python_from_llm_generation(test_response.choices[0].message.content)
    
    logger.info(f"Generated Synthetic Test: {generated_pytest}")

    # Validate with pytest
    if not verify_synthetic_test(generated_pytest):
        raise ValueError("Generated test failed validation")
    
    return [
        create_file_patch(
            generated_spec, "bt_solution.py"
        ),
        create_file_patch(
            generated_pytest, "test_synthetic.py"
        )
    ]

def generate_problem_statements(
    repo_path: Path,
    filepairs: List[FilePair],
    parameters: ProblemGeneratorParameters
) -> List[GeneratedProblemStatement]:
    selected_file_pair = parameters.filepair_selection_logic(filepairs)
    prompt_text = parameters.prompt_template.render(
        dict(
            files=selected_file_pair.files
        )
    )

    model_map = {
        "gpt4omini": "gpt-4o-mini",
        "gpt4o": "gpt-4o"
    }
    model = parameters.problem_gen_model if parameters.problem_gen_model not in model_map \
        else model_map[parameters.problem_gen_model]


    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly {parameters.num_problems_to_gen} statements, no more and no less"},
        ],
        response_format=ListOfGeneratedProblems,
    )

    parsed_response = completion.choices[0].message.parsed.generated_problem_statements
    prompt_tokens, completion_tokens = completion.usage.prompt_tokens, completion.usage.completion_tokens
    cost = calculate_price(model, prompt_tokens, completion_tokens)

    # Run tests and generate a starter spec patch and a test for miners to fill out 
    # We could potentially also run the generated test on the function signature and ensure that it fails
    from synthetic_benchmarking.helpers.sweagent import create_testing_sweenv_arguments, run_tests
    from sweagent.environment.swe_env import SWEEnv

    script_arguments = create_testing_sweenv_arguments(
        model_name=parameters.problem_gen_model, 
        repo_path=repo_path
    )

    env = SWEEnv(script_arguments)
    _, _ = env.reset(0)
    
    tests_at_generation = run_tests(env)
    tests_passed_at_generation, tests_failed_at_generation = compare_test_results(tests_at_generation)

    # Generate a spec for the problem statement, and then a test for this spec.
    generated_problem_statements: List[GeneratedProblemStatement] = []

    for statement in parsed_response:
        import ipdb 
        ipdb.set_trace()

        starter_patch, generated_test_patch = generate_spec_and_test_for_problem_statement(
            problem_statement=statement.problem_statement,
            filepairs_selected=selected_file_pair,
            dynamic_checklist=statement.dynamic_checklist
        )

        generated_problem_statements.append(
            GeneratedProblemStatement(
                repo_path=repo_path,
                prompt=prompt_text,
                model=model,
                problem_statement=statement.problem_statement,
                dynamic_checklist=statement.dynamic_checklist,
                starter_patch=starter_patch,
                generated_test_patch=generated_test_patch,
                tests_passed_at_generation=tests_passed_at_generation,
                tests_failed_at_generation=tests_failed_at_generation,
                model_stats=ValidatorModelStats(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cost=cost,
                ),
            )
        )
    
    return generated_problem_statements
