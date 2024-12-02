from typing import List

from model_pricing import calculate_price
from classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement, ListOfGeneratedProblems, \
    ValidatorModelStats
from clients import OPENAI_CLIENT


def generate_problem_statements(
    filepairs: List[FilePair],
    parameters: ProblemGeneratorParameters
) -> List[GeneratedProblemStatement]:
    selected_file_pair = parameters.filepair_selection_logic(filepairs)
    prompt_text = parameters.prompt_template.render(
        dict(
            files=selected_file_pair.files
        )
    )

    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model=parameters.problem_gen_model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly {parameters.num_problems_to_gen} statements, no more and no less"},
        ],
        response_format=ListOfGeneratedProblems,
    )

    parsed_response = completion.choices[0].message.parsed.generated_problem_statements
    prompt_tokens, completion_tokens = completion.usage.prompt_tokens, completion.usage.completion_tokens
    cost = calculate_price(parameters.problem_gen_model, prompt_tokens, completion_tokens)

    return [
        GeneratedProblemStatement(
            prompt=prompt_text,
            model=parameters.problem_gen_model,
            problem_statement=statement,
            model_stats=ValidatorModelStats(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=cost,
            )
        ) for statement in parsed_response
    ]
