from typing import List
from classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement
from clients import OPENAI_CLIENT

# How we select filepair to work on
def generate_problem_statement(
    filepairs: List[FilePair],
    parameters: ProblemGeneratorParameters
) -> GeneratedProblemStatement:
    selected_file_pair = parameters.filepair_selection_logic(filepairs)
    prompt_text = parameters.prompt_template.render(
        dict(
            files=selected_file_pair.files
        )
    )

    response_obj = OPENAI_CLIENT.chat.completions.create(
        model=parameters.problem_gen_model,
        messages=[
            {"role": "user", "content": prompt_text},
        ]
    )
    response = response_obj.choices[0].message.content

    return GeneratedProblemStatement(
        prompt=prompt_text,
        model=parameters.problem_gen_model,
        problem_statement=response
    )
