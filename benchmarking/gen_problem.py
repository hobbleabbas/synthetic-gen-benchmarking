from typing import List
from pydantic import BaseModel

from classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement
from clients import OPENAI_CLIENT

class ListOfGeneratedProblems(BaseModel):
    generated_problem_statements: list[str]

# How we select filepair to work on
def generate_problem_statement(
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
            {"role": "user", "content": "Generate the list of problem statements"},
        ],
        response_format=ListOfGeneratedProblems,
    )

    parsed_response = completion.choices[0].message.parsed.generated_problem_statements

    return [
        GeneratedProblemStatement(
            prompt=prompt_text,
            model=parameters.problem_gen_model,
            problem_statement=statement
        ) for statement in parsed_response
    ]
