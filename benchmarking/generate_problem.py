from typing import List

from classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement, ListOfGeneratedProblems
from clients import OPENAI_CLIENT

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
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly {parameters.num_problems_to_gen} statements, no more and no less"},
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