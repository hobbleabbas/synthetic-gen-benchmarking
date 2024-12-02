from typing import List

from classes import FilePair, ProblemGeneratorParameters, GeneratedProblemStatement
from clients import OPENAI_CLIENT



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
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly {parameters.num_problems_to_gen} statements, no more and no less"},
        ],
        response_format=ListOfGeneratedProblems,
    )

    parsed_response = completion.choices[0].message.parsed.generated_problem_statements

    print(parsed_response)

    return [
        GeneratedProblemStatement(
            prompt=prompt_text,
            model=parameters.problem_gen_model,
            problem_statement=statement
        ) for statement in parsed_response
    ]



def grade_output(
    generated_problem_statement: GeneratedProblemStatement,
    miner_solution: str
):
    SYSTEM_GRADER_PROMPT = f"""
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

    CONTEXT_FOR_SOLUTION = f"""
    Problem Statement: {generated_problem_statement.problem_statement}
    patch: {miner_solution}
    Affected Files:
    {generated_problem_statement.prompt}    
    """

    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": SYSTEM_GRADER_PROMPT},
            {"role": "user", "content": CONTEXT_FOR_SOLUTION},
        ],
        response_format=MinerOutputScore,
    )

    parsed_response = completion.choices[0].message.parsed

    if parsed_response is None:
        raise Exception("OpenAI did not grade miner output")
    
    return parsed_response