from classes import GeneratedProblemStatement, MinerOutputScore
from clients import OPENAI_CLIENT

def grade_output(
    grader_system_prompt: str,
    generated_problem_statement: GeneratedProblemStatement,
    miner_solution: str
):

    CONTEXT_FOR_SOLUTION = f"""
    Problem Statement: {generated_problem_statement.problem_statement}
    patch: {miner_solution}
    Affected Files:
    {generated_problem_statement.prompt}    
    """

    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": grader_system_prompt},
            {"role": "user", "content": CONTEXT_FOR_SOLUTION},
        ],
        response_format=MinerOutputScore,
    )

    parsed_response = completion.choices[0].message.parsed

    if parsed_response is None:
        raise Exception("OpenAI did not grade miner output")
    
    return parsed_response