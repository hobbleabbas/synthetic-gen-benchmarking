import json
from typing import List

from synthetic_benchmarking.helpers.classes import FullEvalData, FullyScoredProblem, dict_to_dataclass_or_basemodel
from synthetic_benchmarking.helpers.helpers import flatten


def main() -> None:
    sample_data: FullEvalData
    with open("full_eval_data.json", "r") as f:
        sample_data = json.load(f)

    flat_data: List[FullyScoredProblem] = flatten(flatten(
        (
            (dict_to_dataclass_or_basemodel(FullyScoredProblem, solution) for solution in solutions)
            for solutions in solutions_dict.values()
        )
        for solutions_dict in sample_data
    ))

    # pprint(flat_data)
    # for problem in flat_data:
    #     print(f"Patch is {pformat(problem)}")
    #     print(
    #         grade_miner_solution(
    #             generated_problem_statement=problem.generated_problem_statement,
    #             miner_solution=problem.miner_solution
    #         )
    #     )
    # for solutions_dict in sample_data:
    #     for repo, problem_list in solutions_dict.items():
    #         problem_list: List[FullyScoredProblem]



if __name__ == "__main__":
    main()