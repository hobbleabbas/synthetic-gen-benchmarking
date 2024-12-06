from dataclasses import is_dataclass, dataclass, asdict
from pathlib import Path
from typing import Any
from typing import List, Callable, Optional

from jinja2 import Template
from pydantic import BaseModel


def convert_to_obj(data: Any) -> Any:
    if is_dataclass(data):
        return {k: convert_to_obj(v) for k, v in asdict(data).items()}
    elif isinstance(data, BaseModel):
        return {k: convert_to_obj(v) for k, v in data.dict().items()}
    elif isinstance(data, list):
        return [convert_to_obj(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_to_obj(v) for k, v in data.items()}
    return data


@dataclass
class IngestionHeuristics:
    min_files_to_consider_dir_for_problems: int
    min_file_content_len: int


@dataclass
class File:
    path: Path
    contents: str


@dataclass
class EmbeddedFile:
    path: str
    contents: str
    embedding: List

    def __str__(self):
        return f"File: {self.path}, Length: {len(self.contents)}"

    def __repr__(self) -> str:
        return f"File: {self.path}, Length: {len(self.contents)}"


@dataclass
class FilePair:
    cosine_similarity: float
    files: List[EmbeddedFile]


@dataclass
class ProblemGeneratorParameters:
    filepair_selection_logic: Callable[[List[FilePair]], FilePair]
    prompt_template: Template
    num_problems_to_gen: int
    problem_gen_model: str


@dataclass
class ValidatorModelStats:
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class GeneratedProblemStatement:
    prompt: str
    model: str
    problem_statement: str
    dynamic_checklist: List[str]
    model_stats: Optional[ValidatorModelStats] = None


@dataclass
class GeneratedProblemStatementList:
    problem_statements: List[GeneratedProblemStatement]
    prompt_tokens: int
    completion_tokens: int


@dataclass
class UnsolvedIssue:
    desc: str
    local_code_path: Path


class MinerModelStats(BaseModel):
    api_calls: int
    instance_cost: float
    tokens_received: int
    tokens_sent: int
    total_cost: float


@dataclass
class IssueSolution:
    patch: str
    model_stats: Optional[MinerModelStats] = None


class GeneratedProblem(BaseModel):
    problem_statement: str
    dynamic_checklist: List[str]


# We use pydantic for some classes because OpenAI json output can structure based on that
class ListOfGeneratedProblems(BaseModel):
    generated_problem_statements: List[GeneratedProblem]


class MinerOutputScore(BaseModel):
    dynamic_checklist_scores: List[float]
    addresses_problem_in_statement: float
    logical_solution: float
    brevity_and_cleanliness_of_code: float
    potential_bugs_generated: float
    explanation_of_scores: str


@dataclass
class FullyScoredProblem:
    generated_problem_statement: GeneratedProblemStatement
    miner_llm: str
    time_to_solve_s: float
    miner_solution: Optional[IssueSolution] = None
    miner_output_score: Optional[MinerOutputScore] = None
