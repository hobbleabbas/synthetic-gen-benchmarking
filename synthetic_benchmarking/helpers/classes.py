from dataclasses import is_dataclass, dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, TypedDict, Type, TypeVar, Union, get_origin, get_args
from typing import List, Callable, Optional

from jinja2 import Template
from pydantic import BaseModel

T = TypeVar('T')

# ================== Utils to for dataclass <-> dict parsing ===========================
def dict_to_dataclass_or_basemodel(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively converts a dictionary into a dataclass or BaseModel instance, handling nested and optional fields."""
    if is_dataclass(cls):
        init_kwargs = {}
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            field_value = data.get(field_name, None)  # Default to None if key is missing

            # Resolve Optional and Union types
            if get_origin(field_type) is Union:
                # Handle Optional (Union[X, None]) by extracting the actual type
                actual_types = get_args(field_type)
                if len(actual_types) == 2 and type(None) in actual_types:
                    actual_type = next(t for t in actual_types if t is not type(None))
                else:
                    actual_type = None
            else:
                actual_type = field_type

            # Check if the actual type is a dataclass or BaseModel
            if actual_type and isinstance(actual_type, type):
                if is_dataclass(actual_type):
                    init_kwargs[field_name] = dict_to_dataclass_or_basemodel(actual_type, field_value) if field_value else None
                elif issubclass(actual_type, BaseModel):
                    init_kwargs[field_name] = actual_type(**field_value) if field_value else None
                else:
                    init_kwargs[field_name] = field_value
            else:
                init_kwargs[field_name] = field_value

        return cls(**init_kwargs)
    elif issubclass(cls, BaseModel):
        return cls(**data)
    else:
        raise TypeError(f"{cls} is neither a dataclass nor a BaseModel.")


def convert_to_obj(data: Any) -> Any:
    if is_dataclass(data):
        return {k: convert_to_obj(v) for k, v in asdict(data).items()}
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, BaseModel):
        return {k: convert_to_obj(v) for k, v in data.dict().items()}
    elif isinstance(data, list):
        return [convert_to_obj(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_to_obj(v) for k, v in data.items()}
    return data
# ========================================================================================

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
    repo_path: Path
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


class MinerLLMEvaluation(BaseModel):
    addresses_problem_in_statement: bool
    logical_solution: bool
    brevity_and_cleanliness_of_code: bool
    potential_bugs_generated: bool
    dynamic_checklist_scores: list[bool]
    explanation_of_scores: str

@dataclass 
class MinerSolutionTestResults:
    pass_previously: int
    pass_after: int
    fail_previously: int
    fail_after: int
    synthetic_test_passed: bool

@dataclass
class MinerSolutionScore:
    total_score: float
    llm_evaluation: MinerLLMEvaluation
    test_results: MinerSolutionTestResults

EMPTY_PATCH_SCORE = MinerSolutionScore(
    total_score=0,
    llm_evaluation=MinerLLMEvaluation(
        addresses_problem_in_statement=False,
        logical_solution=False,
        brevity_and_cleanliness_of_code=False,
        potential_bugs_generated=False,
        dynamic_checklist_scores=[],
        explanation_of_scores="Patch was empty"
    ),
    test_results=MinerSolutionTestResults(
        pass_previously=0,
        pass_after=0,
        fail_previously=0,
        fail_after=0,
        synthetic_test_passed=False
    )
)

@dataclass
class FullyScoredProblem:
    repo: str  # e.g. "pytest-dev/pytest"
    generated_problem_statement: GeneratedProblemStatement
    miner_llm: str
    time_to_solve_s: float
    miner_solution: Optional[IssueSolution] = None
    miner_output_score: Optional[MinerSolutionScore] = None

# Dynamically create a TypedDict class based on the dataclass
def create_typed_dict_from_dataclass(dataclass_type: Type) -> Type[TypedDict]:
    return TypedDict(
        dataclass_type.__name__ + "Dict",
        {field.name: field.type for field in fields(dataclass_type)}
    )

# Generate FullyScoredProblemDict TypedDict
FullyScoredProblemDict = create_typed_dict_from_dataclass(FullyScoredProblem)

FullEvalData = List[Dict[str, List[FullyScoredProblemDict]]]


# SWE Agent Related Classes

from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from sweagent.agent.agents import AgentArguments
from sweagent.environment.swe_env import EnvironmentArguments
from sweagent.environment.utils import (
    get_data_path_name,
)

@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""

    # Open a PR with the patch if we can solve the issue
    open_pr: bool = False
    # When working with local repository: Apply patch
    apply_patch_locally: bool = False
    # Option to be used with open_pr: Skip action if there are already commits claiming
    # to fix the issue. Please only set this to False if you are sure the commits are
    # not fixes or if this is your own repository!
    skip_if_commits_reference_issue: bool = True
    # OBSOLETE. Do not use, will raise error. Please specify --repo_path instead.
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if self.push_gh_repo_url:
            msg = "push_gh_repo_url is obsolete. Use repo_path instead"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""

    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    # Only run instances that completely match this regex
    instance_filter: str = ".*"
    # Skip instances with existing trajectories
    skip_existing: bool = True
    # Suffix for the run name (used for example in trajectory directory naming)
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # Dump the entire config to the log
    print_config: bool = True
    # Run the agent in CTF mode (SWE-agent: EnIGMA)
    ctf: bool = False

    @property
    def run_name(self) -> str:
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        assert self.agent.config_file is not None  # mypy
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )