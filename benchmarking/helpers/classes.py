from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional
from jinja2 import Template
from pydantic import BaseModel

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
    embedding: list

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
    dynamic_checklist: list[str]
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
    duration_s: float

@dataclass
class IssueSolution:
    patch: str
    model_stats: Optional[MinerModelStats] = None

class GeneratedProblem(BaseModel):
    problem_statement: str
    dynamic_checklist: list[str]

# We use pydantic for some classes because OpenAI json output can structure based on that
class ListOfGeneratedProblems(BaseModel):
    generated_problem_statements: list[GeneratedProblem]

class MinerLLMEvaluation(BaseModel):
    addresses_problem_in_statement: bool
    logical_solution: bool
    brevity_and_cleanliness_of_code: bool
    potential_bugs_generated: bool
    dynamic_checklist_scores: list[bool]

@dataclass
class TestResults:
    passed: int
    failed: int

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

@dataclass
class FullyScoredProblem:
    generated_problem_statement: GeneratedProblemStatement
    miner_solution: IssueSolution
    miner_llm: str
    miner_solution_score: MinerSolutionScore


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