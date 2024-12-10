from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

from synthetic_benchmarking.helpers.clients import logger

from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from sweagent.agent.models import ModelArguments
from sweagent.agent.agents import AgentArguments
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
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


def create_testing_sweenv_arguments(model_name: str, repo_path: Path) -> ScriptArguments:
    swe_agent_root = Path("../SWE-agent")
    return ScriptArguments(
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path=f"text://this-doesnt-matter-for-tests",
            repo_path=str(repo_path),
            verbose=True,
            install_environment=True,
            environment_setup=str(swe_agent_root / "config/environment_setup/seaborn.yaml")
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name= model_name,
            ),
            config_file=Path(swe_agent_root / "config/default_from_url.yaml"),
        ),
        actions=ActionsArguments(
            open_pr=False,
            skip_if_commits_reference_issue=False,
            apply_patch_locally=True,
        ),
        print_config=True,
    )

def run_tests(env: SWEEnv) -> Dict[str, str]:
    """
    Runs tests in the given environment and returns the results.
    Returns:
        Dict[str, str]: A dictionary with test names as keys and their status (passed, failed) as values.
    """
    try:
        env.communicate("pip install pytest-json-report")
        env.communicate("pytest --json-report --json-report-file=/tmp/report.json --json-report-omit collector", timeout_duration=300)
        pytest_report = env.communicate("cat /tmp/report.json")
        data = json.loads(pytest_report)

        tests = {}
        for test in data["tests"]:
            if test["outcome"] in ["passed", "failed"]:
                tests[test["nodeid"]] = test["outcome"].lower()

        return tests
    except Exception as error:
        logger.exception(f"Error running tests: {error}")
        return None

def apply_patch(env: SWEEnv, patch: str) -> bool:
    """
    Applies the given patch to the environment.
    Args:
        env (SWEEnv): The environment to apply the patch to.
        patch (str): The patch to apply.
    """
    try:
        env.communicate(f"echo '{patch}' > /root/patch.patch")
        env.communicate_with_handling("git apply /root/patch.patch", error_msg="Error applying patch")
        return True
    except Exception:
        logger.exception(f"Error applying patch")
        return False
