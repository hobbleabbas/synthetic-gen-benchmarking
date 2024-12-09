import json
from textwrap import dedent
from typing import List

from synthetic_benchmarking.helpers.classes import FullEvalData, FullyScoredProblem, dict_to_dataclass_or_basemodel
from synthetic_benchmarking.helpers.helpers import flatten, compute_overall_score
from synthetic_benchmarking.validator.grade_output import grade_miner_solution

PATCHES = {
    "fails_to_apply:":  dedent("""
        diff --git a/example.txt b/example.txt
        index 1234567..89abcde 100644
        --- a/example.txt
        +++ b/example.txt
        @@ -1,4 +1,6 @@
        -Hello, world!
        -This is the original content.
        +Hello, Git!
        +This line has been modified.
        +Here's some new content added.
         
        +Adding another random line for testing.
         ---End of file---
        """),
    "applies": dedent("""
    diff --git a/seaborn/algorithms.py b/seaborn/algorithms.py
index 2e34b9dd..f5f81eb2 100644
--- a/seaborn/algorithms.py
+++ b/seaborn/algorithms.py
@@ -2,6 +2,98 @@
 import numpy as np
 import warnings
 
+def bootstrap(*args, **kwargs):
+    \"\"\"Resample one or more arrays with replacement and store aggregate values.
+
+    Positional arguments are a sequence of arrays to bootstrap along the first
+    axis and pass to a summary function.
+
+    Keyword arguments:
+        n_boot : int, default=10000
+            Number of iterations
+        axis : int, default=None
+            Will pass axis to ``func`` as a keyword argument.
+        units : array, default=None
+            Array of sampling unit IDs. When used the bootstrap resamples units
+            and then observations within units instead of individual
+            datapoints.
+            data, will try to use nan-aware version of named function.
+        seed : Generator | SeedSequence | RandomState | int | None
+            Seed for the random number generator; useful if you want
+            reproducible resamples.
+
+    Returns
+    -------
+    boot_dist: array
+        array of bootstrapped statistic values
+
+    \"\"\"
+    # Ensure list of arrays are same length
+    if len(np.unique(list(map(len, args)))) > 1:
+        raise ValueError("All input arrays must have the same length")
+    n = len(args[0])
+
+    # Default keyword arguments
+    n_boot = kwargs.get("n_boot", 10000)
+    units = kwargs.get("units", None)
+    random_seed = kwargs.get("random_seed", None)
+    if random_seed is not None:
+        msg = "`random_seed` has been renamed to `seed` and will be removed"
+        warnings.warn(msg)
+    seed = kwargs.get("seed", random_seed)
+    if axis is None:
+        func_kwargs = dict()
+    else:
+        func_kwargs = dict(axis=axis)
+
+    # Initialize the resampler
+    if isinstance(seed, np.random.RandomState):
+        rng = seed
+    else:
+        rng = np.random.default_rng(seed)
+
+    # Coerce to arrays
+    args = list(map(np.asarray, args))
+    if units is not None:
+        units = np.asarray(units)
+
+    if isinstance(func, str):
+
+        # Allow named numpy functions
+        f = getattr(np, func)
+
+        # Try to use nan-aware version of function if necessary
+        missing_data = np.isnan(np.sum(np.column_stack(args)))
+
+        if missing_data and not func.startswith("nan"):
+            nanf = getattr(np, f"nan{func}", None)
+            if nanf is None:
+                msg = f"Data contain nans but no nan-aware version of `{func}` found"
+                warnings.warn(msg, UserWarning)
+            else:
+                f = nanf
+
+    else:
+        f = func
+
+    # Handle numpy changes
+    try:
+        integers = rng.integers
+    except AttributeError:
+        integers = rng.randint
+
+    # Do the bootstrap
+    if units is not None:
+        return _structured_bootstrap(args, n_boot, units, f,
+                                     func_kwargs, integers)
+
+    boot_dist = []
+    for i in range(int(n_boot)):
+        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
+        sample = [a.take(resampler, axis=0) for a in args]
+        boot_dist.append(f(*sample, **func_kwargs))
+    return np.array(boot_dist)
+
 
 def bootstrap(*args, **kwargs):
     \"\"\"Resample one or more arrays with replacement and store aggregate values.
@@ -19,7 +111,6 @@ def bootstrap(*args, **kwargs):
             and then observations within units instead of individual
             datapoints.
         func : string or callable, default="mean"
-            Function to call on the args that are passed in. If string, uses as
             name of function in the numpy namespace. If nans are present in the
             data, will try to use nan-aware version of named function.
         seed : Generator | SeedSequence | RandomState | int | None
@@ -34,8 +125,6 @@ def bootstrap(*args, **kwargs):
     \"\"\"
     # Ensure list of arrays are same length
     if len(np.unique(list(map(len, args)))) > 1:
-        raise ValueError("All input arrays must have the same length")
-    n = len(args[0])
 
     # Default keyword arguments
     n_boot = kwargs.get("n_boot", 10000)

    """),
}

def load_full_eval_data() -> List[FullyScoredProblem]:
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
    return flat_data


def main() -> None:
    flat_data = load_full_eval_data()

    for problem in flat_data:
        problem.miner_solution.patch = PATCHES["applies"]
        overall_scores = []
        miner_output_scores = []
        for _ in range(5):
            miner_output_score = grade_miner_solution(
                repo=problem.repo,
                generated_problem_statement=problem.generated_problem_statement,
                miner_solution=problem.miner_solution,
            )
            overall_score = compute_overall_score(miner_output_score)

            print(f"Overall score: {overall_score}\nExplanation: {miner_output_score.explanation_of_scores}")

            overall_scores.append(overall_score)
            miner_output_scores.append(miner_output_score)

        print(f"Overall scores: {overall_scores}")


if __name__ == "__main__":
    main()