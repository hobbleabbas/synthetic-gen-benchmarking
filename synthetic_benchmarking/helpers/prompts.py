GENERATE_SYNTHETIC_TEST_PROMPT = """
Generate a pytest test case that:
    1. Writes a pytest test for a problem statement given to you
    2. Considers some or all of the items in the checklist of things the solution should consider
    2. Includes assertions to verify the results
    3. Uses basic pytest features
    4. Includes docstring and comments
    
DO NOT import anything other than pytest. NO IMPORTS ALLOWED BUT PYTEST, AND THE FUNCTION SPEC PROVIDED (IN bt_solution.py).

THE TEST SHOULD TEST THE FUNCTION SPEC PROVIDED, according to the problem statement.

Generate only ONE test. The test should be self-contained and not require external data. Generate ONLY the code. Generate nothing else. 
"""

GENERATE_FUNCTION_SPEC_PROMPT = """
Generate a python function spec given a problem statement and some context files. 

This function spec should consider the context, and the problem, and provide a python function to be filled out in the form of function_name(param_one: type, param_two: type) -> output_type:

Avoid using non builtin types (use only ints, floats, strings, lists, etc). 

Generate only the problem spec, and wrap it in a ```python {function_spec} ``` decorator, to make it easy to extract. Do not fill out the problem or start the completion, another model will complete this task.
"""