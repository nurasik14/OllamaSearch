"""AST checker for BFCL single-turn Python categories.

A port of the Python path of BFCL's ``bfcl_eval/eval_checker/ast_eval/ast_checker.py``
(github.com/ShishirPatil/gorilla, Apache-2.0). The logic, the order of the checks and the
error-type strings follow the original so results are comparable with the leaderboard.

Model output is a list of calls in BFCL's decoded shape: ``[{function_name: {param: value}}]``.
Ground truth (``possible_answer``) has the same shape, but each parameter maps to a list of
accepted values; ``""`` in that list means the parameter may be omitted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

#: BFCL's PYTHON_TYPE_MAPPING.
PYTHON_TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}
# Types whose items are type-checked one level deep.
PYTHON_NESTED_TYPE_CHECK_LIST = ("array", "tuple")

_STANDARDIZE = re.compile(r"[ \,\.\/\-\_\*\^]")

Call = dict[str, Any]  # {function_name: arguments}


class CheckResult(NamedTuple):
    valid: bool
    error_type: str  # "" when valid
    detail: str = ""


_OK = CheckResult(True, "")


def tool_name(name: str) -> str:
    """Function name as sent to the model: OpenAI-style tools reject ``.`` in names."""
    return name.replace(".", "_")


def restore_names(calls: Sequence[Call], functions: Sequence[Mapping[str, Any]]) -> list[Call]:
    """Map underscored tool names back to the dataset's (possibly dotted) names."""
    names = {tool_name(str(f["name"])): str(f["name"]) for f in functions}
    out: list[Call] = []
    for call in calls:
        out.append({names.get(k, k): v for k, v in call.items()})
    return out


def standardize_string(text: str) -> str:
    """BFCL's normalisation: drop spaces and ``,./-_*^``, lowercase, ``'`` -> ``"``."""
    return _STANDARDIZE.sub("", text).lower().replace("'", '"')


def _find_description(
    functions: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, Any] | None:
    for func in functions:
        if func["name"] == name:
            return func
    return None


def _possible_answer_type(possible: Sequence[Any]) -> type | None:
    for answer in possible:
        if answer != "":  # optional marker
            return type(answer)
    return None


def _type_checker(
    param: str,
    value: Any,
    possible: Sequence[Any],
    expected_description: str,
    expected_type: type,
    nested_type: type | None,
) -> tuple[CheckResult, bool]:
    """Returns (result, is_variable). Exact ``type()`` comparison, as BFCL (bool is not int)."""
    is_variable = False
    possible_type = _possible_answer_type(possible)
    if possible_type is not None and possible_type is not expected_type:
        is_variable = True

    result = _OK
    if type(value) is expected_type:
        if nested_type is None:
            return _OK, is_variable
        for possible_item in possible:
            flag = True
            if type(possible_item) is list:
                for value_item in cast("list[Any]", value):
                    res, _ = _type_checker(
                        param, value_item, possible_item, nested_type.__name__, nested_type, None
                    )
                    if not res.valid:
                        flag = False
                        break
            if flag:
                return _OK, is_variable
        result = CheckResult(
            False,
            "type_error:nested",
            f"Nested type checking failed for parameter {param!r}. Expected outer type "
            f"{expected_description} with inner type {nested_type.__name__}. "
            f"Parameter value: {value!r}.",
        )

    # Falls through like BFCL: a value of the possible answers' type counts as a "variable"
    # (and keeps a nested error if there was one).
    if possible_type is not None and type(value) is possible_type:
        return result, True

    return (
        CheckResult(
            False,
            "type_error:simple",
            f"Incorrect type for parameter {param!r}. Expected type {expected_description}, "
            f"got {type(value).__name__}. Parameter value: {value!r}.",
        ),
        is_variable,
    )


def _string_checker(param: str, value: str, possible: Sequence[Any]) -> CheckResult:
    accepted = [standardize_string(p) for p in possible if type(p) is str]
    if standardize_string(value) not in accepted:
        return CheckResult(
            False,
            "value_error:string",
            f"Invalid value for parameter {param!r}: {value!r}. Expected one of "
            f"{list(possible)}. Case insensitive.",
        )
    return _OK


def _list_checker(param: str, value: Sequence[Any], possible: Sequence[Any]) -> CheckResult:
    std_value = [standardize_string(v) if type(v) is str else v for v in value]
    # As in BFCL, the optional marker "" iterates to [] (so an empty list matches it).
    std_possible = [
        [standardize_string(x) if type(x) is str else x for x in p]
        if isinstance(p, (list, str))
        else p
        for p in possible
    ]
    if std_value not in std_possible:
        return CheckResult(
            False,
            "value_error:list/tuple",
            f"Invalid value for parameter {param!r}: {list(value)!r}. Expected one of "
            f"{list(possible)}.",
        )
    return _OK


def _dict_checker(
    param: str, value: Mapping[str, Any], possible_answers: Sequence[Any]
) -> CheckResult:
    """Flat dict comparison (BFCL does not recurse into nested dicts either)."""
    result = CheckResult(False, "dict_checker:unclear")
    for possible in possible_answers:
        if possible == "":
            continue
        result = CheckResult(False, "dict_checker:unclear")
        flag = True
        for key, item in value.items():
            if key not in possible:
                result = CheckResult(
                    False, "value_error:dict_key", f"Unexpected dict key parameter: {key!r}."
                )
                flag = False
                break
            std_item = standardize_string(item) if type(item) is str else item
            std_possible = [standardize_string(x) if type(x) is str else x for x in possible[key]]
            if std_item not in std_possible:
                result = CheckResult(
                    False,
                    "value_error:dict_value",
                    f"Invalid value for parameter {key!r}: {item!r}. "
                    f"Expected one of {std_possible}.",
                )
                flag = False
                break
        # BFCL runs the missing-key check even after a failure above (result is overwritten).
        for key, accepted in possible.items():
            if key not in value and "" not in accepted:
                result = CheckResult(
                    False, "value_error:dict_key", f"Missing dict key parameter: {key!r}."
                )
                flag = False
                break
        if flag:
            return _OK
    return result


def _list_dict_checker(
    param: str, value: Sequence[Any], possible_answers: Sequence[Any]
) -> CheckResult:
    """List of dicts; the order must match the possible answer's order."""
    result = CheckResult(False, "list_dict_checker:unclear")
    for possible in possible_answers:
        if not isinstance(possible, (list, str)) or len(value) != len(possible):
            result = CheckResult(
                False, "value_error:list_dict_count", "Wrong number of dictionaries in the list."
            )
            continue
        flag = True
        for i, item in enumerate(value):
            if not isinstance(item, Mapping):
                result = CheckResult(
                    False, "type_error:nested", f"List item {i} of {param!r} is not a dict."
                )
                flag = False
                break
            result = _dict_checker(param, item, [possible[i]])
            if not result.valid:
                flag = False
                break
        if flag:
            return _OK
    return result


def simple_function_checker(
    func: Mapping[str, Any], model_call: Call, possible_answer: Call
) -> CheckResult:
    """One call against one function doc and its ground truth ``{name: {param: [values]}}``."""
    possible: Mapping[str, Sequence[Any]] = next(iter(possible_answer.values()))
    func_name = str(func["name"])
    param_details: Mapping[str, Any] = func["parameters"].get("properties", {})
    required: Sequence[str] = func["parameters"].get("required", [])

    if func_name not in model_call:
        return CheckResult(
            False,
            "simple_function_checker:wrong_func_name",
            f"Function name {func_name!r} not found in model output.",
        )
    model_params = model_call[func_name]
    if not isinstance(model_params, Mapping):
        return CheckResult(
            False,
            "ast_decoder:decoder_failed",
            f"Arguments of {func_name!r} are not an object: {model_params!r}.",
        )

    for param in required:
        if param not in model_params:
            return CheckResult(
                False,
                "simple_function_checker:missing_required",
                f"Missing required parameter: {param!r}.",
            )

    for param, raw_value in model_params.items():
        value = raw_value
        if param not in param_details or param not in possible:
            return CheckResult(
                False,
                "simple_function_checker:unexpected_param",
                f"Unexpected parameter: {param!r}.",
            )
        details: Mapping[str, Any] = param_details[param]
        expected_description = str(details.get("type", "string"))
        expected_type = PYTHON_TYPE_MAPPING.get(expected_description, str)
        nested_type: type | None = None
        if expected_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_name = str(details.get("items", {}).get("type", "any"))
            nested_type = PYTHON_TYPE_MAPPING.get(nested_name, str)

        if expected_description == "tuple" and type(value) is tuple:
            value = list(value)
        # Python auto-converts int to float.
        if expected_description == "float" and type(value) is int:
            value = float(value)

        type_result, is_variable = _type_checker(
            param, value, possible[param], expected_description, expected_type, nested_type
        )
        if not type_result.valid:
            return type_result

        if not is_variable:
            if expected_type is dict:
                result = _dict_checker(param, value, possible[param])
            elif expected_type is list and nested_type is dict:
                result = _list_dict_checker(param, value, possible[param])
            elif expected_type is str:
                result = _string_checker(param, value, possible[param])
            elif expected_type is list:
                result = _list_checker(param, value, possible[param])
            else:
                result = None
            if result is not None:
                if not result.valid:
                    return result
                continue

        if value not in possible[param]:
            return CheckResult(
                False,
                "value_error:others",
                f"Invalid value for parameter {param!r}: {value!r}. "
                f"Expected one of {list(possible[param])}.",
            )

    for param, accepted in possible.items():
        if param not in model_params and "" not in accepted:
            return CheckResult(
                False,
                "simple_function_checker:missing_optional",
                f"Optional parameter {param!r} not provided and not marked as optional.",
            )
    return _OK


def _missing_doc(name: str) -> CheckResult:
    return CheckResult(False, "ast_checker:missing_function_doc", f"No doc for {name!r}.")


def parallel_function_checker_no_order(
    functions: Sequence[Mapping[str, Any]],
    model_output: Sequence[Call],
    possible_answers: Sequence[Call],
) -> CheckResult:
    """Every expected call must match a distinct model call, in any order."""
    if len(model_output) != len(possible_answers):
        return CheckResult(
            False,
            "parallel_function_checker_no_order:wrong_count",
            f"Wrong number of functions: expected {len(possible_answers)}, "
            f"got {len(model_output)}.",
        )
    matched: set[int] = set()
    for i, expected in enumerate(possible_answers):
        name = next(iter(expected))
        func = _find_description(functions, name)
        if func is None:
            return _missing_doc(name)
        errors: list[str] = []
        found = False
        for index, call in enumerate(model_output):
            if index in matched:
                continue
            result = simple_function_checker(func, call, expected)
            if result.valid:
                matched.add(index)
                found = True
                break
            errors.append(f"[{index}] {result.error_type}: {result.detail}")
        if not found:
            return CheckResult(
                False,
                "parallel_function_checker_no_order:cannot_find_match",
                f"No model call matches expected call {i} ({name}). " + " ".join(errors),
            )
    return _OK


def multiple_function_checker(
    functions: Sequence[Mapping[str, Any]],
    model_output: Sequence[Call],
    possible_answers: Sequence[Call],
) -> CheckResult:
    """Several functions offered, exactly one expected call."""
    if len(model_output) != len(possible_answers):
        return CheckResult(
            False,
            "multiple_function_checker:wrong_count",
            f"Wrong number of functions: expected {len(possible_answers)}, "
            f"got {len(model_output)}.",
        )
    name = next(iter(possible_answers[0]))
    func = _find_description(functions, name)
    if func is None:
        return _missing_doc(name)
    return simple_function_checker(func, model_output[0], possible_answers[0])


def ast_checker(
    functions: Sequence[Mapping[str, Any]],
    model_output: Sequence[Call],
    possible_answers: Sequence[Call],
    category: str,
) -> CheckResult:
    """Dispatch on the category name exactly as BFCL does."""
    if "parallel" in category:
        return parallel_function_checker_no_order(functions, model_output, possible_answers)
    if "multiple" in category:
        return multiple_function_checker(functions, model_output, possible_answers)
    if len(model_output) != 1:
        return CheckResult(
            False,
            "simple_function_checker:wrong_count",
            f"Wrong number of functions: expected 1, got {len(model_output)}.",
        )
    return simple_function_checker(functions[0], model_output[0], possible_answers[0])


def check(
    category: str,
    functions: Sequence[Mapping[str, Any]],
    calls: Sequence[Call],
    ground_truth: Sequence[Call] | None,
) -> CheckResult:
    """Score one item. ``calls`` use the dataset's function names (see :func:`restore_names`).

    ``irrelevance``: correct iff the model made no call. Otherwise no call at all is BFCL's
    ``ast_decoder:decoder_failed``.
    """
    if category == "irrelevance" or ground_truth is None:
        if calls:
            names = ", ".join(next(iter(c), "?") for c in calls)
            return CheckResult(
                False, "irrelevance:function_called", f"Called {names} for an unanswerable request."
            )
        return _OK
    if not calls:
        return CheckResult(False, "ast_decoder:decoder_failed", "The model made no function call.")
    return ast_checker(functions, calls, ground_truth, category)
