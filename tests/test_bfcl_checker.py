"""BFCL AST checker (port of bfcl_eval/eval_checker/ast_eval/ast_checker.py)."""

from __future__ import annotations

from typing import Any

import pytest

from ollamasearch.bench.bfcl_checker import (
    check,
    restore_names,
    standardize_string,
    tool_name,
)

TRIANGLE: dict[str, Any] = {
    "name": "calculate_triangle_area",
    "description": "Area of a triangle.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "Base."},
            "height": {"type": "integer", "description": "Height."},
            "unit": {"type": "string", "description": "Unit."},
        },
        "required": ["base", "height"],
    },
}
TRIANGLE_GT = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]

INTEREST: dict[str, Any] = {
    "name": "finance.compound_interest",
    "description": "Compound interest.",
    "parameters": {
        "type": "dict",
        "properties": {
            "principal": {"type": "float", "description": ""},
            "rate": {"type": "float", "description": ""},
            "flags": {"type": "array", "items": {"type": "string"}, "description": ""},
            "options": {"type": "dict", "description": ""},
            "people": {"type": "array", "items": {"type": "dict"}, "description": ""},
            "exact": {"type": "boolean", "description": ""},
        },
        "required": ["principal", "rate"],
    },
}


def interest_gt(**extra: list[Any]) -> list[dict[str, Any]]:
    return [{"finance.compound_interest": {"principal": [1000.0], "rate": [0.05], **extra}}]


def simple(args: dict[str, Any], gt: list[dict[str, Any]] = TRIANGLE_GT, func: Any = TRIANGLE):
    return check("simple", [func], [{func["name"]: args}], gt)


def test_correct_call_and_optional_param() -> None:
    assert simple({"base": 10, "height": 5}).valid
    assert simple({"base": 10, "height": 5, "unit": "units"}).valid
    result = simple({"base": 10, "height": 5, "unit": "cm"})
    assert (result.valid, result.error_type) == (False, "value_error:string")


def test_wrong_function_name() -> None:
    result = check("simple", [TRIANGLE], [{"area": {"base": 10, "height": 5}}], TRIANGLE_GT)
    assert result.error_type == "simple_function_checker:wrong_func_name"


def test_missing_required_and_unexpected_param() -> None:
    assert simple({"base": 10}).error_type == "simple_function_checker:missing_required"
    assert (
        simple({"base": 10, "height": 5, "color": "red"}).error_type
        == "simple_function_checker:unexpected_param"
    )


def test_missing_non_optional_answer_param() -> None:
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["cm"]}}]
    result = simple({"base": 10, "height": 5}, gt)
    assert result.error_type == "simple_function_checker:missing_optional"


def test_type_mismatch_and_bool_is_not_int() -> None:
    assert simple({"base": "10", "height": 5}).error_type == "type_error:simple"
    assert simple({"base": True, "height": 5}).error_type == "type_error:simple"
    assert simple({"base": 10.0, "height": 5}).error_type == "type_error:simple"


def test_int_accepted_for_float() -> None:
    result = simple({"principal": 1000, "rate": 0.05}, interest_gt(), INTEREST)
    assert result.valid, result
    assert simple({"principal": 999, "rate": 0.05}, interest_gt(), INTEREST).error_type == (
        "value_error:others"
    )


def test_string_normalisation() -> None:
    assert standardize_string("New York, NY") == "newyorkny"
    assert standardize_string("It's") == 'it"s'
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["square_meters"]}}]
    assert simple({"base": 10, "height": 5, "unit": "Square Meters"}, gt).valid
    assert simple({"base": 10, "height": 5, "unit": "square-meters."}, gt).valid


def test_list_values() -> None:
    base = {"principal": 1000.0, "rate": 0.05}
    gt = interest_gt(flags=[["Daily", "Compound"], ""])
    assert simple({**base, "flags": ["daily", "compound"]}, gt, INTEREST).valid
    assert simple(base, gt, INTEREST).valid  # "" = may be omitted
    wrong_order = simple({**base, "flags": ["compound", "daily"]}, gt, INTEREST)
    assert wrong_order.error_type == "value_error:list/tuple"
    # BFCL quirk kept: with "" among the answers the item type check passes trivially and
    # the value check reports the error instead.
    assert simple({**base, "flags": [1, 2]}, gt, INTEREST).error_type == "value_error:list/tuple"
    strict = interest_gt(flags=[["Daily", "Compound"]])
    nested = simple({**base, "flags": [1, 2]}, strict, INTEREST)
    assert nested.error_type == "type_error:nested"


def test_dict_values() -> None:
    gt = interest_gt(options=[{"mode": ["Monthly"], "fee": [0, ""]}])
    ok = {"principal": 1000.0, "rate": 0.05, "options": {"mode": "monthly"}}
    assert simple(ok, gt, INTEREST).valid
    extra_key = {"principal": 1000.0, "rate": 0.05, "options": {"mode": "monthly", "x": 1}}
    assert simple(extra_key, gt, INTEREST).error_type == "value_error:dict_key"
    bad_value = {"principal": 1000.0, "rate": 0.05, "options": {"mode": "weekly"}}
    assert simple(bad_value, gt, INTEREST).error_type == "value_error:dict_value"
    missing = {"principal": 1000.0, "rate": 0.05, "options": {"fee": 0}}
    assert simple(missing, gt, INTEREST).error_type == "value_error:dict_key"


def test_list_of_dicts_in_order() -> None:
    gt = interest_gt(people=[[{"name": ["Ann"]}, {"name": ["Bob"]}]])
    base = {"principal": 1000.0, "rate": 0.05}
    assert simple({**base, "people": [{"name": "ann"}, {"name": "bob"}]}, gt, INTEREST).valid
    swapped = simple({**base, "people": [{"name": "bob"}, {"name": "ann"}]}, gt, INTEREST)
    assert swapped.error_type == "value_error:dict_value"
    short = simple({**base, "people": [{"name": "ann"}]}, gt, INTEREST)
    assert short.error_type == "value_error:list_dict_count"


def test_boolean_value() -> None:
    gt = interest_gt(exact=[True])
    assert simple({"principal": 1000.0, "rate": 0.05, "exact": True}, gt, INTEREST).valid
    result = simple({"principal": 1000.0, "rate": 0.05, "exact": False}, gt, INTEREST)
    assert result.error_type == "value_error:others"


def test_args_not_an_object() -> None:
    result = check("simple", [TRIANGLE], [{"calculate_triangle_area": "base=10"}], TRIANGLE_GT)
    assert result.error_type == "ast_decoder:decoder_failed"


def test_simple_wrong_count_and_no_call() -> None:
    call = {"calculate_triangle_area": {"base": 10, "height": 5}}
    assert check("simple", [TRIANGLE], [call, call], TRIANGLE_GT).error_type == (
        "simple_function_checker:wrong_count"
    )
    assert check("simple", [TRIANGLE], [], TRIANGLE_GT).error_type == "ast_decoder:decoder_failed"


PLAY: dict[str, Any] = {
    "name": "spotify.play",
    "description": "Play.",
    "parameters": {
        "type": "dict",
        "properties": {
            "artist": {"type": "string", "description": ""},
            "duration": {"type": "integer", "description": ""},
        },
        "required": ["artist", "duration"],
    },
}
PLAY_GT = [
    {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
    {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
]


def test_parallel_is_order_independent() -> None:
    a = {"spotify.play": {"artist": "Taylor Swift", "duration": 20}}
    b = {"spotify.play": {"artist": "Maroon 5", "duration": 15}}
    assert check("parallel", [PLAY], [a, b], PLAY_GT).valid
    assert check("parallel", [PLAY], [b, a], PLAY_GT).valid
    assert check("parallel", [PLAY], [a, a], PLAY_GT).error_type == (
        "parallel_function_checker_no_order:cannot_find_match"
    )
    assert check("parallel", [PLAY], [a], PLAY_GT).error_type == (
        "parallel_function_checker_no_order:wrong_count"
    )


def test_parallel_multiple_mixed_functions() -> None:
    gt = [PLAY_GT[0], TRIANGLE_GT[0]]
    calls = [
        {"calculate_triangle_area": {"base": 10, "height": 5}},
        {"spotify.play": {"artist": "taylor swift", "duration": 20}},
    ]
    assert check("parallel_multiple", [PLAY, TRIANGLE], calls, gt).valid


def test_multiple_wrong_function() -> None:
    gt = [PLAY_GT[0]]
    right = [{"spotify.play": {"artist": "Taylor Swift", "duration": 20}}]
    wrong = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert check("multiple", [TRIANGLE, PLAY], right, gt).valid
    assert check("multiple", [TRIANGLE, PLAY], wrong, gt).error_type == (
        "simple_function_checker:wrong_func_name"
    )
    assert check("multiple", [TRIANGLE, PLAY], right * 2, gt).error_type == (
        "multiple_function_checker:wrong_count"
    )


@pytest.mark.parametrize(
    ("calls", "valid"),
    [([], True), ([{"calculate_triangle_area": {"base": 1, "height": 2}}], False)],
)
def test_irrelevance(calls: list[dict[str, Any]], valid: bool) -> None:
    result = check("irrelevance", [TRIANGLE], calls, None)
    assert result.valid is valid
    if not valid:
        assert result.error_type == "irrelevance:function_called"


def test_dotted_names_round_trip() -> None:
    assert tool_name("spotify.play") == "spotify_play"
    restored = restore_names([{"spotify_play": {}}, {"unknown_fn": {}}], [PLAY])
    assert restored == [{"spotify.play": {}}, {"unknown_fn": {}}]
