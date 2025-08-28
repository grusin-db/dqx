import itertools
import pprint
import logging
import datetime
import json
import re
from pathlib import Path
from unittest.mock import Mock
import yaml
import pytest
import pyspark.sql.functions as F
from pydantic import ValidationError
from pyspark.sql import Column
from databricks.labs.dqx.check_funcs import (
    is_not_null,
    is_not_null_and_not_empty,
    sql_expression,
    is_in_list,
    is_not_null_and_not_empty_array,
    is_not_null_and_is_in_list,
    is_unique,
    is_aggr_not_greater_than,
    is_aggr_not_less_than,
    foreign_key,
    is_valid_ipv4_address,
    is_ipv4_address_in_cidr,
    is_not_less_than,
    is_not_greater_than,
    is_valid_date,
    regex_match,
    compare_datasets,
)
from databricks.labs.dqx.rule import (
    DQForEachColRule,
    DQRowRule,
    DQRule,
    CHECK_FUNC_REGISTRY,
    REGISTERED_FUNCTIONS,
    REGISTERED_RULE_KLASSES,
    REGISTERED_RULE_TYPES,
    register_rule,
    register_rule_type,
    DQDatasetRule,
)
from databricks.labs.dqx.checks_serializer import (
    deserialize_checks,
    serialize_checks,
    serialize_checks_to_bytes,
)


SCHEMA = "a: int, b: int, c: int"


def test_get_for_each_rules():
    actual_rules = (
        # set of columns for the same check
        DQForEachColRule(
            columns=["a", "b"],
            check_func=is_not_null_and_not_empty,
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ).get_rules()
        # with check function params provided as positional arguments
        + DQForEachColRule(
            columns=["c", "d"], criticality="error", check_func=is_in_list, check_func_args=[[1, 2]]
        ).get_rules()
        # with check function params provided as named arguments
        + DQForEachColRule(
            columns=["e"], criticality="warn", check_func=is_in_list, check_func_kwargs={"allowed": [3]}
        ).get_rules()
        # should be skipped
        + DQForEachColRule(columns=[], criticality="error", check_func=is_not_null_and_not_empty).get_rules()
        # set of columns for the same check
        + DQForEachColRule(columns=["a", "b"], check_func=is_not_null_and_not_empty_array).get_rules()
        # set of columns for the same check with the same custom name
        + DQForEachColRule(columns=["a", "b"], check_func=is_not_null, name="custom_common_name").get_rules()
        # set of columns for check taking as input multiple columns
        + DQForEachColRule(
            columns=[["a", "b"], ["c"]], check_func=is_unique, check_func_kwargs={"nulls_distinct": False}
        ).get_rules()
    )

    expected_rules = [
        DQRowRule(
            name="a_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="a",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(
            name="b_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="b",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(
            name="c_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="c",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="d_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="d",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="e_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="e",
            check_func_kwargs={"allowed": [3]},
        ),
        DQRowRule(
            name="a_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="a",
        ),
        DQRowRule(
            name="b_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="b",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="a",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="b",
        ),
        DQDatasetRule(
            name="struct_a_b_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["a", "b"],
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQDatasetRule(
            name="c_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["c"],
            check_func_kwargs={"nulls_distinct": False},
        ),
    ]

    assert pprint.pformat(actual_rules) == pprint.pformat(expected_rules)


def test_build_rules():
    actual_rules = _build_rules_foreach_col(
        # set of columns for the same check
        DQForEachColRule(
            columns=["a", "b"],
            criticality="error",
            filter="c>0",
            check_func=is_not_null_and_not_empty,
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQForEachColRule(columns=["c"], criticality="warn", check_func=is_not_null_and_not_empty),
        # with check function params provided as positional arguments
        DQForEachColRule(columns=["d", "e"], criticality="error", check_func=is_in_list, check_func_args=[[1, 2]]),
        # with check function params provided as named arguments
        DQForEachColRule(columns=["f"], criticality="warn", check_func=is_in_list, check_func_kwargs={"allowed": [3]}),
        # should be skipped
        DQForEachColRule(columns=[], criticality="error", check_func=is_not_null_and_not_empty),
        # set of columns for the same check
        DQForEachColRule(columns=["a", "b"], criticality="error", check_func=is_not_null_and_not_empty_array),
        DQForEachColRule(columns=["c"], criticality="warn", check_func=is_not_null_and_not_empty_array),
        # set of columns for the same check with the same custom name
        DQForEachColRule(columns=["a", "b"], check_func=is_not_null, name="custom_common_name"),
        # set of columns for check taking as input multiple columns
        DQForEachColRule(
            columns=[["a", "b"], ["c"]], check_func=is_unique, check_func_kwargs={"nulls_distinct": False}
        ),
        DQForEachColRule(
            name="is_unique_with_filter",
            columns=[["a", "b"], ["c"]],
            filter="a > b",
            check_func=is_unique,
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQForEachColRule(
            name="count_aggr_greater_than",
            columns=["a", "*"],
            filter="a > b",
            check_func=is_aggr_not_greater_than,
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQForEachColRule(
            name="count_aggr_less_than",
            columns=["a", "*"],
            filter="a > b",
            check_func=is_aggr_not_less_than,
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQForEachColRule(
            name="foreign_key",
            criticality="warn",
            columns=[["a"], ["c"]],
            filter="a > b",
            check_func=foreign_key,
            check_func_kwargs={"ref_columns": ["ref_a"], "ref_df_name": "ref_df_key"},
        ),
    ) + [
        DQRowRule(
            name="g_is_null_or_empty",
            criticality="warn",
            filter="a=0",
            check_func=is_not_null_and_not_empty,
            column="g",
        ),
        DQRowRule(criticality="warn", check_func=is_in_list, column="h", check_func_args=[[1, 2]]),
        DQRowRule(
            criticality="warn",
            check_func=is_in_list,
            check_func_args=[[1, 2]],
            check_func_kwargs={"column": "h_as_kwargs"},
        ),
        DQRowRule(
            criticality="warn",
            check_func=is_in_list,
            check_func_args=[[1, 2]],
            # column field should be instead of column kwargs
            column="i",
            check_func_kwargs={"column": "i_as_kwargs"},
        ),
        DQDatasetRule(criticality="warn", check_func=is_unique, columns=["g"]),
        DQDatasetRule(criticality="warn", check_func=is_unique, check_func_kwargs={"columns": ["g_as_kwargs"]}),
        # columns field should be used instead of columns kwargs
        DQDatasetRule(
            criticality="warn", check_func=is_unique, columns=["j"], check_func_kwargs={"columns": ["j_as_kwargs"]}
        ),
        DQRowRule(criticality="warn", check_func=is_valid_ipv4_address, column="g"),
        DQRowRule(
            criticality="warn", check_func=is_ipv4_address_in_cidr, column="g", check_func_args=["192.168.1.0/24"]
        ),
    ]

    expected_rules = [
        DQRowRule(
            name="a_is_null_or_empty",
            criticality="error",
            filter="c>0",
            check_func=is_not_null_and_not_empty,
            column="a",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(
            name="b_is_null_or_empty",
            criticality="error",
            filter="c>0",
            check_func=is_not_null_and_not_empty,
            column="b",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(name="c_is_null_or_empty", criticality="warn", check_func=is_not_null_and_not_empty, column="c"),
        DQRowRule(
            name="d_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="d",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="e_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="e",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="f_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="f",
            check_func_kwargs={"allowed": [3]},
        ),
        DQRowRule(
            name="a_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="a",
        ),
        DQRowRule(
            name="b_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="b",
        ),
        DQRowRule(
            name="c_is_null_or_empty_array",
            criticality="warn",
            check_func=is_not_null_and_not_empty_array,
            column="c",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="a",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="b",
        ),
        DQDatasetRule(
            name="struct_a_b_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["a", "b"],
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQDatasetRule(
            name="c_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["c"],
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQDatasetRule(
            name="is_unique_with_filter",
            criticality="error",
            check_func=is_unique,
            columns=["a", "b"],
            filter="a > b",
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQDatasetRule(
            name="is_unique_with_filter",
            criticality="error",
            check_func=is_unique,
            columns=["c"],
            filter="a > b",
            check_func_kwargs={"nulls_distinct": False},
        ),
        DQDatasetRule(
            name="count_aggr_greater_than",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="a",
            filter="a > b",
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQDatasetRule(
            name="count_aggr_greater_than",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="*",
            filter="a > b",
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQDatasetRule(
            name="count_aggr_less_than",
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="a",
            filter="a > b",
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQDatasetRule(
            name="count_aggr_less_than",
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="*",
            filter="a > b",
            check_func_kwargs={"limit": 1, "group_by": ["c"], "aggr_type": "count"},
        ),
        DQDatasetRule(
            name="foreign_key",
            criticality="warn",
            check_func=foreign_key,
            columns=["a"],
            filter="a > b",
            check_func_kwargs={
                "ref_columns": ["ref_a"],
                "ref_df_name": "ref_df_key",
            },
        ),
        DQDatasetRule(
            name="foreign_key",
            criticality="warn",
            check_func=foreign_key,
            columns=["c"],
            filter="a > b",
            check_func_kwargs={
                "ref_columns": ["ref_a"],
                "ref_df_name": "ref_df_key",
            },
        ),
        DQRowRule(
            name="g_is_null_or_empty",
            criticality="warn",
            filter="a=0",
            check_func=is_not_null_and_not_empty,
            column="g",
        ),
        DQRowRule(
            name="h_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="h",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="h_as_kwargs_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            check_func_kwargs={"column": "h_as_kwargs"},
            column="h_as_kwargs",
            check_func_args=[[1, 2]],
        ),
        DQRowRule(
            name="i_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="i",
            check_func_kwargs={"column": "i_as_kwargs"},
            check_func_args=[[1, 2]],
        ),
        DQDatasetRule(
            name="g_is_not_unique",
            criticality="warn",
            check_func=is_unique,
            columns=["g"],
        ),
        DQDatasetRule(
            name="g_as_kwargs_is_not_unique",
            criticality="warn",
            check_func=is_unique,
            check_func_kwargs={"columns": ["g_as_kwargs"]},
            columns=["g_as_kwargs"],
        ),
        DQDatasetRule(
            name="j_is_not_unique",
            criticality="warn",
            check_func=is_unique,
            columns=["j"],
            check_func_kwargs={"columns": ["j_as_kwargs"]},
        ),
        DQRowRule(
            name="g_does_not_match_pattern_ipv4_address",
            criticality="warn",
            check_func=is_valid_ipv4_address,
            column="g",
        ),
        DQRowRule(
            name="g_is_not_ipv4_in_cidr",
            criticality="warn",
            check_func=is_ipv4_address_in_cidr,
            column="g",
            check_func_args=["192.168.1.0/24"],
        ),
    ]

    assert pprint.pformat(actual_rules) == pprint.pformat(expected_rules)


def _build_rules_foreach_col(*rules_col_set: DQForEachColRule) -> list[DQRule]:
    """
    Build rules for each column from DQForEachColRule sets.

    Args:
      rules_col_set: list of dq rules which define multiple columns for the same check function

    Returns:
      list of dq rules
    """
    rules_nested = [rule_set.get_rules() for rule_set in rules_col_set]
    flat_rules = list(itertools.chain(*rules_nested))

    return list(filter(None, flat_rules))


def test_build_rules_by_metadata():
    test_build_rules_by_metadata_checks = [
        {
            "check": {
                "function": "is_not_null_and_not_empty",
                "for_each_column": ["a", "b"],
                "arguments": {},
            },
            "user_metadata": {"check_type": "completeness", "check_owner": "someone@email.com"},
        },
        {
            "criticality": "warn",
            "filter": "a>0",
            "check": {
                "function": "is_not_null_and_not_empty",
                "for_each_column": ["c"],
                "arguments": {"column": "another"},
            },
        },
        {
            "criticality": "error",
            "filter": "c=0",
            "check": {"function": "is_in_list", "for_each_column": ["d", "e"], "arguments": {"allowed": [1, 2]}},
        },
        {
            "criticality": "warn",
            "check": {"function": "is_in_list", "for_each_column": ["f"], "arguments": {"allowed": [3]}},
        },
        {
            "name": "g_is_null_or_empty",
            "criticality": "warn",
            "check": {"function": "is_not_null_and_not_empty", "arguments": {"column": "g"}},
        },
        {
            "criticality": "warn",
            "check": {"function": "is_in_list", "arguments": {"column": "h", "allowed": [1, 2]}},
        },
        {
            "name": "d_not_in_a",
            "criticality": "error",
            "check": {
                "function": "sql_expression",
                "arguments": {"expression": "a != substring(b, 8, 1)", "msg": "a not found in b"},
            },
        },
        {
            "criticality": "error",
            "check": {
                "function": "sql_expression",
                "arguments": {
                    "expression": "a != substring(b, 8, 1)",
                    "msg": "a not found in b",
                    "columns": ["a", "b"],
                },
            },
        },
        {
            "check": {"function": "is_not_null_and_not_empty_array", "for_each_column": ["a", "b"]},
        },
        {
            "criticality": "warn",
            "check": {"function": "is_not_null_and_not_empty_array", "for_each_column": ["c"], "arguments": {}},
        },
        {
            "name": "custom_common_name",
            "criticality": "error",
            "check": {"function": "is_not_null", "for_each_column": ["a", "b"], "arguments": {}},
        },
        {
            "criticality": "error",
            "check": {
                "function": "is_unique",
                "for_each_column": [["a", "b"], ["c"]],
                "arguments": {"nulls_distinct": True},
            },
        },
        {
            "name": "is_not_unique_with_filter",
            "criticality": "error",
            "filter": "a > b",
            "check": {
                "function": "is_unique",
                "for_each_column": [["a", "b"], ["c"]],
                "arguments": {"nulls_distinct": True},
            },
        },
        {
            "criticality": "error",
            "check": {
                "function": "is_aggr_not_greater_than",
                "for_each_column": ["a", "*"],
                "arguments": {"limit": 1, "aggr_type": "count", "group_by": ["c"]},
            },
        },
        {
            "name": "count_group_by_c_greater_than_limit_with_filter",
            "criticality": "error",
            "filter": "a > b",
            "check": {
                "function": "is_aggr_not_greater_than",
                "for_each_column": ["a", "*"],
                "arguments": {"limit": 1, "aggr_type": "count", "group_by": ["c"]},
            },
        },
        {
            "criticality": "error",
            "check": {
                "function": "is_aggr_not_less_than",
                "for_each_column": ["a", "*"],
                "arguments": {"limit": 1, "aggr_type": "count", "group_by": ["c"]},
            },
        },
        {
            "name": "foreign_key",
            "criticality": "warn",
            "filter": "a > b",
            "check": {
                "function": "foreign_key",
                "for_each_column": [["a"], ["c"]],
                "arguments": {"ref_columns": ["ref_a"], "ref_df_name": "ref_df_key"},
            },
        },
        {
            "name": "a_does_not_match_pattern_ipv4_address",
            "criticality": "error",
            "check": {"function": "is_valid_ipv4_address", "arguments": {"column": "a"}},
        },
        {
            "name": "a_is_ipv4_address_in_cidr",
            "criticality": "error",
            "check": {
                "function": "is_ipv4_address_in_cidr",
                "arguments": {"column": "a", "cidr_block": "192.168.1.0/24"},
            },
        },
    ]

    actual_rules = deserialize_checks(test_build_rules_by_metadata_checks)

    expected_rules = [
        DQRowRule(
            name="a_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="a",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(
            name="b_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="b",
            user_metadata={"check_type": "completeness", "check_owner": "someone@email.com"},
        ),
        DQRowRule(
            name="c_is_null_or_empty",
            criticality="warn",
            filter="a>0",
            check_func=is_not_null_and_not_empty,
            column="c",
        ),
        DQRowRule(
            name="d_is_not_in_the_list",
            criticality="error",
            filter="c=0",
            check_func=is_in_list,
            column="d",
            check_func_kwargs={"allowed": [1, 2]},
        ),
        DQRowRule(
            name="e_is_not_in_the_list",
            criticality="error",
            filter="c=0",
            check_func=is_in_list,
            column="e",
            check_func_kwargs={"allowed": [1, 2]},
        ),
        DQRowRule(
            name="f_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="f",
            check_func_kwargs={"allowed": [3]},
        ),
        DQRowRule(
            name="g_is_null_or_empty",
            criticality="warn",
            check_func=is_not_null_and_not_empty,
            column="g",
        ),
        DQRowRule(
            name="h_is_not_in_the_list",
            criticality="warn",
            check_func=is_in_list,
            column="h",
            check_func_kwargs={"allowed": [1, 2]},
        ),
        DQRowRule(
            name="d_not_in_a",
            criticality="error",
            check_func=sql_expression,
            check_func_kwargs={"expression": "a != substring(b, 8, 1)", "msg": "a not found in b"},
        ),
        DQRowRule(
            name="a_b_not_a_substring_b_8_1",
            criticality="error",
            check_func=sql_expression,
            columns=["a", "b"],
            check_func_kwargs={
                "expression": "a != substring(b, 8, 1)",
                "msg": "a not found in b",
            },
        ),
        DQRowRule(
            name="a_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="a",
        ),
        DQRowRule(
            name="b_is_null_or_empty_array",
            criticality="error",
            check_func=is_not_null_and_not_empty_array,
            column="b",
        ),
        DQRowRule(
            name="c_is_null_or_empty_array",
            criticality="warn",
            check_func=is_not_null_and_not_empty_array,
            column="c",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="a",
        ),
        DQRowRule(
            name="custom_common_name",
            criticality="error",
            check_func=is_not_null,
            column="b",
        ),
        DQDatasetRule(
            name="struct_a_b_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["a", "b"],
            check_func_kwargs={"nulls_distinct": True},
        ),
        DQDatasetRule(
            name="c_is_not_unique",
            criticality="error",
            check_func=is_unique,
            columns=["c"],
            check_func_kwargs={"nulls_distinct": True},
        ),
        DQDatasetRule(
            name="is_not_unique_with_filter",
            criticality="error",
            check_func=is_unique,
            columns=["a", "b"],
            filter="a > b",
            check_func_kwargs={"nulls_distinct": True},
        ),
        DQDatasetRule(
            name="is_not_unique_with_filter",
            criticality="error",
            check_func=is_unique,
            columns=["c"],
            filter="a > b",
            check_func_kwargs={"nulls_distinct": True},
        ),
        DQDatasetRule(
            name="a_count_group_by_c_greater_than_limit",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="a",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="count_group_by_c_greater_than_limit",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="*",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="count_group_by_c_greater_than_limit_with_filter",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="a",
            filter="a > b",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="count_group_by_c_greater_than_limit_with_filter",
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="*",
            filter="a > b",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="a_count_group_by_c_less_than_limit",
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="a",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="count_group_by_c_less_than_limit",
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="*",
            check_func_kwargs={"limit": 1, "aggr_type": "count", "group_by": ["c"]},
        ),
        DQDatasetRule(
            name="foreign_key",
            criticality="warn",
            check_func=foreign_key,
            columns=["a"],
            filter="a > b",
            check_func_kwargs={
                "ref_columns": ["ref_a"],
                "ref_df_name": "ref_df_key",
            },
        ),
        DQDatasetRule(
            name="foreign_key",
            criticality="warn",
            check_func=foreign_key,
            columns=["c"],
            filter="a > b",
            check_func_kwargs={
                "ref_columns": ["ref_a"],
                "ref_df_name": "ref_df_key",
            },
        ),
        DQRowRule(
            name="a_does_not_match_pattern_ipv4_address",
            criticality="error",
            check_func=is_valid_ipv4_address,
            column="a",
        ),
        DQRowRule(
            name="a_is_ipv4_address_in_cidr",
            criticality="error",
            check_func=is_ipv4_address_in_cidr,
            column="a",
            check_func_kwargs={"cidr_block": "192.168.1.0/24"},
        ),
    ]

    assert pprint.pformat(actual_rules) == pprint.pformat(expected_rules)


def test_build_checks_by_metadata_when_check_spec_is_missing() -> None:
    checks: list[dict] = [{}]  # missing check spec

    with pytest.raises(ValueError, match="'check' field is missing"):
        deserialize_checks(checks)


def test_build_checks_by_metadata_when_function_spec_is_missing() -> None:
    checks: list[dict] = [{"check": {}}]  # missing func spec

    with pytest.raises(ValueError, match="'function' field is missing in the 'check' block"):
        deserialize_checks(checks)


def test_build_checks_by_metadata_when_arguments_are_missing():
    checks = [
        {
            "check": {
                "function": "is_not_null_and_not_empty"
                # missing arguments spec
            }
        }
    ]

    with pytest.raises(
        ValueError, match="No arguments provided for function 'is_not_null_and_not_empty' in the 'arguments' block"
    ):
        deserialize_checks(checks)


def test_build_checks_by_metadata_when_function_does_not_exist():
    checks = [{"check": {"function": "function_does_not_exists", "arguments": {"column": "a"}}}]

    with pytest.raises(ValueError, match="function 'function_does_not_exists' is not defined"):
        deserialize_checks(checks)


def test_build_checks_by_metadata_logging_debug_calls(caplog):
    checks = [
        {
            "criticality": "error",
            "check": {"function": "is_not_null_and_not_empty", "for_each_column": ["a", "b"], "arguments": {}},
        }
    ]
    logger = logging.getLogger("databricks.labs.dqx.checks_resolver")
    logger.setLevel(logging.DEBUG)
    with caplog.at_level("DEBUG"):
        deserialize_checks(checks)
        assert "Resolving function: is_not_null_and_not_empty" in caplog.text


def test_validate_check_func_arguments_too_many_positional():
    with pytest.raises(TypeError, match=re.escape("Too many check_func_args: extra_arg")):
        DQRowRule(
            name="col1_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="col1",
            check_func_args=[[1, 2], "extra_arg"],
        )


def test_validate_check_func_arguments_invalid_keyword():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "2 validation errors for parameters of check_func 'is_in_list'. Verify check_func_args and check_func_kwargs\ninvalid_kwarg\n  Extra inputs are not permitted [type=extra_forbidden, input_value='invalid_kwarg', input_type=str]\n"
        ),
    ):
        DQRowRule(
            name="col1_is_not_in_the_list",
            criticality="error",
            check_func=is_in_list,
            column="col1",
            check_func_kwargs={"allowed": [3], "invalid_kwarg": "invalid_kwarg", "invalid_kwarg2": "invalid_kwarg2"},
        )


def test_validate_correct_dataset_rule_used():
    with pytest.raises(ValueError, match="Function 'is_not_null' is not a dataset-level rule"):
        DQDatasetRule(criticality="error", check_func=is_not_null, columns=["a"])


def test_validate_correct_row_rule_used():
    with pytest.raises(ValueError, match="Function 'is_unique' is not a row-level rule"):
        DQRowRule(criticality="error", check_func=is_unique, column="a")


def test_validate_column_and_columns_provided():
    with pytest.raises(ValueError, match="Both 'column' and 'columns' cannot be provided at the same time"):
        DQDatasetRule(check_func=is_unique, column="a", columns=["b"])


def test_validate_column_and_columns_provided_as_args():
    with pytest.raises(ValueError, match="Both 'column' and 'columns' cannot be provided at the same time"):
        DQDatasetRule(check_func=is_unique, check_func_kwargs={"column": "a", "columns": ["b"]})


def test_register_rule():

    @register_rule_type('single_column')
    class DQMockRule(DQRule):
        """My mocked class"""

        pass

    # Assert registered rule types
    assert "single_column" in REGISTERED_RULE_TYPES
    assert REGISTERED_RULE_TYPES["single_column"] == DQMockRule

    # Assert registered rule klasses
    assert DQMockRule in REGISTERED_RULE_KLASSES
    assert REGISTERED_RULE_KLASSES[DQMockRule] == "single_column"

    @register_rule("single_column")
    def mock_check_func():
        pass

    # Assert that the function is registered correctly
    assert "mock_check_func" in CHECK_FUNC_REGISTRY
    assert CHECK_FUNC_REGISTRY["mock_check_func"] == "single_column"

    assert "mock_check_func" in REGISTERED_FUNCTIONS['single_column']
    assert REGISTERED_FUNCTIONS['single_column']['mock_check_func'] == mock_check_func


def test_row_rule_null_column():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "1 validation error for parameters of check_func 'is_not_null'. Verify check_func_args and check_func_kwargs\ncolumn\n  Field required [type=missing, input_value={}, input_type=dict]\n"
        ),
    ):
        DQRowRule(
            criticality="warn",
            check_func=is_not_null,
            column=None,
        )


def test_dataset_rule_null_column():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "1 validation error for parameters of check_func 'is_aggr_not_greater_than'. Verify check_func_args and check_func_kwargs\ncolumn\n  Field required [type=missing, input_value={'limit': 1}, input_type=dict]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_aggr_not_greater_than,
            column=None,
            check_func_kwargs={
                "limit": 1,
            },
        )


def test_dataset_rule_null_columns_items():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "2 validation errors for DQDatasetRule\ncolumns.0.str\n  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_unique,
            columns=[None],
            check_func_kwargs={
                "limit": 1,
            },
        )


def test_dataset_rule_empty_columns():
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "1 validation error for DQDatasetRule\ncolumns\n  List should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_unique,
            columns=[],
            check_func_kwargs={
                "limit": 1,
            },
        )


def test_row_rule_null_column_in_kwargs():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "2 validation errors for parameters of check_func 'is_not_null'. Verify check_func_args and check_func_kwargs\ncolumn.str\n  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]\n"
        ),
    ):
        DQRowRule(
            criticality="warn",
            check_func=is_not_null,
            check_func_kwargs={
                "column": None,
            },
        )


def test_dataset_rule_null_column_in_kwargs():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "2 validation errors for parameters of check_func 'is_aggr_not_greater_than'. Verify check_func_args and check_func_kwargs\ncolumn.str\n  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]\n    For further information visit https://errors.pydantic.dev/2.10/v/string_type\ncolumn.is-instance[Column]\n  Input should be an instance of Column [type=is_instance_of, input_value=None, input_type=NoneType]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_aggr_not_greater_than,
            check_func_kwargs={
                "column": None,
                "limit": 1,
            },
        )


def test_dataset_rule_empty_columns_in_kwargs():
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "1 validation error for DQDatasetRule\ncolumns\n  List should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_unique,
            check_func_kwargs={
                "columns": [],
                "limit": 1,
            },
        )


@pytest.mark.parametrize(
    "columns, ref_columns, exclude_columns",
    [
        ([F.col("a") + F.lit(1)], [F.col("b")], [F.col("c")]),
        ([F.col("b")], [F.col("a") + F.lit(1)], None),
        ([F.col("a")], [F.col("b")], [F.col("c") + F.lit(1)]),
    ],
)
def test_compare_datasets_when_column_expression_is_complex(
    columns: list[str | Column], ref_columns: list[str | Column], exclude_columns: list[str | Column]
) -> None:
    with pytest.raises(ValueError, match="Unable to interpret column expression. Only simple references are allowed"):
        DQDatasetRule(
            criticality="error",
            check_func=compare_datasets,
            columns=columns,
            check_func_kwargs={
                "ref_columns": ref_columns,
                "exclude_columns": exclude_columns,
                "ref_table": "_",
            },
        )


def test_dataset_rule_null_columns_items_in_kwargs():
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "3 validation errors for DQDatasetRule\ncolumns.0.str\n  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]\n    For further information visit https://errors.pydantic.dev/2.10/v/string_type\ncolumns.0.is-instance[Column]\n  Input should be an instance of Column [type=is_instance_of, input_value=None, input_type=NoneType]\n"
        ),
    ):
        DQDatasetRule(
            criticality="warn",
            check_func=is_unique,
            check_func_kwargs={
                "columns": [None],
                "limit": 1,
            },
        )


def test_convert_dq_rules_to_metadata():
    checks = [
        DQRowRule(
            check_func=is_not_null_and_not_empty,
            column="a",
        ),
        DQRowRule(
            criticality="warn",
            check_func=is_not_null_and_is_in_list,
            column=F.col("c"),
            check_func_kwargs={"allowed": ["a", F.col("d")]},
        ),
        DQRowRule(
            criticality="warn",
            check_func=sql_expression,
            check_func_kwargs={"expression": "col1 like 'str%'", "msg": "col1 not starting with 'str'"},
        ),
        DQRowRule(criticality="error", check_func=is_not_null_and_not_empty, column="col1"),
        DQRowRule(
            name="col3_is_null_or_empty",
            criticality="warn",
            check_func=is_not_null_and_not_empty,
            column=F.col("col3"),
        ),
        DQRowRule(
            criticality="warn",
            check_func=is_not_null_and_not_empty,
            column=F.col("col3"),
            user_metadata={"check_type": "completeness", "responsible_data_steward": "someone@email.com"},
        ),
        DQRowRule(criticality="warn", check_func=regex_match, column=F.col("col3"), check_func_kwargs={"regex": "dqx"}),
        DQRowRule(criticality="warn", check_func=is_in_list, column="col1", check_func_args=[[1, 2]]),
        DQRowRule(criticality="warn", check_func=is_in_list, column="col2", check_func_kwargs={"allowed": [1, 2]}),
        DQRowRule(check_func=is_not_null, column="col7.field1"),
        DQRowRule(
            name="b_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="b",
            filter="a<3",
        ),
        DQRowRule(
            name="b_is_less_than",
            criticality="error",
            check_func=is_not_less_than,
            column="b",
            check_func_kwargs={"limit": datetime.date(2024, 7, 28)},
        ),
        DQRowRule(
            name="d_is_less_than",
            criticality="error",
            check_func=is_not_less_than,
            column="d",
            check_func_kwargs={"limit": "2025-01-21"},
        ),
        DQRowRule(criticality="error", check_func=is_not_greater_than, column="c", check_func_args=["2022-01-01"]),
        DQRowRule(
            criticality="error",
            check_func=is_not_greater_than,
            column="a",
            check_func_args=[datetime.datetime(2022, 1, 1, 14, 30, 0)],
        ),
        DQRowRule(
            criticality="error", check_func=is_valid_date, column="b", check_func_kwargs={"date_format": "yyyy-MM-dd"}
        ),
        DQDatasetRule(criticality="error", check_func=is_unique, columns=["col1", "col2"]),
        DQDatasetRule(
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="col1",
            check_func_kwargs={"aggr_type": "count", "limit": 10},
        ),
        DQDatasetRule(
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="col1",
            check_func_kwargs={"aggr_type": "avg", "limit": 1.2},
        ),
        *DQForEachColRule(columns=["col1", "col2"], criticality="error", check_func=is_not_null).get_rules(),
        *DQForEachColRule(
            name="common_name2",
            check_func=is_unique,
            criticality="warn",
            columns=[["a", "b"], ["c"], ["d"]],
            check_func_kwargs={"nulls_distinct": False},
        ).get_rules(),
        DQDatasetRule(
            criticality="error", check_func=is_unique, columns=["col1"], check_func_kwargs={"row_filter": "col2 > 0"}
        ),
    ]
    actual_metadata = serialize_checks(checks)

    expected_metadata = [
        {
            "name": "a_is_null_or_empty",
            "criticality": "error",
            "check": {
                "function": "is_not_null_and_not_empty",
                "arguments": {"column": "a"},
            },
        },
        {
            "name": "c_is_null_or_is_not_in_the_list",
            "criticality": "warn",
            "check": {
                "function": "is_not_null_and_is_in_list",
                "arguments": {"column": "c", "allowed": ["a", "d"]},
            },
        },
        {
            "name": "not_col1_like_str",
            "criticality": "warn",
            "check": {
                "function": "sql_expression",
                "arguments": {"expression": "col1 like 'str%'", "msg": "col1 not starting with 'str'"},
            },
        },
        {
            "name": "col1_is_null_or_empty",
            "criticality": "error",
            "check": {
                "function": "is_not_null_and_not_empty",
                "arguments": {"column": "col1"},
            },
        },
        {
            "name": "col3_is_null_or_empty",
            "criticality": "warn",
            "check": {
                "function": "is_not_null_and_not_empty",
                "arguments": {"column": "col3"},
            },
        },
        {
            "name": "col3_is_null_or_empty",
            "criticality": "warn",
            "check": {
                "function": "is_not_null_and_not_empty",
                "arguments": {"column": "col3"},
            },
            "user_metadata": {"check_type": "completeness", "responsible_data_steward": "someone@email.com"},
        },
        {
            "name": "col3_not_matching_regex",
            "criticality": "warn",
            "check": {
                "function": "regex_match",
                "arguments": {"column": "col3", "regex": "dqx"},
            },
        },
        {
            "name": "col1_is_not_in_the_list",
            "criticality": "warn",
            "check": {
                "function": "is_in_list",
                "arguments": {"column": "col1", "allowed": [1, 2]},
            },
        },
        {
            "name": "col2_is_not_in_the_list",
            "criticality": "warn",
            "check": {
                "function": "is_in_list",
                "arguments": {"column": "col2", "allowed": [1, 2]},
            },
        },
        {
            "name": "col7_field1_is_null",
            "criticality": "error",
            "check": {
                "function": "is_not_null",
                "arguments": {"column": "col7.field1"},
            },
        },
        {
            "name": "b_is_null_or_empty",
            "criticality": "error",
            "filter": "a<3",
            "check": {
                "function": "is_not_null_and_not_empty",
                "arguments": {"column": "b"},
            },
        },
        {
            "name": "b_is_less_than",
            "criticality": "error",
            "check": {
                "function": "is_not_less_than",
                "arguments": {"column": "b", "limit": "2024-07-28"},
            },
        },
        {
            "name": "d_is_less_than",
            "criticality": "error",
            "check": {
                "function": "is_not_less_than",
                "arguments": {"column": "d", "limit": "2025-01-21"},
            },
        },
        {
            "name": "c_greater_than_limit",
            "criticality": "error",
            "check": {
                "function": "is_not_greater_than",
                "arguments": {"column": "c", "limit": "2022-01-01"},
            },
        },
        {
            "name": "a_greater_than_limit",
            "criticality": "error",
            "check": {
                "function": "is_not_greater_than",
                "arguments": {"column": "a", "limit": "2022-01-01 14:30:00"},
            },
        },
        {
            "name": "b_is_not_valid_date",
            "criticality": "error",
            "check": {
                "function": "is_valid_date",
                "arguments": {"column": "b", "date_format": "yyyy-MM-dd"},
            },
        },
        {
            "name": "struct_col1_col2_is_not_unique",
            "criticality": "error",
            "check": {
                "function": "is_unique",
                "arguments": {"columns": ["col1", "col2"]},
            },
        },
        {
            "name": "col1_count_greater_than_limit",
            "criticality": "error",
            "check": {
                "function": "is_aggr_not_greater_than",
                "arguments": {"column": "col1", "aggr_type": "count", "limit": 10},
            },
        },
        {
            "name": "col1_avg_less_than_limit",
            "criticality": "error",
            "check": {
                "function": "is_aggr_not_less_than",
                "arguments": {"column": "col1", "aggr_type": "avg", "limit": 1.2},
            },
        },
        {
            "name": "col1_is_null",
            "criticality": "error",
            "check": {
                "function": "is_not_null",
                "arguments": {"column": "col1"},
            },
        },
        {
            "name": "col2_is_null",
            "criticality": "error",
            "check": {
                "function": "is_not_null",
                "arguments": {"column": "col2"},
            },
        },
        {
            "name": "common_name2",
            "criticality": "warn",
            "check": {
                "function": "is_unique",
                "arguments": {"columns": ["a", "b"], "nulls_distinct": False},
            },
        },
        {
            "name": "common_name2",
            "criticality": "warn",
            "check": {
                "function": "is_unique",
                "arguments": {"columns": ["c"], "nulls_distinct": False},
            },
        },
        {
            "name": "common_name2",
            "criticality": "warn",
            "check": {
                "function": "is_unique",
                "arguments": {"columns": ["d"], "nulls_distinct": False},
            },
        },
        {
            'name': 'col1_is_not_unique',
            'criticality': 'error',
            'check': {'function': 'is_unique', 'arguments': {'columns': ['col1'], 'row_filter': 'col2 > 0'}},
        },
    ]

    assert actual_metadata == expected_metadata


def test_convert_dq_rules_to_metadata_when_empty() -> None:
    checks: list = []
    actual_metadata = serialize_checks(checks)
    expected_metadata: list[dict] = []
    assert actual_metadata == expected_metadata


def test_convert_dq_rules_to_metadata_when_not_dq_rule() -> None:
    checks: list = [1]
    with pytest.raises(TypeError, match="Expected DQRule instance, got int"):
        serialize_checks(checks)


def test_dq_rules_to_dict_when_column_expression_is_complex() -> None:
    r = DQRowRule(
        criticality="error",
        check_func=is_not_null_and_not_empty,
        column=F.col("val") + F.lit(1),
    )
    with pytest.raises(ValueError, match="Unable to interpret column expression"):
        r.to_dict()


def test_dq_rules_to_dict_when_invalid_arg_type() -> None:
    col_dict = {"key1": "col1"}
    r = DQRowRule(
        criticality="warn",
        check_func=is_not_null_and_is_in_list,
        column=F.col("c"),
        check_func_kwargs={"allowed": col_dict.values()},
    )

    with pytest.raises(ValueError, match=re.escape("Unable to serialize unknown type: <class 'dict_values'>")):
        r.to_dict()


def test_metadata_round_trip_conversion_preserves_rules() -> None:
    checks = [
        DQRowRule(
            check_func=is_not_null_and_not_empty,
            column="a",
        ),
        DQRowRule(
            criticality="warn",
            check_func=is_not_null_and_is_in_list,
            column=F.col("c"),
            check_func_kwargs={"allowed": ["a", F.col("d")]},
        ),
        DQRowRule(
            criticality="warn",
            check_func=sql_expression,
            check_func_kwargs={"expression": "col1 like 'str%'", "msg": "col1 not starting with 'str'"},
        ),
        DQRowRule(criticality="error", check_func=is_not_null_and_not_empty, column="col1"),
        DQRowRule(
            name="col3_is_null_or_empty",
            criticality="warn",
            check_func=is_not_null_and_not_empty,
            column=F.col("col3"),
        ),
        DQRowRule(
            criticality="warn",
            check_func=is_not_null_and_not_empty,
            column=F.col("col3"),
            user_metadata={"check_type": "completeness", "responsible_data_steward": "someone@email.com"},
        ),
        DQRowRule(criticality="warn", check_func=regex_match, column=F.col('col3'), check_func_kwargs={"regex": "dqx"}),
        DQRowRule(criticality="warn", check_func=is_in_list, column="col1", check_func_args=[[1, 2]]),
        DQRowRule(criticality="warn", check_func=is_in_list, column="col2", check_func_kwargs={"allowed": [1, 2]}),
        DQRowRule(check_func=is_not_null, column="col7.field1"),
        DQRowRule(
            name="b_is_null_or_empty",
            criticality="error",
            check_func=is_not_null_and_not_empty,
            column="b",
            filter="a<3",
        ),
        DQRowRule(
            name="b_is_less_than",
            criticality="error",
            check_func=is_not_less_than,
            column="b",
            check_func_kwargs={"limit": datetime.date(2024, 7, 28)},
        ),
        DQRowRule(
            name="d_is_less_than",
            criticality="error",
            check_func=is_not_less_than,
            column="d",
            check_func_kwargs={"limit": "2025-01-21"},
        ),
        DQRowRule(criticality="error", check_func=is_not_greater_than, column="c", check_func_args=["2022-01-01"]),
        DQRowRule(
            criticality="error",
            check_func=is_not_greater_than,
            column="a",
            check_func_args=[datetime.datetime(2022, 1, 1, 14, 30, 0)],
        ),
        DQRowRule(
            criticality="error", check_func=is_valid_date, column="b", check_func_kwargs={"date_format": "yyyy-MM-dd"}
        ),
        DQDatasetRule(criticality="error", check_func=is_unique, columns=["col1", "col2"]),
        DQDatasetRule(
            criticality="error",
            check_func=is_aggr_not_greater_than,
            column="col1",
            check_func_kwargs={"aggr_type": "count", "limit": 10},
        ),
        DQDatasetRule(
            criticality="error",
            check_func=is_aggr_not_less_than,
            column="col1",
            check_func_kwargs={"aggr_type": "avg", "limit": 1.2},
        ),
        *DQForEachColRule(columns=["col1", "col2"], criticality="error", check_func=is_not_null).get_rules(),
        *DQForEachColRule(
            name="common_name2",
            check_func=is_unique,
            criticality="warn",
            columns=[["a", "b"], ["c"], ["d"]],
            check_func_kwargs={"nulls_distinct": False},
        ).get_rules(),
        DQDatasetRule(
            criticality="error", check_func=is_unique, columns=["col1"], check_func_kwargs={"row_filter": "col2 > 0"}
        ),
    ]

    checks_dict = serialize_checks(checks)
    converted_checks = deserialize_checks(checks_dict)

    assert serialize_checks(converted_checks) == serialize_checks(checks)


@pytest.mark.parametrize(
    "checks, file_path_suffix, expected_output",
    [
        ([{"key": "value"}], ".json", json.dumps([{"key": "value"}]).encode("utf-8")),
        ([{"key": "value"}], ".yaml", yaml.safe_dump([{"key": "value"}]).encode("utf-8")),
        ([{"key": "value"}], ".yml", yaml.safe_dump([{"key": "value"}]).encode("utf-8")),
        ([{"key": "value"}], "", yaml.safe_dump([{"key": "value"}]).encode("utf-8")),  # Default to YAML if no extension
    ],
)
def test_serialize_checks_to_bytes(checks, file_path_suffix, expected_output):
    mock_path = Mock(spec=Path)
    mock_path.suffix = file_path_suffix
    result = serialize_checks_to_bytes(checks, mock_path)
    assert result == expected_output
