"""Unit tests for the cached tool-parameter schema check."""

import unittest
from unittest import mock

from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.function_call import utils
from sglang.srt.function_call.utils import check_tool_parameters_schema
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


class TestCheckToolParametersSchema(CustomTestCase):
    def setUp(self):
        utils._valid_tool_schemas.clear()
        self.real_check = Draft202012Validator.check_schema
        patcher = mock.patch.object(
            utils.Draft202012Validator,
            "check_schema",
            side_effect=self.real_check,
        )
        self.check = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(utils._valid_tool_schemas.clear)

    def test_valid_schema_is_checked_once(self):
        check_tool_parameters_schema(
            _schema(a={"type": "string"}, b={"type": "integer"})
        )
        # Equal content, different key order and object identity.
        check_tool_parameters_schema(
            {
                "required": ["a", "b"],
                "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
                "type": "object",
            }
        )
        self.assertEqual(self.check.call_count, 1)

    def test_different_schemas_are_each_checked(self):
        check_tool_parameters_schema(_schema(a={"type": "string"}))
        check_tool_parameters_schema(_schema(a={"type": "number"}))
        self.assertEqual(self.check.call_count, 2)

    def test_invalid_schema_raises_every_time(self):
        bad = _schema(a={"type": "not-a-type"})
        for _ in range(2):
            with self.assertRaises(SchemaError):
                check_tool_parameters_schema(bad)
        self.assertEqual(self.check.call_count, 2)

    def test_nan_is_not_confused_with_null(self):
        # orjson would serialize both as null; the NaN one is valid, null is not.
        check_tool_parameters_schema({"type": "number", "minimum": float("nan")})
        with self.assertRaises(SchemaError):
            check_tool_parameters_schema({"type": "number", "minimum": None})

    def test_non_json_schema_is_checked_without_caching(self):
        schema = {"type": "object", "default": {1, 2}}  # a set: not JSON data
        check_tool_parameters_schema(schema)
        check_tool_parameters_schema(schema)
        self.assertEqual(self.check.call_count, 2)
        self.assertEqual(len(utils._valid_tool_schemas), 0)

    def test_cyclic_schema_still_raises_recursion_error(self):
        schema = {"type": "object", "properties": {}}
        schema["properties"]["self"] = schema
        with self.assertRaises(RecursionError):
            check_tool_parameters_schema(schema)

    def test_least_recently_used_schema_is_evicted(self):
        a, b, c = (_schema(x={"type": t}) for t in ("string", "number", "boolean"))
        with mock.patch.object(utils, "_VALID_TOOL_SCHEMAS_MAX_SIZE", 2):
            check_tool_parameters_schema(a)
            check_tool_parameters_schema(b)
            check_tool_parameters_schema(a)  # hit: a becomes most recent
            check_tool_parameters_schema(c)  # evicts b
            self.assertEqual(self.check.call_count, 3)
            check_tool_parameters_schema(a)
            self.assertEqual(self.check.call_count, 3)
            check_tool_parameters_schema(b)
            self.assertEqual(self.check.call_count, 4)


if __name__ == "__main__":
    unittest.main()
