import inspect
import logging
from enum import Enum
from typing import Any
from abc import abstractmethod

import functools
import yaml

from typing import Callable, Annotated

from pydantic import (
    BaseModel,
    model_validator,
    Field,
    ConfigDict,
    field_validator,
    field_serializer,
    PlainSerializer,
    create_model,
    WithJsonSchema,
    ValidationError,
)

from databricks.labs.dqx.utils import get_column_name_or_alias

from .types import Column


logger = logging.getLogger(__name__)


class Criticality(Enum):
    """Enum class to represent criticality of the check."""

    WARN = "warn"
    ERROR = "error"


class DefaultColumnNames(Enum):
    """Enum class to represent columns in the dataframe that will be used for error and warning reporting."""

    ERRORS = "_errors"
    WARNINGS = "_warnings"


class ColumnArguments(Enum):
    """Enum class that is used as input parsing for custom column naming."""

    ERRORS = "errors"
    WARNINGS = "warnings"


class LazyFunctionBaseModel(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    def __init__(self, *args, **kwargs):
        # make args work as if they were kwargs
        mapped_args = dict(zip(self.model_fields, args))
        fixed_kwargs = {**mapped_args, **kwargs}
        super().__init__(**fixed_kwargs)

    @classmethod
    def create_model_from_function(cls, func: Callable):
        return cls.create_model_from_function_signature(func.__name__, inspect.signature(func))

    @classmethod
    def create_model_from_function_signature(
        cls, model_name: str, sig: inspect.Signature
    ) -> type['LazyFunctionBaseModel']:
        model_fields = {}

        for name, prop in sig.parameters.items():
            annotation = prop.annotation if prop.annotation else None
            field = Field() if prop.default is inspect.Parameter.empty else Field(default=prop.default)

            model_fields[name] = (annotation, field)

        validator = create_model(model_name, __base__=LazyFunctionBaseModel, **model_fields)

        return validator


def lazy_validate_call(func, *args, **kw):
    """Returns a callable with validated arguments. This decorator is a replacement for functools.partial, but it also validates the arguments before returning the callable."""
    sig = inspect.signature(func)
    validator = LazyFunctionBaseModel.create_model_from_function_signature(func.__qualname__, sig)
    validator.__call__ = lambda self: func(**{name: getattr(self, name) for name in sig.parameters})
    return validator


REGISTERED_RULE_TYPES: dict[str, type['DQCheckFunctionBaseModel']] = {}
REGISTERED_RULE_KLASSES: dict[type['DQCheckFunctionBaseModel'], str] = {}
REGISTERED_FUNCTIONS: dict[str, dict[str, Callable]] = {}
CHECK_FUNC_REGISTRY: dict[str, str] = {}


def register_rule_type(rule_type: str):
    """Registeres class for handling the rule type. Cleans up all functions for a rule type"""

    def _wrapper(klass):
        REGISTERED_RULE_TYPES[rule_type] = klass
        REGISTERED_RULE_KLASSES[klass] = rule_type
        REGISTERED_FUNCTIONS[rule_type] = {}
        return klass

    return _wrapper


def _get_rule_klasses_for_function(value: str | Callable) -> list[type['DQCheckFunctionBaseModel']]:
    """Returns list of rule type klasses that implements given function name, or callable"""
    result: list[type['DQCheckFunctionBaseModel']] = []

    for rule_type, func_map in REGISTERED_FUNCTIONS.items():
        for func_name, func in func_map.items():
            if isinstance(value, str) and func_name == value:
                result.append(REGISTERED_RULE_TYPES[rule_type])
            if callable(value) and func == value:
                result.append(REGISTERED_RULE_TYPES[rule_type])

    return result


def register_rule(rule_type: str) -> Callable:
    """Registers rule callable of provided rule_type"""

    def _wrapper(func: Callable) -> Callable:
        mapping = REGISTERED_FUNCTIONS.get(rule_type)
        if mapping is None:
            raise NotImplementedError(f"Not supported rule type: {rule_type!r}")

        # guard for functions having overlapping names across rule_types
        if existing_type := CHECK_FUNC_REGISTRY.get(func.__name__):
            if rule_type != existing_type:
                raise ValueError(
                    f"Cannot reigsterd function {func.__name__!r} with rule type: {rule_type!r}: already registered with rule_type={existing_type!r}"
                )

        CHECK_FUNC_REGISTRY[func.__name__] = rule_type

        mapping[func.__name__] = func
        return func

    return _wrapper


# https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers
# https://github.com/pydantic/pydantic/discussions/7510
SerializableCallable = Annotated[
    Callable,
    PlainSerializer(lambda x: x.__name__, return_type=str, when_used='json'),
    WithJsonSchema({'type': 'string'}, mode='serialization'),
    WithJsonSchema({'type': 'string'}, mode='validation'),
]


class DQBaseModel(BaseModel):
    # extra='forbid': don't allow extra fields, that are not defined in a model
    # arbitrary_types_allowed=True to handle pyspark Column
    model_config = ConfigDict(extra='forbid', frozen=True, arbitrary_types_allowed=True)

    def to_dict(self):
        return self.model_dump(exclude_defaults=True, exclude_none=True, mode='json')

    def model_dump_yaml(self):
        d = self.to_dict()
        return yaml.safe_dump(d)

    @classmethod
    def model_validate_yaml(cls, body: str):
        d = yaml.safe_load(body)
        return cls.model_validate(d)


class DQCheckFunctionBaseModel(DQBaseModel):
    # https://docs.pydantic.dev/latest/concepts/models/#class-variables
    check_func: SerializableCallable
    check_func_args: list[Any] = Field(default_factory=list)
    check_func_kwargs: dict[str, Any] = Field(default_factory=dict)

    @functools.cached_property
    def check_func_signature(self):
        return inspect.signature(self.check_func)

    @functools.cached_property
    def check_func_optional_params(self) -> dict[str, inspect.Parameter]:
        return {
            name: param
            for name, param in self.check_func_signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

    @functools.cached_property
    def check_func_required_params(self) -> dict[str, inspect.Parameter]:
        return {
            name: param
            for name, param in self.check_func_signature.parameters.items()
            if name not in self.check_func_optional_params
        }

    @property
    def check_func_rule_type(self):
        return CHECK_FUNC_REGISTRY[self.check_func.__name__]

    @classmethod
    def get_check_function(cls, name: str) -> Callable:
        rule_tupe = REGISTERED_RULE_KLASSES[cls]
        if not rule_tupe:
            raise ValueError(f"{rule_tupe!r} is not a registered rule type")

        mapping = REGISTERED_FUNCTIONS.get(rule_tupe)
        if mapping is None:
            raise ValueError(f"{rule_tupe!r} is not a registered rule type")

        func = mapping.get(name)
        if not func:
            found_klasses = _get_rule_klasses_for_function(name)
            msg = f"Check function {name!r} is not registered for {cls.__qualname__}"
            if found_klasses:
                msg += f". Did you mean to use {', '.join([k.__name__ for k in found_klasses])}?"
            raise ValueError(msg)

        return func

    # https://docs.pydantic.dev/latest/concepts/validators/#field-validators
    # before validators are plain data type validators, without business logic
    @field_validator('check_func', mode='before')
    @classmethod
    def ensure_check_func_is_callable(cls, value: Any) -> Callable:
        """Gets execued each time value is setattr, but before it's actually set"""
        if callable(value):
            return value
        elif isinstance(value, str):
            func = cls.get_check_function(value)

            return func

        # whatever else, will be passed to default pydantic validators
        return value

    @classmethod
    def _serialize(cls, val: Any):
        if val is None:
            return None

        if isinstance(val, (list, tuple, set)):
            return [cls._serialize(v) for v in val]

        if isinstance(val, dict):
            return {k: cls._serialize(v) for k, v in val.items()}

        if isinstance(val, Column):
            return get_column_name_or_alias(val, allow_simple_expressions_only=True)

        if callable(val):
            return val.__name__

        return val

    @field_serializer('check_func', 'check_func_args', 'check_func_kwargs', when_used='json')
    def serialize_check_func_fields(self, val: Any, mode):
        return self._serialize(val)

    def _check_func_call(self):
        return self.check_func(*self.check_func_args, **self.check_func_kwargs)


class DQRule(DQCheckFunctionBaseModel):
    """Represents a data quality rule that applies a quality check function to column(s) or
    column expression(s). This class includes the following attributes:
    * *check_func* - The function used to perform the quality check.
    * *name* (optional) - A custom name for the check; autogenerated if not provided.
    * *criticality* (optional) - Defines the severity level of the check:
        - *error*: Critical issues.
        - *warn*: Potential issues.
    * *column* (optional) - A single column to which the check function is applied.
    * *columns* (optional) - A list of columns to which the check function is applied.
    * *filter* (optional) - A filter expression to apply the check only to rows meeting specific conditions.
    * *check_func_args* (optional) - Positional arguments for the check function (excluding *column*).
    * *check_func_kwargs* (optional) - Keyword arguments for the check function (excluding *column*).
    * *user_metadata* (optional) - User-defined key-value pairs added to metadata generated by the check.
    """

    name: str | None = Field(default=None)
    criticality: Criticality = Field(default=Criticality.ERROR)
    column: str | Column | None = Field(default=None)
    columns: list[str | Column] | None = Field(default=None, min_length=1)
    filter: str | None = Field(default=None)
    user_metadata: dict[str, str] | None = None

    @property
    def rule_type(self):
        my_type = REGISTERED_RULE_KLASSES.get(self.__class__)
        if not my_type:
            raise ValueError(f"{self.__class__.__qualname__} is not a registered rule type")

        return my_type

    @property
    @abstractmethod
    def check_condition(self) -> Column:
        """
        Compute the check condition for this rule.
        Returns:
            The Spark Column representing the check condition.
        """
        pass

    # model_validator are executed in order they are defined
    @model_validator(mode="before")
    def _validate_dq_row_rules_raw(cls, data: Any):
        # it should be a dict, if not passthough and let higher level thrown nice erros
        if not data or not isinstance(data, dict):
            return data

        # copy kwargs to model, if model fields are empty
        cls._sync_kwarg_fields_to_model(data, "column", "column")
        cls._sync_kwarg_fields_to_model(data, "columns", "columns")
        cls._sync_kwarg_fields_to_model(data, "filter", "row_filter")

        # the type checks will be performed on `data` once this function finishes
        return data

    @model_validator(mode="after")
    def _validate_dq_row_rules(self):
        # executed after `self` if created, and all fields have validated data type
        if self.column is not None and self.columns is not None:
            raise ValueError("Both 'column' and 'columns' cannot be provided at the same time.")

        my_type = REGISTERED_RULE_KLASSES.get(self.__class__)
        if not my_type:
            raise ValueError(f"{self.__class__.__qualname__} is not a registered rule type")

        if self.check_func_rule_type != my_type:
            raise ValueError(
                f"Function '{self.check_func.__name__}' is not a {my_type}-level rule. "
                f"Use {self.check_func_rule_type} instead."
            )

        # populate kwargs, based on model fields, overwrite them if kwargs have them already
        self._sync_model_fields_to_required_kwargs("column", "column")
        self._sync_model_fields_to_required_kwargs("columns", "columns")
        self._sync_model_fields_to_required_kwargs("filter", "row_filter")

        # promote args to kwargs
        self.check_func_kwargs.update(
            self._build_kwargs_from_out_of_order_params(
                self.check_func_signature, *self.check_func_args, **self.check_func_kwargs
            )
        )
        self.check_func_args.clear()

        return self

    @model_validator(mode="after")
    def _validate_check_func_bind(self):
        bind_check = lazy_validate_call(self.check_func)

        try:
            # returns callable, without executing it
            # validates the params to match signature
            c = bind_check(*self.check_func_args, **self.check_func_kwargs)
            assert callable(c)
        except ValidationError as e:
            # let's make cleaner title, validation error is not mutable
            nicer_e = ValidationError.from_exception_data(
                title=f"parameters of check_func {self.check_func.__name__!r}. Verify check_func_args and check_func_kwargs",
                line_errors=e.errors(),  # type: ignore
            )
            # throwing ValidationError will remove nice title and replace with generic one, let's throw TypeError
            # supress unhandled context: https://docs.python.org/3/library/exceptions.html#exception-context
            final_exception = TypeError(str(nicer_e))
            final_exception.__suppress_context__ = True
            raise final_exception

        return self

    @model_validator(mode="after")
    def _validate_name(self):
        # make it always run, to cache the value
        check_condition = self.check_condition

        if not self.name:
            normalized_name = get_column_name_or_alias(check_condition, normalize=True)
            if not normalized_name:
                raise ValueError("name is not provided")

            object.__setattr__(self, "name", normalized_name)

        return self

    @staticmethod
    def _sync_kwarg_fields_to_model(
        data: dict[str, Any], model_field: str, kwargs_field: str, raise_on_conflict: bool = False
    ):
        """Mutates data, so that kwargs_field is copied over from check_func_kwargs into model_field, if model_field is not empty"""
        m = data.get(model_field)
        k = data.get('check_func_kwargs', {}).get(kwargs_field)

        if raise_on_conflict:
            if m is not None and k is not None and m != k:
                raise ValueError(
                    f"""Both '{model_field}' (value={m!r}) and 'check_func_kwargs["{kwargs_field}"]' (value={k!r}) cannot be provided with different values at the same time. Provide value in either of them, or make sure that both values are the same."""
                )

        if m is None and k is not None:
            data[model_field] = k

    def _sync_model_fields_to_required_kwargs(self, model_field: str, kwargs_field: str):
        """Mutates current model, so that model_field is added to check_func_kwargs, if it's absent, and is requires on a check_func_singature"""
        required_params = self.check_func_required_params

        m = getattr(self, model_field)
        if m is None:
            return

        # not an optional param
        if kwargs_field not in required_params:
            return

        # despite model being frozen, dicts are mutable, hence can be modified at will
        self.check_func_kwargs[kwargs_field] = m

    @classmethod
    def _build_kwargs_from_out_of_order_params(cls, __sig: inspect.Signature, *args, **kwargs) -> dict[str, Any]:
        """Returns kwargs, by combining both args and kwargs. Handles out of order notation where args are put after kwargs."""
        not_kwargs_fields = [f for f in __sig.parameters if f not in kwargs]

        diff = len(args) - len(not_kwargs_fields)
        if diff > 0:
            too_many_args = args[diff:]
            raise TypeError(f"Too many check_func_args: {', '.join(too_many_args)}")

        mapped_args = dict(zip(not_kwargs_fields, args))  # type: ignore
        fixed_kwargs = {**mapped_args, **kwargs}

        # print(f"{fixed_kwargs=}")
        # print(f"  {args=}")
        # print(f"  {kwargs=}")

        return fixed_kwargs

    @field_serializer('column', when_used='json')
    def serialize_column(self, column: str | Column, _info):
        if column is None:
            return None

        return get_column_name_or_alias(column)


@register_rule_type('row')
class DQRowRule(DQRule):
    """
    Represents a row-level data quality rule that applies a quality check function to a column or column expression.
    Works with check functions that take a single column or no column as input.
    """

    @functools.cached_property
    def check(self) -> Column:
        condition = self.check_func(*self.check_func_args, **self.check_func_kwargs)
        return condition

    @property
    def check_condition(self) -> Column:
        """
        Compute the check condition for this rule.
        Returns:
            The Spark Column representing the check condition.
        """
        return self.check


@register_rule_type('dataset')
class DQDatasetRule(DQRule):
    """
    Represents a dataset-level data quality rule that applies a quality check function to a column or
    column expression or list of columns depending on the check function.
    Either column or columns can be provided but not both. The rules are applied to the entire dataset or group of rows
    rather than individual rows. Failed checks are appended to the result columns in the same way as row-level rules.
    """

    @property
    def check(self) -> tuple[Column, Callable]:
        condition, apply_func = self.check_func(*self.check_func_args, **self.check_func_kwargs)
        return condition, apply_func

    @property
    def check_condition(self) -> Column:
        """
        Compute the check condition for this rule.
        Returns:
            The Spark Column representing the check condition.
        """
        check_condition, _ = self.check  # lazy evaluation of check function parameters
        return check_condition


class DQForEachColRule(DQBaseModel):
    """Represents a data quality rule that applies to a quality check function
    repeatedly on each specified column of the provided list of columns.
    This class includes the following attributes:
    * *columns* - A list of column names or expressions to which the check function should be applied.
    * *check_func* - The function used to perform the quality check.
    * *name* (optional) - A custom name for the check; autogenerated if not provided.
    * *criticality* - The severity level of the check:
        - *warn* for potential issues.
        - *error* for critical issues.
    * *filter* (optional) - A filter expression to apply the check only to rows meeting specific conditions.
    * *check_func_args* (optional) - Positional arguments for the check function (excluding column names).
    * *check_func_kwargs* (optional) - Keyword arguments for the check function (excluding column names).
    * *user_metadata* (optional) - User-defined key-value pairs added to metadata generated by the check.
    """

    columns: list[str | Column] | list[list[str | Column]] | None = Field(default=None)
    name: str | None = Field(default=None)
    criticality: Criticality = Field(default=Criticality.ERROR)
    filter: str | None = Field(default=None)
    user_metadata: dict[str, str] | None = None
    check_func: SerializableCallable
    check_func_args: list[Any] = Field(default_factory=list)
    check_func_kwargs: dict[str, Any] = Field(default_factory=dict)

    def get_rules(self) -> list[DQRule]:
        """Build a list of rules for a set of columns.

        Returns:
            list of dq rules
        """
        rules: list[DQRule] = []
        for column in self.columns or []:
            rule_type = CHECK_FUNC_REGISTRY.get(self.check_func.__name__)
            if not rule_type:
                raise ValueError(f"{self.check_func.__name__!r} is not a registered function")

            effective_column = column if not isinstance(column, list) else None
            effective_columns = column if isinstance(column, list) else None

            if rule_type == "dataset":  # user must register dataset-level rules
                rules.append(
                    DQDatasetRule(
                        column=effective_column,
                        columns=effective_columns,
                        check_func=self.check_func,
                        check_func_kwargs=self.check_func_kwargs,
                        check_func_args=self.check_func_args,
                        name=self.name,
                        criticality=self.criticality,
                        filter=self.filter,
                        user_metadata=self.user_metadata,
                    )
                )
            elif rule_type == "row":
                rules.append(
                    DQRowRule(
                        column=effective_column,
                        columns=effective_columns,
                        check_func=self.check_func,
                        check_func_kwargs=self.check_func_kwargs,
                        check_func_args=self.check_func_args,
                        name=self.name,
                        criticality=self.criticality,
                        filter=self.filter,
                        user_metadata=self.user_metadata,
                    )
                )
            else:
                raise ValueError(f"{self.check_func.__name__!r} is not of supported type: {rule_type!r}")

        return rules

    @property
    def check_condition(self) -> Column:
        raise NotImplementedError()
