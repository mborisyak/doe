from __future__ import annotations

import math
from typing import Any, Dict, List, NotRequired, Optional, Required, Tuple
import sympy as sp
from typing_extensions import TypedDict

try:
    from pydantic.v1 import BaseModel, Field, root_validator, validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, root_validator, validator

REQUIRED_CONDITION_NAMES: Tuple[str, ...] = ("A", "B", "E", "temperature")
MODEL_SPEC_CONDITION_SYMBOLS: Tuple[str, ...] = ("A0", "B0", "E0", "T")


def _ensure_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


class ModelSpecValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.details = details


def _validate_key_set(
    *,
    map_name: str,
    keys: set[str],
    required_keys: Tuple[str, ...],
) -> None:
    if keys != set(required_keys):
        missing = sorted(set(required_keys) - keys)
        extra = sorted(keys - set(required_keys))
        details: List[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ValueError(f"{map_name} keys mismatch ({'; '.join(details)}).")


def _validate_range_map(
    *,
    map_name: str,
    value: Dict[str, List[float]],
    required_keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, List[float]]:
    if required_keys is not None:
        _validate_key_set(
            map_name=map_name,
            keys=set(value.keys()),
            required_keys=required_keys,
        )
        keys_to_validate = required_keys
    else:
        if not value:
            raise ValueError(f"{map_name} must contain at least one key.")
        keys_to_validate = tuple(value.keys())

    cleaned: Dict[str, List[float]] = {}
    for key in keys_to_validate:
        bounds = value[key]
        if len(bounds) != 2:
            raise ValueError(f"{map_name}.{key} must have exactly two values.")
        low = _ensure_finite(f"{map_name}.{key}[0]", float(bounds[0]))
        high = _ensure_finite(f"{map_name}.{key}[1]", float(bounds[1]))
        if high <= low:
            raise ValueError(f"{map_name}.{key} must satisfy high > low.")
        cleaned[key] = [low, high]

    return cleaned


def _validate_scalar_map(
    *,
    map_name: str,
    value: Dict[str, float],
    required_keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, float]:
    if required_keys is not None:
        _validate_key_set(
            map_name=map_name,
            keys=set(value.keys()),
            required_keys=required_keys,
        )
        keys_to_validate = required_keys
    else:
        if not value:
            raise ValueError(f"{map_name} must contain at least one key.")
        keys_to_validate = tuple(value.keys())

    cleaned: Dict[str, float] = {}
    for key in keys_to_validate:
        cleaned[key] = _ensure_finite(f"{map_name}.{key}", float(value[key]))
    return cleaned


def _validate_string_name(name: Any, *, field_name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ModelSpecValidationError(f"{field_name} must contain non-empty strings.")
    return name


def _validate_model_states(model_spec: Dict[str, Any]) -> Tuple[str, ...]:
    states = model_spec.get("states")
    if not isinstance(states, list) or not states:
        raise ModelSpecValidationError("model_spec.states must be a non-empty list.")

    cleaned_states = tuple(
        _validate_string_name(state, field_name="model_spec.states")
        for state in states
    )
    if len(set(cleaned_states)) != len(cleaned_states):
        raise ModelSpecValidationError("model_spec.states must not contain duplicates.")
    return cleaned_states


def _validate_model_section(
    model_spec: Dict[str, Any],
    *,
    section_name: str,
) -> Dict[str, str]:
    payload = model_spec.get(section_name)
    if not isinstance(payload, dict):
        raise ModelSpecValidationError(f"model_spec.{section_name} must be an object.")

    cleaned: Dict[str, str] = {}
    for raw_name, raw_expression in payload.items():
        name = _validate_string_name(raw_name, field_name=f"model_spec.{section_name}")
        if not isinstance(raw_expression, str) or not raw_expression:
            raise ModelSpecValidationError(
                f"model_spec.{section_name}.{name} must be a non-empty string."
            )
        cleaned[name] = raw_expression
    return cleaned


def _validate_model_spec_name_collisions(
    *,
    states: Tuple[str, ...],
    parameters: Dict[str, List[float]],
    algebraics: Dict[str, str],
) -> None:
    groups = {
        "states": set(states),
        "parameters": set(parameters),
        "algebraics": set(algebraics),
    }
    for name_a, name_b in (
        ("states", "parameters"),
        ("states", "algebraics"),
        ("parameters", "algebraics"),
    ):
        overlap = sorted(groups[name_a] & groups[name_b])
        if overlap:
            raise ModelSpecValidationError(
                f"Variable name conflict between '{name_a}' and '{name_b}': {overlap}"
            )


def _parse_model_spec_expression(
    *,
    expression: str,
    allowed_symbols: Dict[str, sp.Symbol],
    location: str,
) -> None:
    try:
        parsed = sp.parse_expr(expression, local_dict=allowed_symbols)
    except Exception as exc:
        raise ModelSpecValidationError(
            f"{location} is not a valid symbolic expression.",
            details={"location": location, "type": type(exc).__name__},
        ) from exc

    allowed_symbol_values = set(allowed_symbols.values())
    unknown_symbols = sorted(
        str(symbol)
        for symbol in parsed.free_symbols
        if symbol not in allowed_symbol_values
    )
    if unknown_symbols:
        raise ModelSpecValidationError(
            f"{location} references unknown symbols: {', '.join(unknown_symbols)}.",
            details={"location": location, "unknown_symbols": unknown_symbols},
        )


def _validate_model_spec_expressions(
    *,
    states: Tuple[str, ...],
    parameters: Dict[str, List[float]],
    initial_state: Dict[str, str],
    algebraics: Dict[str, str],
    rhs: Dict[str, str],
    observables: Dict[str, str],
) -> None:
    state_symbols = {name: sp.Symbol(name) for name in states}
    parameter_symbols = {name: sp.Symbol(name) for name in parameters}
    condition_symbols = {
        name: sp.Symbol(name) for name in MODEL_SPEC_CONDITION_SYMBOLS
    }

    state_names = set(states)
    if set(initial_state) != state_names:
        missing = sorted(state_names - set(initial_state))
        extra = sorted(set(initial_state) - state_names)
        details: List[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ModelSpecValidationError(
            f"model_spec.initial_state keys mismatch ({'; '.join(details)})."
        )

    if set(rhs) != state_names:
        missing = sorted(state_names - set(rhs))
        extra = sorted(set(rhs) - state_names)
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ModelSpecValidationError(
            f"model_spec.rhs keys mismatch ({'; '.join(details)})."
        )

    if set(observables) != {"A"}:
        raise ModelSpecValidationError(
            "model_spec.observables must contain exactly one key: 'A'."
        )

    for state_name, expression in initial_state.items():
        _parse_model_spec_expression(
            expression=expression,
            allowed_symbols=condition_symbols,
            location=f"model_spec.initial_state.{state_name}",
        )

    algebraic_symbols: Dict[str, sp.Symbol] = {}
    for name, expression in algebraics.items():
        _parse_model_spec_expression(
            expression=expression,
            allowed_symbols={
                **state_symbols,
                **condition_symbols,
                **parameter_symbols,
                **algebraic_symbols,
            },
            location=f"model_spec.algebraics.{name}",
        )
        algebraic_symbols[name] = sp.Symbol(name)

    for state_name, expression in rhs.items():
        _parse_model_spec_expression(
            expression=expression,
            allowed_symbols={
                **state_symbols,
                **condition_symbols,
                **parameter_symbols,
                **algebraic_symbols,
            },
            location=f"model_spec.rhs.{state_name}",
        )

    _parse_model_spec_expression(
        expression=observables["A"],
        allowed_symbols={**state_symbols, **parameter_symbols},
        location="model_spec.observables.A",
    )


def _model_spec_parameter_names(model_spec: Dict[str, Any]) -> Tuple[str, ...]:
    parameters = model_spec.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("model_spec.parameters must contain at least one key.")
    return tuple(parameters.keys())


def _validate_model_spec_input(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ModelSpecValidationError("model_spec must be a non-empty object.")

    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise ModelSpecValidationError("model_spec.parameters must be an object.")

    cleaned_parameters = _validate_range_map(
        map_name="model_spec.parameters",
        value=parameters,
    )

    states = _validate_model_states(value)
    initial_state = _validate_model_section(value, section_name="initial_state")
    algebraics = _validate_model_section(value, section_name="algebraics")
    rhs = _validate_model_section(value, section_name="rhs")
    observables = _validate_model_section(value, section_name="observables")

    _validate_model_spec_name_collisions(
        states=states,
        parameters=cleaned_parameters,
        algebraics=algebraics,
    )
    _validate_model_spec_expressions(
        states=states,
        parameters=cleaned_parameters,
        initial_state=initial_state,
        algebraics=algebraics,
        rhs=rhs,
        observables=observables,
    )

    normalized = dict(value)
    normalized["states"] = list(states)
    normalized["parameters"] = cleaned_parameters
    normalized["initial_state"] = initial_state
    normalized["algebraics"] = algebraics
    normalized["rhs"] = rhs
    normalized["observables"] = observables
    return normalized


def validate_model_spec_payload(value: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _validate_model_spec_input(value)
    except ModelSpecValidationError:
        raise
    except ValueError as exc:
        raise ModelSpecValidationError(str(exc)) from exc


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"


class ConditionPayload(TypedDict):
    A: float
    B: float
    E: float
    temperature: float


class MeasurementSeriesPayload(TypedDict):
    timestamps: List[float]
    measurements: List[float]


class OptimizerConfigPayload(TypedDict, total=False):
    iterations: int
    rtol: float
    dtype: str


class EstimateDoeParametersRequestPayload(TypedDict):
    model_spec: Required[Dict[str, Any]]
    conditions: Required[Dict[str, ConditionPayload]]
    measurements: Required[Dict[str, MeasurementSeriesPayload]]
    initial_parameters: NotRequired[Dict[str, float]]
    optimizer: NotRequired[OptimizerConfigPayload]


class HistoryPayload(TypedDict):
    conditions: Dict[str, ConditionPayload]
    timestamps: Dict[str, List[float]]
    measurements: Dict[str, List[float]]


class ProposalConfigPayload(TypedDict, total=False):
    n_proposals: int
    iterations: int
    criterion: str
    regularization: Optional[float]
    seed: int


class ProposeDoeExperimentsRequestPayload(TypedDict):
    model_spec: Required[Dict[str, Any]]
    parameters: NotRequired[Dict[str, float]]
    history: NotRequired[HistoryPayload]
    proposal_config: Required[ProposalConfigPayload]


class Condition(StrictBaseModel):
    A: float
    B: float
    E: float
    temperature: float

    @validator("A", "B", "E")
    def validate_non_negative(cls, value: float, field: Any) -> float:
        value = _ensure_finite(field.name, value)
        if value < 0.0:
            raise ValueError(f"{field.name} must be non-negative.")
        return value

    @validator("temperature")
    def validate_temperature(cls, value: float) -> float:
        return _ensure_finite("temperature", value)


class MeasurementSeries(StrictBaseModel):
    timestamps: List[float]
    measurements: List[float]

    @validator("timestamps")
    def validate_timestamps(cls, value: List[float]) -> List[float]:
        if len(value) < 2:
            raise ValueError("timestamps must contain at least two values.")

        cleaned = [_ensure_finite("timestamps", float(v)) for v in value]
        for idx in range(1, len(cleaned)):
            if cleaned[idx] <= cleaned[idx - 1]:
                raise ValueError("timestamps must be strictly increasing.")
        return cleaned

    @validator("measurements")
    def validate_measurements(cls, value: List[float]) -> List[float]:
        if len(value) < 2:
            raise ValueError("measurements must contain at least two values.")
        return [_ensure_finite("measurements", float(v)) for v in value]

    @root_validator
    def validate_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        timestamps = values.get("timestamps") or []
        measurements = values.get("measurements") or []
        if len(timestamps) != len(measurements):
            raise ValueError(
                "timestamps and measurements must have the same number of points."
            )
        return values


class OptimizerConfig(StrictBaseModel):
    iterations: int = Field(512, ge=1)
    rtol: float = Field(1.0e-6)
    dtype: str = Field("float32")

    @validator("rtol")
    def validate_rtol(cls, value: float) -> float:
        value = _ensure_finite("optimizer.rtol", value)
        if value <= 0.0:
            raise ValueError("optimizer.rtol must be positive.")
        return value

    @validator("dtype")
    def validate_dtype(cls, value: str) -> str:
        allowed = {"float32", "float64"}
        if value not in allowed:
            raise ValueError(f"optimizer.dtype must be one of {sorted(allowed)}.")
        return value


class EstimateDoeParametersRequest(StrictBaseModel):
    model_spec: Dict[str, Any]
    conditions: Dict[str, Condition]
    measurements: Dict[str, MeasurementSeries]
    initial_parameters: Optional[Dict[str, float]] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @validator("model_spec")
    def validate_model_spec(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return validate_model_spec_payload(value)

    @validator("conditions")
    def validate_conditions(cls, value: Dict[str, Condition]) -> Dict[str, Condition]:
        if not value:
            raise ValueError("conditions must contain at least one experiment.")
        return value

    @validator("measurements")
    def validate_measurements_map(
        cls, value: Dict[str, MeasurementSeries]
    ) -> Dict[str, MeasurementSeries]:
        if not value:
            raise ValueError("measurements must contain at least one experiment.")
        return value

    @validator("initial_parameters")
    def validate_initial_parameters(
        cls, value: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if value is None:
            return value
        return _validate_scalar_map(
            map_name="initial_parameters",
            value=value,
        )

    @root_validator
    def validate_label_alignment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        conditions = values.get("conditions") or {}
        measurements = values.get("measurements") or {}
        model_spec = values.get("model_spec")
        initial_parameters = values.get("initial_parameters")

        condition_labels = set(conditions.keys())
        measurement_labels = set(measurements.keys())
        if condition_labels != measurement_labels:
            missing = sorted(condition_labels - measurement_labels)
            extra = sorted(measurement_labels - condition_labels)
            details: List[str] = []
            if missing:
                details.append(f"missing measurements for labels: {', '.join(missing)}")
            if extra:
                details.append(f"measurements without conditions: {', '.join(extra)}")
            raise ValueError(
                f"conditions/measurements labels mismatch ({'; '.join(details)})."
            )

        if isinstance(model_spec, dict) and isinstance(initial_parameters, dict):
            _validate_key_set(
                map_name="initial_parameters",
                keys=set(initial_parameters.keys()),
                required_keys=_model_spec_parameter_names(model_spec),
            )
        return values


class HistoryConfig(StrictBaseModel):
    conditions: Dict[str, Condition]
    timestamps: Dict[str, List[float]]
    measurements: Dict[str, List[float]]

    @validator("timestamps")
    def validate_timestamps_map(
        cls, value: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        cleaned: Dict[str, List[float]] = {}
        for label, ts in value.items():
            if len(ts) < 2:
                raise ValueError(
                    f"history.timestamps[{label}] must contain at least two values."
                )
            cleaned_ts = [
                _ensure_finite(f"history.timestamps[{label}]", float(v)) for v in ts
            ]
            for idx in range(1, len(cleaned_ts)):
                if cleaned_ts[idx] <= cleaned_ts[idx - 1]:
                    raise ValueError(
                        f"history.timestamps[{label}] must be strictly increasing."
                    )
            cleaned[label] = cleaned_ts
        return cleaned

    @validator("measurements")
    def validate_measurements_map(
        cls, value: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        return {
            label: [_ensure_finite(f"history.measurements[{label}]", float(v)) for v in ys]
            for label, ys in value.items()
        }

    @root_validator
    def validate_alignment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        conditions = values.get("conditions") or {}
        timestamps = values.get("timestamps") or {}
        measurements = values.get("measurements") or {}

        labels = set(conditions.keys())
        if labels != set(timestamps.keys()) or labels != set(measurements.keys()):
            raise ValueError(
                "history.conditions, history.timestamps, and history.measurements "
                "must have the same set of experiment labels."
            )
        for label in labels:
            if len(timestamps[label]) != len(measurements[label]):
                raise ValueError(
                    f"history.timestamps[{label}] and history.measurements[{label}] "
                    "must have the same length."
                )
        return values


class ProposalConfig(StrictBaseModel):
    n_proposals: int = Field(..., ge=1)
    iterations: int = Field(64, ge=1)
    criterion: str = "D"
    regularization: Optional[float] = None
    seed: int

    @validator("criterion")
    def validate_criterion(cls, value: str) -> str:
        if value not in {"A", "D"}:
            raise ValueError("proposal_config.criterion must be either 'A' or 'D'.")
        return value

    @validator("regularization")
    def validate_regularization(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        return _ensure_finite("proposal_config.regularization", float(value))


class ProposeDoeExperimentsRequest(StrictBaseModel):
    model_spec: Dict[str, Any]
    parameters: Optional[Dict[str, float]] = None
    history: Optional[HistoryConfig] = None
    proposal_config: ProposalConfig

    @validator("model_spec")
    def validate_model_spec(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return validate_model_spec_payload(value)

    @validator("parameters")
    def validate_parameters(
        cls, value: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if value is None:
            return value
        return _validate_scalar_map(
            map_name="parameters",
            value=value,
        )

    @root_validator
    def validate_parameter_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        model_spec = values.get("model_spec")
        parameters = values.get("parameters")
        if isinstance(model_spec, dict) and isinstance(parameters, dict):
            _validate_key_set(
                map_name="parameters",
                keys=set(parameters.keys()),
                required_keys=_model_spec_parameter_names(model_spec),
            )
        return values


class EstimateDoeParametersResponse(StrictBaseModel):
    parameters: Dict[str, float]
    loss_trace: List[float]
    predictions: Dict[str, List[float]]

    @validator("parameters")
    def validate_parameters(cls, value: Dict[str, float]) -> Dict[str, float]:
        return _validate_scalar_map(
            map_name="parameters",
            value=value,
        )

    @validator("loss_trace")
    def validate_loss_trace(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("loss_trace cannot be empty.")
        return [_ensure_finite("loss_trace", float(v)) for v in value]

    @validator("predictions")
    def validate_predictions(
        cls, value: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        if not value:
            raise ValueError("predictions cannot be empty.")
        cleaned: Dict[str, List[float]] = {}
        for label, series in value.items():
            if not series:
                raise ValueError(f"predictions[{label}] cannot be empty.")
            cleaned[label] = [
                _ensure_finite(f"predictions[{label}]", float(v)) for v in series
            ]
        return cleaned


class ProposeDoeExperimentsResponse(StrictBaseModel):
    proposed_conditions: List[Condition]
    proposal_timestamps: List[float]
    expected: List[List[float]]

    @validator("proposed_conditions")
    def validate_proposed_conditions(cls, value: List[Condition]) -> List[Condition]:
        if not value:
            raise ValueError("proposed_conditions cannot be empty.")
        return value

    @validator("proposal_timestamps")
    def validate_proposal_timestamps(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("proposal_timestamps cannot be empty.")
        return [_ensure_finite("proposal_timestamps", float(v)) for v in value]

    @validator("expected")
    def validate_expected(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("expected cannot be empty.")
        return [
            [_ensure_finite("expected", float(v)) for v in row]
            for row in value
        ]

    @root_validator
    def validate_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        proposed = values.get("proposed_conditions") or []
        expected = values.get("expected") or []
        if len(proposed) != len(expected):
            raise ValueError(
                "proposed_conditions and expected must have equal length."
            )
        return values


# ---------------------------------------------------------------------------
# Enzyme simulation contracts
# ---------------------------------------------------------------------------

TOOL_CONTRACT_VERSION = "1.0"


def _ensure_json_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number.")
    return float(value)


class EnzymeConditionPayload(TypedDict):
    A: float
    B: float
    E: float
    temperature: float


class SimulateEnzymeDynamicsRequestPayload(TypedDict):
    conditions: Required[Dict[str, EnzymeConditionPayload]]


class EnzymeCondition(BaseModel):
    A: float = Field(..., description="Volume of substrate A solution.")
    B: float = Field(..., description="Volume of substrate B solution.")
    E: float = Field(..., description="Volume of enzyme solution.")
    temperature: float = Field(..., description="Temperature in Celsius.")

    class Config:
        extra = "forbid"

    @validator("A", "B", "E", pre=True)
    def validate_non_negative_volume(cls, value: float, field: Any) -> float:
        value = _ensure_json_number(field.name, value)
        value = _ensure_finite(field.name, value)
        if value < 0.0:
            raise ValueError(f"{field.name} must be non-negative.")
        return value

    @validator("temperature", pre=True)
    def validate_temperature(cls, value: float) -> float:
        value = _ensure_json_number("temperature", value)
        return _ensure_finite("temperature", value)


class SimulateEnzymeDynamicsRequest(BaseModel):
    conditions: Dict[str, EnzymeCondition]

    class Config:
        extra = "forbid"

    @validator("conditions")
    def validate_conditions(
        cls, value: Dict[str, EnzymeCondition]
    ) -> Dict[str, EnzymeCondition]:
        if not value:
            raise ValueError("conditions must contain at least one experiment.")
        return value


class ExperimentTrajectory(BaseModel):
    timestamps: List[float]
    A: List[float]

    @validator("timestamps")
    def validate_timestamps(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("timestamps cannot be empty.")
        return [_ensure_finite("timestamps", float(v)) for v in value]

    @validator("A")
    def validate_A(
        cls,
        value: List[float],
        values: Dict[str, Any],
    ) -> List[float]:
        expected = len(values.get("timestamps", []))
        if expected != len(value):
            raise ValueError(f"Length mismatch for 'A': expected {expected}, got {len(value)}.")
        return [_ensure_finite("A", float(v)) for v in value]


class MetadataofRun(BaseModel):
    model_identifier: str
    model_version: str
    solver: Dict[str, Any]
    units_map: Dict[str, str]
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    deterministic: bool
    seed: Optional[int]
    tool_contract_version: str = TOOL_CONTRACT_VERSION


class SimulateEnzymeDynamicsResponse(BaseModel):
    contract_version: str = TOOL_CONTRACT_VERSION
    experiments: Dict[str, ExperimentTrajectory]
    metadata: MetadataofRun
