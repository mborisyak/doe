from __future__ import annotations

import math
from typing import Any, Dict, List, NotRequired, Optional, Required, Tuple
from typing_extensions import TypedDict

try:
    from pydantic.v1 import BaseModel, Field, root_validator, validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, root_validator, validator

REQUIRED_CONDITION_NAMES: Tuple[str, ...] = ("A", "B", "E", "temperature")


def _ensure_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


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


def _model_spec_parameter_names(model_spec: Dict[str, Any]) -> Tuple[str, ...]:
    parameters = model_spec.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("model_spec.parameters must contain at least one key.")
    return tuple(parameters.keys())


def _validate_model_spec_input(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError("model_spec must be a non-empty object.")

    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("model_spec.parameters must be an object.")

    cleaned_parameters = _validate_range_map(
        map_name="model_spec.parameters",
        value=parameters,
    )

    normalized = dict(value)
    normalized["parameters"] = cleaned_parameters
    return normalized


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
    batches: Required[Dict[str, Any]]  # name -> batch record
    initial_parameters: NotRequired[Dict[str, float]]
    optimizer: NotRequired[OptimizerConfigPayload]


class ProposalConfigPayload(TypedDict, total=False):
    n_proposals: int
    iterations: int
    criterion: str
    regularization: Optional[float]
    seed: int


class ProposeDoeExperimentsRequestPayload(TypedDict):
    model_spec: Required[Dict[str, Any]]
    parameters: NotRequired[Dict[str, float]]
    history: NotRequired[Dict[str, Any]]  # name -> batch record
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
    # Resolved by the MCP from refs: the model spec + batch records. The script reads them
    # from temp files and writes a plain result; the MCP stamps + commits the record.
    model_spec: Dict[str, Any]
    batches: Dict[str, Any]  # batch name -> batch record
    initial_parameters: Optional[Dict[str, float]] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @validator("batches")
    def validate_batches(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("batches must contain at least one batch.")
        return value


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
    # Resolved by the MCP from the fitted_model ref (spec + parameters) and history refs.
    model_spec: Dict[str, Any]
    parameters: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, Any]] = None  # batch name -> batch record
    proposal_config: ProposalConfig


# The fit/propose scripts now own their full record body (parameters/experiments + fit +
# auxiliary); the engine passes that dict through and compute_tools stamps linkage refs, so
# there are no EstimateDoeParameters/ProposeDoeExperiments *response* contracts to validate.


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


# Sampling window (duration, measurements), solution concentrations, and noise are
# config-owned (config/enzyme.yaml), and the device comes from the ENZYME_DEVICE env var;
# none of them are agent parameters, so the request carries only the conditions to run.
class SimulateEnzymeDynamicsRequestPayload(TypedDict):
    conditions: Required[Dict[str, EnzymeConditionPayload]]
    contract_version: NotRequired[str]


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


class UnitsMap(BaseModel):
    time: str = "s"
    temperature: str = "C"
    solution_volume: str = "mL"
    concentration: str = "mM"

    class Config:
        extra = "forbid"


class SimulateEnzymeDynamicsRequest(BaseModel):
    # The agent supplies only the conditions to run; the sampling window, concentrations,
    # noise, and device are config/env-owned (read by EnzymeCliRunner), not request fields.
    contract_version: str = Field(TOOL_CONTRACT_VERSION)
    conditions: Dict[str, EnzymeCondition]

    class Config:
        extra = "forbid"

    @validator("contract_version")
    def validate_contract_version(cls, value: str) -> str:
        if value != TOOL_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract_version '{value}'. Expected '{TOOL_CONTRACT_VERSION}'."
            )
        return value

    @validator("conditions")
    def validate_conditions(
        cls, value: Dict[str, EnzymeCondition]
    ) -> Dict[str, EnzymeCondition]:
        if not value:
            raise ValueError("conditions must contain at least one experiment.")
        return value


class ExperimentTrajectory(BaseModel):
    time_points: List[float]
    state_trajectories: Dict[str, List[float]]

    @validator("time_points")
    def validate_time_points(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("time_points cannot be empty.")
        return [_ensure_finite("time_points", float(v)) for v in value]

    @validator("state_trajectories")
    def validate_state_trajectories(
        cls,
        value: Dict[str, List[float]],
        values: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        if not value:
            raise ValueError("state_trajectories cannot be empty.")

        expected = len(values.get("time_points", []))
        cleaned: Dict[str, List[float]] = {}
        for name, series in value.items():
            if len(series) != expected:
                raise ValueError(
                    f"Length mismatch for '{name}': expected {expected}, got {len(series)}."
                )
            cleaned[name] = [_ensure_finite(name, float(v)) for v in series]
        return cleaned


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
