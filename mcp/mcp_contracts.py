from __future__ import annotations

import math
from typing import Any, Dict, List, NotRequired, Optional, Required, Tuple, TypedDict

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
    conditions: Required[Dict[str, ConditionPayload]]
    measurements: Required[Dict[str, MeasurementSeriesPayload]]
    initial_parameters: NotRequired[Dict[str, float]]
    optimizer: NotRequired[OptimizerConfigPayload]


class HistoryPayload(TypedDict):
    conditions: Dict[str, ConditionPayload]
    timestamps: Dict[str, List[float]]


class ProposalConfigPayload(TypedDict, total=False):
    n_proposals: int
    timestamps: List[float]
    iterations: int
    criterion: str
    regularization: Optional[float]
    seed: int


class ProposeDoeExperimentsRequestPayload(TypedDict):
    model_spec: Required[Dict[str, Any]]
    condition_ranges: Required[Dict[str, List[float]]]
    parameters: Required[Dict[str, float]]
    history: Required[HistoryPayload]
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
        return _validate_model_spec_input(value)

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

    @validator("conditions")
    def validate_conditions(cls, value: Dict[str, Condition]) -> Dict[str, Condition]:
        if not value:
            raise ValueError("history.conditions must contain at least one experiment.")
        return value

    @validator("timestamps")
    def validate_timestamps_map(
        cls, value: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        if not value:
            raise ValueError("history.timestamps must contain at least one experiment.")

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

    @root_validator
    def validate_alignment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        conditions = values.get("conditions") or {}
        timestamps = values.get("timestamps") or {}

        condition_labels = set(conditions.keys())
        timestamp_labels = set(timestamps.keys())
        if condition_labels != timestamp_labels:
            missing = sorted(condition_labels - timestamp_labels)
            extra = sorted(timestamp_labels - condition_labels)
            details: List[str] = []
            if missing:
                details.append(f"missing timestamps for labels: {', '.join(missing)}")
            if extra:
                details.append(f"timestamps without conditions: {', '.join(extra)}")
            raise ValueError(
                f"history.conditions/history.timestamps labels mismatch ({'; '.join(details)})."
            )
        return values


class ProposalConfig(StrictBaseModel):
    n_proposals: int = Field(..., ge=1)
    timestamps: List[float]
    iterations: int = Field(64, ge=1)
    criterion: str = "D"
    regularization: Optional[float] = None
    seed: int

    @validator("timestamps")
    def validate_timestamps(cls, value: List[float]) -> List[float]:
        if len(value) < 2:
            raise ValueError(
                "proposal_config.timestamps must contain at least two values."
            )
        cleaned = [
            _ensure_finite("proposal_config.timestamps", float(v)) for v in value
        ]
        for idx in range(1, len(cleaned)):
            if cleaned[idx] <= cleaned[idx - 1]:
                raise ValueError(
                    "proposal_config.timestamps must be strictly increasing."
                )
        return cleaned

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
    condition_ranges: Dict[str, List[float]]
    parameters: Dict[str, float]
    history: HistoryConfig
    proposal_config: ProposalConfig

    @validator("model_spec")
    def validate_model_spec(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_model_spec_input(value)

    @validator("condition_ranges")
    def validate_condition_ranges(
        cls, value: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        return _validate_range_map(
            map_name="condition_ranges",
            value=value,
            required_keys=REQUIRED_CONDITION_NAMES,
        )

    @validator("parameters")
    def validate_parameters(cls, value: Dict[str, float]) -> Dict[str, float]:
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
    encoded_proposals: List[List[float]]
    loss_trace: List[float]

    @validator("proposed_conditions")
    def validate_proposed_conditions(cls, value: List[Condition]) -> List[Condition]:
        if not value:
            raise ValueError("proposed_conditions cannot be empty.")
        return value

    @validator("encoded_proposals")
    def validate_encoded_proposals(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("encoded_proposals cannot be empty.")
        cleaned: List[List[float]] = []
        for row in value:
            if len(row) != len(REQUIRED_CONDITION_NAMES):
                raise ValueError("each encoded proposal must have four values.")
            cleaned.append([_ensure_finite("encoded_proposals", float(v)) for v in row])
        return cleaned

    @validator("loss_trace")
    def validate_loss_trace(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("loss_trace cannot be empty.")
        return [_ensure_finite("loss_trace", float(v)) for v in value]

    @root_validator
    def validate_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        proposed = values.get("proposed_conditions") or []
        encoded = values.get("encoded_proposals") or []
        if len(proposed) != len(encoded):
            raise ValueError(
                "proposed_conditions and encoded_proposals must have equal length."
            )
        return values
