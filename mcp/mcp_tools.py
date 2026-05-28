from __future__ import annotations

from typing import Any, Dict, Type

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, ValidationError

from mcp_contracts import (
    EstimateDoeParametersRequest,
    ProposeDoeExperimentsRequest,
    SimulateEnzymeDynamicsRequest,
)
from mcp_engine import DoeEngine
from mcp_errors import ToolExecutionError, error_response, success_response
from mcp_simulation import EnzymeCliRunner


class DoeMcpService:
    def __init__(self, engine: DoeEngine | None = None) -> None:
        self.engine = engine or DoeEngine()

    def _validate_request(
        self,
        payload: Dict[str, Any],
        model: Type[BaseModel],
    ) -> BaseModel:
        return model.parse_obj(payload)

    def fit_parameters(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed = self._validate_request(request, EstimateDoeParametersRequest)
            response = self.engine.estimate_parameters(parsed)
        except ValidationError as exc:
            return error_response(
                code="validation_error",
                message="Request payload failed schema validation.",
                details={"errors": exc.errors()},
            )
        except ToolExecutionError as exc:
            return error_response(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except Exception as exc:  # pragma: no cover
            return error_response(
                code="internal_error",
                message="Internal error while fitting parameters.",
                details={"type": type(exc).__name__},
            )

        return success_response(response)

    def propose_doe_experiments(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed = self._validate_request(request, ProposeDoeExperimentsRequest)
            response = self.engine.propose_experiments(parsed)
        except ValidationError as exc:
            return error_response(
                code="validation_error",
                message="Request payload failed schema validation.",
                details={"errors": exc.errors()},
            )
        except ToolExecutionError as exc:
            return error_response(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except Exception as exc:  # pragma: no cover
            return error_response(
                code="internal_error",
                message="Internal error while proposing experiments.",
                details={"type": type(exc).__name__},
            )

        return success_response(response)

    # GP tools: the engine runs the store-free GP scripts in a subprocess and returns the
    # plain result dict; the resolver commits any record. Requests are plain dicts.
    def _gp_call(self, fn, request: Dict[str, Any], what: str) -> Dict[str, Any]:
        try:
            return success_response(fn(request))
        except ToolExecutionError as exc:
            return error_response(code=exc.code, message=exc.message, details=exc.details)
        except Exception as exc:  # pragma: no cover
            return error_response(code="internal_error", message=f"Internal error during {what}.",
                                  details={"type": type(exc).__name__})

    def fit_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._gp_call(self.engine.fit_gp, request, "GP fit")

    def hyper_fit_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._gp_call(self.engine.hyper_fit_gp, request, "GP hyper-fit")

    def predict_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._gp_call(self.engine.predict_gp, request, "GP predict")

    def doe_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._gp_call(self.engine.doe_gp, request, "GP DoE")

    def discriminate_gp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._gp_call(self.engine.discriminate_gp, request, "discriminative GP DoE")


class EnzymeMcpService:
    def __init__(self, runner: EnzymeCliRunner | None = None) -> None:
        self.runner = runner or EnzymeCliRunner()

    def _validate_request(
        self,
        payload: Dict[str, Any],
        model: Type[BaseModel],
    ) -> BaseModel:
        return model.parse_obj(payload)

    def simulate_enzyme_dynamics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed = self._validate_request(request, SimulateEnzymeDynamicsRequest)
            response = self.runner.simulate(parsed)
        except ValidationError as exc:
            return error_response(
                code="validation_error",
                message="Request payload failed schema validation.",
                details={"errors": exc.errors()},
            )
        except ToolExecutionError as exc:
            return error_response(
                code=exc.code, message=exc.message, details=exc.details
            )
        except Exception as exc:  # pragma: no cover
            return error_response(
                code="internal_error",
                message="Internal error while running simulation.",
                details={"type": type(exc).__name__},
            )

        return success_response(response.dict())
