from __future__ import annotations

from typing import Any, Dict, Type

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, ValidationError

from mcp_contracts import (
    EstimateDoeParametersRequest,
    ProposeDoeExperimentsRequest,
)
from mcp_engine import DoeEngine
from mcp_errors import ToolExecutionError, error_response, success_response


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

        return success_response(response.dict())

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

        return success_response(response.dict())
