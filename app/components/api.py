"""HTTP client helper for the Streamlit app -> FastAPI backend.

``trust_env=False`` bypasses any system/corporate proxy so localhost requests
are not intercepted. All failures return a structured ``{"ok": False, ...}``
dict rather than raising, so the UI can degrade gracefully.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE = os.environ.get("MEDSIGNAL_API", "http://localhost:8000")


def api_call(endpoint: str, method: str = "GET", data: dict | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """Call the FastAPI backend and return a structured result.

    Args:
        endpoint: Path beginning with ``/`` (e.g. ``/api/query``).
        method: ``GET`` or ``POST``.
        data: JSON body for POST requests.
        timeout: Request timeout in seconds.

    Returns:
        ``{"ok": True, "data": <json>}`` on success, else
        ``{"ok": False, "error": <str>, "detail": <str>}``.
    """
    url = f"{API_BASE}{endpoint}"
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            if method.upper() == "POST":
                response = client.post(url, json=data or {})
            else:
                response = client.get(url, params=data or None)
        response.raise_for_status()
        return {"ok": True, "data": response.json()}
    except httpx.ConnectError:
        return {"ok": False, "error": "API not connected", "detail": f"Could not reach {url}. Start it with: uvicorn api.main:app --port 8000"}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "error": f"HTTP {exc.response.status_code}", "detail": str(detail)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def api_healthy() -> bool:
    """Return True when the backend health endpoint responds ok."""
    result = api_call("/", timeout=5.0)
    return bool(result.get("ok") and result.get("data", {}).get("status") == "ok")
