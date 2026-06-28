"""Validator test fixtures.

`tests/unit/conftest.py` calls `unittest.mock.patch("boto3.client", ...).start()`
at module-load time so app modules can be imported without real AWS credentials.
That patch stays active for the whole session and silently breaks moto's
`@mock_aws` here (every `boto3.client(...)` call returns a MagicMock instead of
reaching moto's mock backend).

We stop the inherited patch for the duration of validator tests, then re-apply
a fresh MagicMock patch on teardown so any unit-test suite scheduled after us
still gets the mocked client behavior it expects. We target only the
`boto3.client` patch from `tests/unit/conftest.py` (looked up by its target
attribute name) rather than blindly stopping every active `mock.patch.*` in
the process — keeps validator-local mocks alive if any are ever introduced.
"""
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch


def _find_active_boto3_client_patch():
    """Locate the active `boto3.client` patch started by tests/unit/conftest.py
    by walking `mock._patch._active_patches` (the list `stopall` uses
    internally). Returns the patch instance, or None if nothing is currently
    patching boto3.client."""
    for p in list(mock._patch._active_patches):
        # patch("boto3.client", ...) stores attribute="client" + target=boto3 module
        attribute = getattr(p, "attribute", None)
        target = getattr(p, "target", None)
        target_name = getattr(target, "__name__", "")
        if attribute == "client" and target_name == "boto3":
            return p
    return None


@pytest.fixture(autouse=True, scope="session")
def _restore_boto3_client_for_validator():
    inherited = _find_active_boto3_client_patch()
    if inherited is not None:
        inherited.stop()
    try:
        yield
    finally:
        if inherited is not None:
            patch("boto3.client", return_value=MagicMock()).start()
