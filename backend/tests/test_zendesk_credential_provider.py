"""Tests for the Zendesk credential layer (auth seam).

Fully hermetic: no real Zendesk calls. Settings are lightweight stand-ins
(``SimpleNamespace``) so the suite never touches ``.env``, and the OAuth token
endpoint is exercised through an injected fake httpx client. Time is controlled
by monkeypatching ``time.monotonic`` in the module under test so the proactive
refresh window can be verified deterministically.
"""

import base64
from types import SimpleNamespace

import httpx
import pytest

from app.integrations.zendesk import credential_provider as cp
from app.integrations.zendesk import (
    OAuthCredentialProvider,
    TokenCredentialProvider,
    ZendeskAuthError,
    ZendeskCredentialError,
    get_zendesk_credential_provider,
)


# --- test doubles ----------------------------------------------------------


def make_settings(**overrides) -> SimpleNamespace:
    """Settings stub with sensible defaults; override per test."""
    base = {
        "ZENDESK_AUTH_MODE": "token",
        "ZENDESK_SUBDOMAIN": "aaai",
        "ZENDESK_EMAIL": "chair@example.org",
        "ZENDESK_API_TOKEN": "tok_123",
        "ZENDESK_OAUTH_CLIENT_ID": "confmail",
        "ZENDESK_OAUTH_CLIENT_SECRET": "s3cret",
        "ZENDESK_OAUTH_SCOPE": "read",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeResponse:
    def __init__(self, payload, *, status_code=200, raise_exc=None):
        self._payload = payload
        self.status_code = status_code
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeClient:
    """Records POSTs and returns queued/generated responses."""

    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.calls = []

    def post(self, url, json=None):
        self.calls.append({"url": url, "json": json})
        return self._response_factory(len(self.calls), json)


# === TokenCredentialProvider ==============================================


def test_token_provider_builds_basic_auth_header():
    provider = TokenCredentialProvider(make_settings())
    header = provider.get_auth_header()

    expected = base64.b64encode(b"chair@example.org/token:tok_123").decode("ascii")
    assert header == {"Authorization": f"Basic {expected}"}
    assert provider.base_url == "https://aaai.zendesk.com/api/v2"


def test_token_provider_missing_fields_raise_naming_them():
    settings = make_settings(ZENDESK_EMAIL="", ZENDESK_API_TOKEN=None)
    with pytest.raises(ZendeskCredentialError) as exc:
        TokenCredentialProvider(settings)
    msg = str(exc.value)
    assert "ZENDESK_EMAIL" in msg and "ZENDESK_API_TOKEN" in msg
    assert "ZENDESK_SUBDOMAIN" not in msg  # present, so not named


# === Factory ===============================================================


def test_factory_token_mode_returns_token_provider():
    provider = get_zendesk_credential_provider(make_settings(ZENDESK_AUTH_MODE="token"))
    assert isinstance(provider, TokenCredentialProvider)


def test_factory_oauth_mode_returns_oauth_provider():
    settings = make_settings(ZENDESK_AUTH_MODE="oauth")
    provider = get_zendesk_credential_provider(settings)
    assert isinstance(provider, OAuthCredentialProvider)


def test_factory_unsupported_mode_raises():
    with pytest.raises(ZendeskCredentialError) as exc:
        get_zendesk_credential_provider(make_settings(ZENDESK_AUTH_MODE="magic"))
    assert "magic" in str(exc.value)


# === OAuthCredentialProvider ==============================================


def _ok_factory(call_n, _json):
    # Distinct token per fetch so a refresh is observable in the header.
    return FakeResponse({"access_token": f"access_{call_n}"})


def test_oauth_missing_fields_raise_like_token_provider():
    settings = make_settings(
        ZENDESK_AUTH_MODE="oauth",
        ZENDESK_OAUTH_CLIENT_ID=None,
        ZENDESK_OAUTH_CLIENT_SECRET="   ",
    )
    with pytest.raises(ZendeskCredentialError) as exc:
        OAuthCredentialProvider(settings)
    msg = str(exc.value)
    assert "ZENDESK_OAUTH_CLIENT_ID" in msg
    assert "ZENDESK_OAUTH_CLIENT_SECRET" in msg


def test_oauth_successful_fetch_returns_bearer_header(monkeypatch):
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)
    client = FakeClient(_ok_factory)
    provider = OAuthCredentialProvider(make_settings(ZENDESK_AUTH_MODE="oauth"), client=client)

    header = provider.get_auth_header()

    assert header == {"Authorization": "Bearer access_1"}
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent["url"] == "https://aaai.zendesk.com/oauth/tokens"
    assert sent["json"] == {
        "grant_type": "client_credentials",
        "client_id": "confmail",
        "client_secret": "s3cret",
        "scope": "read",
    }


def test_oauth_token_reused_within_slack_window(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(cp.time, "monotonic", lambda: now["t"])
    client = FakeClient(_ok_factory)
    provider = OAuthCredentialProvider(make_settings(ZENDESK_AUTH_MODE="oauth"), client=client)

    first = provider.get_auth_header()
    # Advance to just inside the refresh window — no new HTTP call.
    now["t"] = cp.TOKEN_LIFETIME_SLACK_SECONDS - 1
    second = provider.get_auth_header()

    assert first == second == {"Authorization": "Bearer access_1"}
    assert len(client.calls) == 1


def test_oauth_token_refreshed_after_slack_window(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(cp.time, "monotonic", lambda: now["t"])
    client = FakeClient(_ok_factory)
    provider = OAuthCredentialProvider(make_settings(ZENDESK_AUTH_MODE="oauth"), client=client)

    first = provider.get_auth_header()
    # Advance past the slack window — a proactive refresh must fire.
    now["t"] = cp.TOKEN_LIFETIME_SLACK_SECONDS + 1
    second = provider.get_auth_header()

    assert first == {"Authorization": "Bearer access_1"}
    assert second == {"Authorization": "Bearer access_2"}
    assert len(client.calls) == 2


def test_oauth_non_2xx_raises_auth_error(monkeypatch):
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)

    def factory(call_n, _json):
        request = httpx.Request("POST", "https://aaai.zendesk.com/oauth/tokens")
        response = httpx.Response(401, request=request)
        err = httpx.HTTPStatusError("401", request=request, response=response)
        return FakeResponse(None, status_code=401, raise_exc=err)

    provider = OAuthCredentialProvider(
        make_settings(ZENDESK_AUTH_MODE="oauth"), client=FakeClient(factory)
    )
    with pytest.raises(ZendeskAuthError):
        provider.get_auth_header()


def test_oauth_network_error_raises_auth_error(monkeypatch):
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)

    class ExplodingClient:
        calls = []

        def post(self, url, json=None):
            raise httpx.ConnectError("no route to host")

    provider = OAuthCredentialProvider(
        make_settings(ZENDESK_AUTH_MODE="oauth"), client=ExplodingClient()
    )
    with pytest.raises(ZendeskAuthError):
        provider.get_auth_header()


def test_oauth_missing_access_token_in_body_raises_auth_error(monkeypatch):
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)
    client = FakeClient(lambda call_n, _json: FakeResponse({"expires_in": 1800}))
    provider = OAuthCredentialProvider(
        make_settings(ZENDESK_AUTH_MODE="oauth"), client=client
    )
    with pytest.raises(ZendeskAuthError):
        provider.get_auth_header()


# === Per-provider scope: read path vs write path ===========================
#
# The ingest adapter (reads) and the sender (writes) each build their own
# provider. The adapter must request ZENDESK_SYNC_OAUTH_SCOPE ("read") so the
# poller's token CANNOT write; the sender keeps ZENDESK_OAUTH_SCOPE ("read
# write"). Asserted on the token REQUEST each one actually sends.


def _requested_scope(provider, monkeypatch) -> str:
    monkeypatch.setattr(cp.time, "monotonic", lambda: 0.0)
    fake = FakeClient(_ok_factory)
    provider._client = fake  # swap the real httpx.Client for the recorder
    provider.get_auth_header()
    assert len(fake.calls) == 1
    return fake.calls[0]["json"]["scope"]


def test_scope_override_is_what_the_token_request_asks_for(monkeypatch):
    settings = make_settings(ZENDESK_AUTH_MODE="oauth", ZENDESK_OAUTH_SCOPE="read write")
    provider = OAuthCredentialProvider(settings, client=FakeClient(_ok_factory), scope="read")
    assert _requested_scope(provider, monkeypatch) == "read"


def test_no_override_falls_back_to_ZENDESK_OAUTH_SCOPE(monkeypatch):
    """Backward compatible: callers that pass no scope behave exactly as before."""
    settings = make_settings(ZENDESK_AUTH_MODE="oauth", ZENDESK_OAUTH_SCOPE="read write")
    provider = OAuthCredentialProvider(settings, client=FakeClient(_ok_factory))
    assert _requested_scope(provider, monkeypatch) == "read write"


def test_factory_threads_the_scope_to_the_oauth_provider(monkeypatch):
    settings = make_settings(ZENDESK_AUTH_MODE="oauth", ZENDESK_OAUTH_SCOPE="read write")
    provider = get_zendesk_credential_provider(settings, scope="read")
    assert _requested_scope(provider, monkeypatch) == "read"
    default = get_zendesk_credential_provider(settings)
    assert _requested_scope(default, monkeypatch) == "read write"


def test_factory_scope_is_ignored_in_token_mode():
    """An API token cannot be narrowed; passing a scope must not break token mode."""
    provider = get_zendesk_credential_provider(make_settings(ZENDESK_AUTH_MODE="token"), scope="read")
    assert isinstance(provider, TokenCredentialProvider)


@pytest.fixture
def oauth_app_settings(monkeypatch):
    """The REAL settings object the adapter and sender read, in OAuth mode with
    the two scopes deliberately different."""
    from app.core.config import settings

    for key, value in {
        "ZENDESK_AUTH_MODE": "oauth",
        "ZENDESK_SUBDOMAIN": "aaai",
        "ZENDESK_OAUTH_CLIENT_ID": "confmail",
        "ZENDESK_OAUTH_CLIENT_SECRET": "s3cret",
        "ZENDESK_OAUTH_SCOPE": "read write",
        "ZENDESK_SYNC_OAUTH_SCOPE": "read",
    }.items():
        monkeypatch.setattr(settings, key, value)
    return settings


def test_the_ingest_adapter_requests_a_READ_ONLY_token(oauth_app_settings, monkeypatch):
    from app.integrations.zendesk.adapter import ZendeskIngestAdapter

    provider = ZendeskIngestAdapter()._provider_obj()
    assert _requested_scope(provider, monkeypatch) == "read"


def test_the_sender_requests_a_READ_WRITE_token(oauth_app_settings, monkeypatch):
    from app.integrations.zendesk.sender import ZendeskSender

    provider = ZendeskSender()._provider_obj()
    assert _requested_scope(provider, monkeypatch) == "read write"


def test_the_sync_scope_defaults_to_read():
    from app.core.config import Settings

    assert Settings.model_fields["ZENDESK_SYNC_OAUTH_SCOPE"].default == "read"
