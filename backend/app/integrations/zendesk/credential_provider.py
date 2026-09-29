"""Zendesk credential provider — the swappable auth seam.

This module owns the credential abstraction only, no ticket-fetching logic. It
follows the same config-flag swap convention as the pipeline modules — a typed
``ZENDESK_AUTH_MODE`` flag on Settings selects the concrete provider via
:func:`get_zendesk_credential_provider`, so callers depend on the
:class:`ZendeskCredentialProvider` interface and never on how credentials are
obtained.

Two providers exist:

* :class:`TokenCredentialProvider` — API-token (basic) auth. Zendesk's token
  scheme is HTTP Basic with username ``{email}/token`` and the API token as the
  password.
* :class:`OAuthCredentialProvider` — ``client_credentials`` OAuth. It exchanges
  a client id + secret for a short-lived bearer token at the account's
  ``/oauth/tokens`` endpoint and refreshes it proactively before expiry. This is
  the auth mode proven against the incremental ticket export.
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.core.config import Settings

# Zendesk OAuth access tokens expire after 1800 s. Refresh once a cached token
# is older than this many seconds so an in-flight request never races expiry —
# the same slack the incremental ticket pull uses.
TOKEN_LIFETIME_SLACK_SECONDS = 1500


class ZendeskCredentialError(ValueError):
    """Raised when Zendesk credential config is missing or invalid.

    Subclasses :class:`ValueError` so it reads as a configuration/validation
    failure. The message always names exactly which fields are missing — a
    misconfiguration must fail loudly at construction time, never as a silent
    ``None`` that surfaces as a confusing 401 later.
    """


class ZendeskAuthError(RuntimeError):
    """Raised when obtaining an OAuth access token from Zendesk fails.

    Distinct from :class:`ZendeskCredentialError` (a config/validation problem):
    this is a runtime failure of the token endpoint call — a bad secret, a
    non-2xx response, or a network/transport error. It is always raised, never
    swallowed, so a failed token fetch cannot degrade into a silent hang or an
    opaque crash deep in a later request.
    """


class OAuthNotImplementedError(NotImplementedError):
    """Reserved for historical compatibility.

    Earlier revisions raised this while ``ZENDESK_AUTH_MODE=oauth`` was a
    reserved-but-unbuilt seam. OAuth is now implemented by
    :class:`OAuthCredentialProvider`; the class is retained only so existing
    imports keep resolving.
    """


class ZendeskCredentialProvider(ABC):
    """Interface for supplying Zendesk request credentials.

    Callers (a future HTTP client) depend only on this contract:

    * :meth:`get_auth_header` — the ``Authorization`` header(s) to attach to a
      request, as a plain dict ready to merge into request headers.
    * :attr:`base_url` — the account's REST API base, derived from the
      subdomain, since every caller needs it regardless of auth mode.

    A new auth mechanism is a new subclass plus one factory branch; no caller
    changes.
    """

    @abstractmethod
    def get_auth_header(self) -> dict[str, str]:
        """Return the auth header(s) to merge into an outbound request."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the Zendesk REST API base URL for this account."""


def _rest_base_url(subdomain: str) -> str:
    """REST API base for an account subdomain (shared by every provider)."""
    return f"https://{subdomain}.zendesk.com/api/v2"


class TokenCredentialProvider(ZendeskCredentialProvider):
    """API-token (HTTP Basic) credential provider.

    Reads ``ZENDESK_SUBDOMAIN``, ``ZENDESK_EMAIL`` and ``ZENDESK_API_TOKEN`` from
    Settings. Zendesk's token auth is HTTP Basic where the username is
    ``{email}/token`` and the password is the API token; the resulting header is
    ``Authorization: Basic base64("{email}/token:{token}")``.

    All three fields are required. A missing/blank value raises
    :class:`ZendeskCredentialError` at construction time naming every absent
    field, so misconfiguration cannot slip through as a silent ``None``.
    """

    def __init__(self, settings: Settings) -> None:
        subdomain = (settings.ZENDESK_SUBDOMAIN or "").strip()
        email = (settings.ZENDESK_EMAIL or "").strip()
        api_token = (settings.ZENDESK_API_TOKEN or "").strip()

        missing = [
            name
            for name, value in (
                ("ZENDESK_SUBDOMAIN", subdomain),
                ("ZENDESK_EMAIL", email),
                ("ZENDESK_API_TOKEN", api_token),
            )
            if not value
        ]
        if missing:
            raise ZendeskCredentialError(
                "Zendesk token auth requires "
                + ", ".join(missing)
                + " to be set (check your .env / environment)."
            )

        self._subdomain = subdomain
        self._email = email
        self._api_token = api_token

    @property
    def base_url(self) -> str:
        return _rest_base_url(self._subdomain)

    def get_auth_header(self) -> dict[str, str]:
        raw = f"{self._email}/token:{self._api_token}"
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}


class OAuthCredentialProvider(ZendeskCredentialProvider):
    """``client_credentials`` OAuth credential provider.

    Reads ``ZENDESK_SUBDOMAIN``, ``ZENDESK_OAUTH_CLIENT_ID`` and
    ``ZENDESK_OAUTH_CLIENT_SECRET`` from Settings (``ZENDESK_OAUTH_SCOPE``
    defaults to read-only; a per-provider ``scope=`` overrides it). It
    exchanges the client id + secret for a bearer
    token at ``https://{subdomain}.zendesk.com/oauth/tokens`` and caches it,
    refreshing proactively once the token is older than
    :data:`TOKEN_LIFETIME_SLACK_SECONDS`.

    Subdomain, client id and client secret are required; a missing/blank value
    raises :class:`ZendeskCredentialError` at construction time naming every
    absent field — the same fail-loud pattern as :class:`TokenCredentialProvider`.
    A failed token fetch raises :class:`ZendeskAuthError`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        scope: str | None = None,
    ) -> None:
        """``scope`` overrides ``settings.ZENDESK_OAUTH_SCOPE`` for THIS provider.

        Lets the read path ask for a narrower token than the write path: the
        ingest adapter passes ``ZENDESK_SYNC_OAUTH_SCOPE`` (``read``), while the
        sender passes nothing and keeps ``ZENDESK_OAUTH_SCOPE`` (``read write``).
        ``None`` (the default) is the previous behaviour, unchanged.
        """
        subdomain = (settings.ZENDESK_SUBDOMAIN or "").strip()
        client_id = (settings.ZENDESK_OAUTH_CLIENT_ID or "").strip()
        client_secret = (settings.ZENDESK_OAUTH_CLIENT_SECRET or "").strip()

        missing = [
            name
            for name, value in (
                ("ZENDESK_SUBDOMAIN", subdomain),
                ("ZENDESK_OAUTH_CLIENT_ID", client_id),
                ("ZENDESK_OAUTH_CLIENT_SECRET", client_secret),
            )
            if not value
        ]
        if missing:
            raise ZendeskCredentialError(
                "Zendesk OAuth auth requires "
                + ", ".join(missing)
                + " to be set (check your .env / environment)."
            )

        self._subdomain = subdomain
        self._client_id = client_id
        self._client_secret = client_secret
        # Scope has a config default ("read"), so it is always present; still
        # guard against an explicitly-blank override falling through to Zendesk.
        requested = scope if scope is not None else settings.ZENDESK_OAUTH_SCOPE
        self._scope = (requested or "read").strip() or "read"
        self._token_url = f"https://{subdomain}.zendesk.com/oauth/tokens"

        self._client = client or httpx.Client(timeout=60)
        self._token: str | None = None
        # Monotonic timestamp of the current token; -inf means "no token yet".
        self._token_born = float("-inf")

    @property
    def base_url(self) -> str:
        return _rest_base_url(self._subdomain)

    def _token_is_fresh(self) -> bool:
        return (
            self._token is not None
            and (time.monotonic() - self._token_born) < TOKEN_LIFETIME_SLACK_SECONDS
        )

    def _refresh_token(self) -> None:
        """Fetch a new access token; raise ZendeskAuthError on any failure."""
        try:
            resp = self._client.post(
                self._token_url,
                json={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": self._scope,
                },
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
        except httpx.HTTPError as exc:
            # Transport failure or non-2xx response (raise_for_status).
            raise ZendeskAuthError(
                f"Zendesk OAuth token request failed: {exc}"
            ) from exc
        except (KeyError, ValueError) as exc:
            # 2xx with a missing 'access_token' or a non-JSON/malformed body.
            raise ZendeskAuthError(
                "Zendesk OAuth token response was missing an access_token."
            ) from exc

        if not token:
            raise ZendeskAuthError(
                "Zendesk OAuth token response contained an empty access_token."
            )
        self._token = token
        self._token_born = time.monotonic()

    def _ensure_token(self) -> None:
        if not self._token_is_fresh():
            self._refresh_token()

    def get_auth_header(self) -> dict[str, str]:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._token}"}


def get_zendesk_credential_provider(
    settings: Settings, *, scope: str | None = None
) -> ZendeskCredentialProvider:
    """Return the credential provider selected by ``ZENDESK_AUTH_MODE``.

    ``token`` → :class:`TokenCredentialProvider`; ``oauth`` →
    :class:`OAuthCredentialProvider`. Any other value raises
    :class:`ZendeskCredentialError`. Adding a future auth mode is one branch here
    plus a new subclass — callers, which hold only the interface, don't change.

    ``scope`` is passed to the OAuth provider only (``None`` → the provider falls
    back to ``ZENDESK_OAUTH_SCOPE``). ⚠️ It is IGNORED in ``token`` mode: a
    Zendesk API token carries the full permissions of its user and cannot be
    narrowed, so read-only enforcement exists in OAuth mode alone.
    """
    mode = settings.ZENDESK_AUTH_MODE
    if mode == "token":
        return TokenCredentialProvider(settings)
    if mode == "oauth":
        return OAuthCredentialProvider(settings, scope=scope)
    raise ZendeskCredentialError(f"Unsupported ZENDESK_AUTH_MODE: {mode!r}")
