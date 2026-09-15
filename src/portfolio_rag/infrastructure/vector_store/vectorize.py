"""Cloudflare Vectorize, over the REST API directly.

Same reasoning as the Mistral adapter: four endpoints with stable shapes do not
justify an SDK whose models would be translated straight back into the port's
types.

**Writes are asynchronous, and only `mutationId` proves one happened.** Upsert
and delete queue a changeset rather than applying it, so a response means
"accepted", not "queryable". A write that comes back without a mutation id is
treated as a failure here — see :func:`_require_mutation`.

**A write that times out has an unknown outcome, not a failed one.** Vectorize
commits to a durable log before it answers, so a read timeout on ``/upsert`` or
``/delete_by_ids`` can leave the mutation applied while the client never saw the
confirmation. This has been observed live. Such a failure is therefore reported
as unknown and marked non-retryable, so that no caller repeats a write that may
already have landed; connect, write and pool timeouts, which mean the request
never arrived whole, stay retryable.

**One honest limitation.** Vectorize can fetch vectors by id, but it cannot
enumerate an index. There is no "list everything" endpoint, so
:attr:`supports_enumeration` is ``False`` and :meth:`list_state` raises. That is
why the port models enumeration as a capability instead of assuming it: index
planning against this store can decide what to create and update, but cannot
discover records whose chunks have disappeared from the corpus. The CLI says so
rather than quietly skipping stale deletion. Running with ``--rebuild`` against
a fresh index is the deliberate way to clear one out.

**No provisioning.** This adapter uses an index that already exists. Creating
one is an account mutation with billing implications, and nothing in an
application's runtime path should be able to do that by accident.

Credentials come from settings, are sent only in the ``Authorization`` header,
and never appear in an error message or a log line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx2 as httpx

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorIndexSpec,
    VectorMetadata,
    VectorRecord,
    VectorRecordState,
)
from portfolio_rag.domain.knowledge import DocumentType, TrustLevel, Visibility
from portfolio_rag.ports.errors import UnsupportedVectorStoreOperationError, VectorStoreError
from portfolio_rag.ports.vector_store import VectorMatch, VectorQuery

STORE_NAME: Final = "cloudflare-vectorize"
DEFAULT_BASE_URL: Final = "https://api.cloudflare.com/client/v4"

#: Vectorize accepts many vectors per upsert; batching stays here because the
#: limit is the provider's, not the application's.
DEFAULT_BATCH_SIZE: Final = 500

#: `/get_by_ids` takes far fewer ids than `/upsert` takes vectors, and the two
#: limits are unrelated. Verified live against the API, not inferred: 20 ids
#: answer HTTP 200, 21 answer HTTP 400 with code 40007, "too many ids in
#: payload; max id count is 20". Cloudflare's API reference declares no
#: `maxItems` for this field, so the number cannot be read out of the schema —
#: it has to be pinned here. A run that reads 149 chunk states fails outright
#: without this.
FETCH_BATCH_SIZE: Final = 20

_METADATA_PREFIX: Final = "meta_"
_HEADING_SEPARATOR: Final = " > "


class CloudflareVectorizeStore:
    """A :class:`VectorStore` backed by a pre-existing Vectorize index."""

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        index_name: str,
        index_spec: VectorIndexSpec,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not account_id or not api_token or not index_name:
            raise ValueError("account id, API token and index name are all required")

        self._index_spec = index_spec
        self._batch_size = batch_size
        self._prefix = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
            },
        )

    @property
    def index_spec(self) -> VectorIndexSpec:
        return self._index_spec

    @property
    def supports_enumeration(self) -> bool:
        """False: Vectorize has no endpoint that lists an index's contents."""
        return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self._require_compatible(record)

        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            # NDJSON — one vector object per line — uploaded as a *file part*
            # named `vectors`. The v2 API is multipart/form-data here, not a raw
            # body: `files=dict(vectors=embeddings)` in Cloudflare's own HTTP
            # example. The name is not ours to choose — a part named anything
            # else is rejected with HTTP 400 / code 40045, "Got a multipart
            # request without a vectors part in upsert operation". (`body` is
            # what Cloudflare's *SDKs* call the parameter; it is not the wire
            # field name.) The part carries the NDJSON content type; the
            # request's own boundary header is the client's to set.
            ndjson = "\n".join(json.dumps(_to_payload(record)) for record in batch)
            result = await self._request(
                "POST",
                "/upsert",
                files={
                    "vectors": ("vectors.ndjson", ndjson.encode("utf-8"), "application/x-ndjson")
                },
                mutating=True,
            )
            _require_mutation(result, "/upsert")

    async def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        if not ids:
            return []
        # Split by the provider's id limit, not by `self._batch_size` — that one
        # is the write limit and is 25x larger. Batches are concatenated in the
        # order they were asked for; ids the index does not hold are simply
        # absent from the result, exactly as with a single call.
        records: list[VectorRecord] = []
        for start in range(0, len(ids), FETCH_BATCH_SIZE):
            batch = list(ids[start : start + FETCH_BATCH_SIZE])
            result = await self._request("POST", "/get_by_ids", json_body={"ids": batch})
            records.extend(
                _from_payload(entry, self._index_spec.embedding) for entry in _as_list(result)
            )
        return records

    async def fetch_states(self, ids: Sequence[str]) -> list[VectorRecordState]:
        return [record.state() for record in await self.fetch(ids)]

    async def list_state(self) -> list[VectorRecordState]:
        raise UnsupportedVectorStoreOperationError(
            "Cloudflare Vectorize cannot enumerate an index; records can only be "
            "fetched by id. Stale records therefore cannot be discovered against "
            "this store — use `--rebuild` against a fresh index to clear one out."
        )

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        for start in range(0, len(ids), self._batch_size):
            batch = list(ids[start : start + self._batch_size])
            result = await self._request(
                "POST", "/delete_by_ids", json_body={"ids": batch}, mutating=True
            )
            _require_mutation(result, "/delete_by_ids")

    async def query(self, query: VectorQuery) -> list[VectorMatch]:
        if len(query.embedding) != self._index_spec.dimensions:
            raise VectorStoreError(
                f"Query vector has {len(query.embedding)} dimensions; "
                f"this index holds {self._index_spec.dimensions}."
            )

        payload: dict[str, Any] = {
            "vector": list(query.embedding),
            "topK": query.top_k,
            "returnMetadata": "all",
            "returnValues": False,
        }
        if query.filters:
            payload["filter"] = {
                f"{_METADATA_PREFIX}{field}": {"$eq": value}
                for field, value in query.filters.items()
            }

        result = await self._request("POST", "/query", json_body=payload)
        matches = result.get("matches") if isinstance(result, dict) else None
        if not isinstance(matches, list):
            raise VectorStoreError("Vectorize query response is missing its `matches` array.")

        scored = [
            (
                float(entry.get("score", 0.0)),
                _state_from_payload(entry, self._index_spec.embedding),
            )
            for entry in matches
            if isinstance(entry, dict)
        ]
        # The same stable ordering the in-memory store guarantees, so results do
        # not depend on which store answered.
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [VectorMatch(record=state, score=score) for score, state in scored]

    # --- plumbing -----------------------------------------------------------

    def _require_compatible(self, record: VectorRecord) -> None:
        if not self._index_spec.embedding.is_compatible_with(record.spec):
            raise VectorStoreError(
                f"Record `{record.id}` was produced in embedding space "
                f"`{record.spec.identity}`, but this index holds "
                f"`{self._index_spec.embedding.identity}`."
            )
        if len(record.embedding) != self._index_spec.dimensions:
            raise VectorStoreError(
                f"Record `{record.id}` has {len(record.embedding)} dimensions; "
                f"this index holds {self._index_spec.dimensions}."
            )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
        headers: Mapping[str, str] | None = None,
        mutating: bool = False,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                f"{self._prefix}{path}",
                json=json_body,
                content=content,
                files=dict(files) if files else None,
                headers=dict(headers) if headers else None,
            )
        except httpx.ReadTimeout as exc:
            # The request went out in full; Vectorize just did not answer in
            # time. For a write that is *not* the same as "nothing happened":
            # Vectorize commits to a durable log before it responds, so the
            # mutation may already have been accepted. Observed live — an upsert
            # timed out here and the vector was in the index afterwards. Calling
            # that a failure is what invites a retry that writes twice, so it is
            # reported as an unknown outcome and is never retryable.
            if mutating:
                raise VectorStoreError(
                    f"Vectorize did not answer `{path}` in time, after the request had "
                    "been sent in full. The write may already have been accepted: this "
                    "outcome is unknown, not failed. Read the affected ids back, or "
                    "check the index's vector count, before writing them again.",
                    retryable=False,
                ) from exc
            raise VectorStoreError(
                f"Vectorize did not answer `{path}` in time.", retryable=True
            ) from exc
        except httpx.TimeoutException as exc:
            # Connect, write and pool timeouts all mean the request never
            # arrived in full, so nothing can have been applied to the index.
            raise VectorStoreError(
                f"Vectorize request to `{path}` timed out before it was sent.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise VectorStoreError(f"Vectorize could not be reached for `{path}`.") from exc

        if response.status_code >= 400:
            # Cloudflare error bodies can echo request content; only the status
            # and the operation are reported.
            raise VectorStoreError(f"Vectorize returned HTTP {response.status_code} for `{path}`.")

        try:
            body = response.json()
        except ValueError as exc:
            raise VectorStoreError(f"Vectorize returned a non-JSON body for `{path}`.") from exc

        if not isinstance(body, dict):
            raise VectorStoreError(f"Vectorize returned a non-object body for `{path}`.")
        if body.get("success") is False:
            raise VectorStoreError(f"Vectorize reported a failure for `{path}`.")
        return body.get("result")


def _require_mutation(result: Any, path: str) -> None:
    """A write is only done when Vectorize says it queued a changeset.

    Mutations are asynchronous: the response does not mean the vectors are
    queryable yet, it means Cloudflare accepted them. `mutationId` is the only
    evidence of that. Without this check a 200 carrying no changeset — which is
    exactly what a wrongly encoded upsert produces — is indistinguishable from a
    successful write, and the caller reports records written to an index that
    never received any.
    """
    mutation = result.get("mutationId") if isinstance(result, Mapping) else None
    if not isinstance(mutation, str) or not mutation.strip():
        raise VectorStoreError(
            f"Vectorize accepted `{path}` without reporting a mutation id, "
            "so the write cannot be confirmed."
        )


def _as_list(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [entry for entry in result if isinstance(entry, dict)]
    if isinstance(result, dict):
        for key in ("vectors", "matches"):
            nested = result.get(key)
            if isinstance(nested, list):
                return [entry for entry in nested if isinstance(entry, dict)]
    raise VectorStoreError("Vectorize returned an unexpected result shape.")


def _to_payload(record: VectorRecord) -> dict[str, Any]:
    """Flatten a record into Vectorize's vector object.

    Metadata keys are prefixed so that the fields this application filters on
    can never collide with anything else stored in the same index.
    """
    metadata: dict[str, Any] = {
        f"{_METADATA_PREFIX}chunk_id": record.metadata.chunk_id,
        f"{_METADATA_PREFIX}document_id": record.metadata.document_id,
        f"{_METADATA_PREFIX}document_title": record.metadata.document_title,
        f"{_METADATA_PREFIX}heading_path": _HEADING_SEPARATOR.join(record.metadata.heading_path),
        f"{_METADATA_PREFIX}source_path": record.metadata.source_path,
        f"{_METADATA_PREFIX}document_type": record.metadata.document_type.value,
        f"{_METADATA_PREFIX}language": record.metadata.language,
        f"{_METADATA_PREFIX}visibility": record.metadata.visibility.value,
        f"{_METADATA_PREFIX}trust_level": record.metadata.trust_level.value,
        f"{_METADATA_PREFIX}topics": list(record.metadata.topics),
        f"{_METADATA_PREFIX}technologies": list(record.metadata.technologies),
        f"{_METADATA_PREFIX}embedding_fingerprint": record.embedding_fingerprint,
        f"{_METADATA_PREFIX}chunk_fingerprint": record.chunk_fingerprint,
        f"{_METADATA_PREFIX}document_fingerprint": record.document_fingerprint,
        f"{_METADATA_PREFIX}embedding_space": record.spec.identity,
    }
    return {"id": record.id, "values": list(record.embedding), "metadata": metadata}


def _metadata_from_payload(entry: Mapping[str, Any]) -> tuple[VectorMetadata, dict[str, str]]:
    raw = entry.get("metadata")
    if not isinstance(raw, dict):
        raise VectorStoreError(f"Vectorize record `{entry.get('id')}` has no metadata.")

    def string_value(name: str, default: str = "") -> str:
        found = raw.get(f"{_METADATA_PREFIX}{name}")
        return found if isinstance(found, str) else default

    def string_values(name: str) -> tuple[str, ...]:
        found = raw.get(f"{_METADATA_PREFIX}{name}")
        if isinstance(found, list):
            return tuple(str(item) for item in found)
        return ()

    heading = string_value("heading_path")
    try:
        metadata = _build_metadata(
            string_value=string_value,
            string_values=string_values,
            heading=heading,
        )
    except ValueError as exc:
        raise VectorStoreError(
            f"Vectorize record `{entry.get('id')}` carries metadata this application "
            "does not recognise."
        ) from exc
    fingerprints = {
        name: string_value(name)
        for name in ("embedding_fingerprint", "chunk_fingerprint", "document_fingerprint")
    }
    return metadata, fingerprints


def _build_metadata(
    *,
    string_value: Any,
    string_values: Any,
    heading: str,
) -> VectorMetadata:
    return VectorMetadata(
        chunk_id=string_value("chunk_id"),
        document_id=string_value("document_id"),
        document_title=string_value("document_title"),
        heading_path=tuple(heading.split(_HEADING_SEPARATOR)) if heading else (),
        source_path=string_value("source_path"),
        document_type=DocumentType(string_value("document_type")),
        language=string_value("language"),
        visibility=Visibility(string_value("visibility")),
        trust_level=TrustLevel(string_value("trust_level")),
        topics=string_values("topics"),
        technologies=string_values("technologies"),
    )


def _state_from_payload(entry: Mapping[str, Any], spec: EmbeddingSpec) -> VectorRecordState:
    metadata, fingerprints = _metadata_from_payload(entry)
    identifier = entry.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise VectorStoreError("Vectorize returned a record without an id.")
    return VectorRecordState(id=identifier, spec=spec, metadata=metadata, **fingerprints)


def _from_payload(entry: Mapping[str, Any], spec: EmbeddingSpec) -> VectorRecord:
    state = _state_from_payload(entry, spec)
    raw_values = entry.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise VectorStoreError(f"Vectorize record `{state.id}` came back without values.")
    return VectorRecord(
        id=state.id,
        spec=state.spec,
        embedding_fingerprint=state.embedding_fingerprint,
        chunk_fingerprint=state.chunk_fingerprint,
        document_fingerprint=state.document_fingerprint,
        metadata=state.metadata,
        embedding=tuple(float(value) for value in raw_values),
    )
