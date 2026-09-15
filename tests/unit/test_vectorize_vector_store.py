"""The Cloudflare Vectorize adapter, without an account.

Everything checked here is mapping: does a record become the payload Vectorize
expects, does a response become a record, does a failure become a
:class:`VectorStoreError` rather than an HTTP exception. The behavioural
contract itself can only be verified against a real index, which is the opt-in
live test.

No real credential appears anywhere in this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.infrastructure.vector_store import CloudflareVectorizeStore
from portfolio_rag.infrastructure.vector_store.vectorize import FETCH_BATCH_SIZE
from portfolio_rag.ports.errors import UnsupportedVectorStoreOperationError, VectorStoreError
from portfolio_rag.ports.vector_store import VectorQuery
from tests.contracts.vector_store import SPEC, record
from tests.factories import make_metadata
from tests.support import run

ACCOUNT = "test-account"
API_CREDENTIAL = "test-value-not-a-real-credential"
INDEX = "test-index"
INDEX_SPEC = VectorIndexSpec(embedding=SPEC)

#: What Vectorize returns for an accepted write. Mutations are asynchronous, so
#: this id is the only confirmation that a changeset was queued at all.
MUTATION = {"mutationId": "0000aaaa-11bb-22cc-33dd-444444eeeeee"}


#: The multipart field name Cloudflare's v2 upsert requires. A part under any
#: other name is rejected with HTTP 400 / code 40045, "Got a multipart request
#: without a vectors part in upsert operation".
VECTORS_PART = "vectors"


def vectors_part(request: httpx.Request) -> bytes:
    """The raw multipart part named `vectors`, headers included."""
    content_type = request.headers["Content-Type"]
    assert content_type.startswith("multipart/form-data"), content_type
    boundary = content_type.split("boundary=")[1].strip().encode()
    for part in request.content.split(b"--" + boundary):
        if b'name="%s"' % VECTORS_PART.encode() in part:
            return part
    raise AssertionError(f"no `{VECTORS_PART}` part in the multipart upload")


def ndjson_parts(request: httpx.Request) -> list[dict[str, Any]]:
    """The NDJSON records inside the multipart `vectors` file part."""
    _, _, payload = vectors_part(request).partition(b"\r\n\r\n")
    return [json.loads(line) for line in payload.strip().splitlines()]


class _Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


def build(
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> tuple[CloudflareVectorizeStore, _Recorder]:
    recorder = _Recorder(
        handler or (lambda _: httpx.Response(200, json={"success": True, "result": MUTATION}))
    )
    client = httpx.AsyncClient(
        base_url="https://api.cloudflare.com/client/v4",
        transport=httpx.MockTransport(recorder),
        headers={"Authorization": f"Bearer {API_CREDENTIAL}"},
    )
    store = CloudflareVectorizeStore(
        account_id=ACCOUNT,
        api_token=API_CREDENTIAL,
        index_name=INDEX,
        index_spec=INDEX_SPEC,
        client=client,
    )
    return store, recorder


def _vector_payload(record_id: str = "doc--0000") -> dict[str, Any]:
    original = record(record_id)
    return {
        "id": record_id,
        "values": list(original.embedding),
        "metadata": {
            "meta_chunk_id": record_id,
            "meta_document_id": "doc",
            "meta_document_title": "Test Document",
            "meta_heading_path": "Section",
            "meta_source_path": "doc.md",
            "meta_document_type": "reference",
            "meta_language": "en",
            "meta_visibility": "public",
            "meta_trust_level": "verified",
            "meta_topics": [],
            "meta_technologies": [],
            "meta_embedding_fingerprint": original.embedding_fingerprint,
            "meta_chunk_fingerprint": original.chunk_fingerprint,
            "meta_document_fingerprint": original.document_fingerprint,
            "meta_embedding_space": SPEC.identity,
        },
    }


# --- configuration ----------------------------------------------------------


def test_missing_configuration_is_refused_at_construction():
    with pytest.raises(ValueError, match="required"):
        CloudflareVectorizeStore(
            account_id="", api_token=API_CREDENTIAL, index_name=INDEX, index_spec=INDEX_SPEC
        )


def test_the_store_reports_the_index_spec_it_was_given():
    store, _ = build()

    assert store.index_spec == INDEX_SPEC


def test_vectorize_cannot_enumerate_and_says_so():
    """A capability gap in the platform, modelled rather than papered over."""
    store, _ = build()

    assert store.supports_enumeration is False
    with pytest.raises(UnsupportedVectorStoreOperationError, match="cannot enumerate"):
        run(store.list_state())


# --- request mapping --------------------------------------------------------


def test_an_upsert_uploads_ndjson_as_the_vectors_file_part():
    """v2 takes the vectors as a multipart file part named `vectors`.

    The name is the provider's, not ours. Sending the part under any other name
    — `body`, which is what Cloudflare's SDKs call the *parameter*, is the easy
    mistake — is rejected outright with HTTP 400 / code 40045, "Got a multipart
    request without a vectors part in upsert operation". Posting the NDJSON as
    a raw request body instead is worse: it is answered with HTTP 200 and then
    dropped, so the index stays empty while the caller believes it wrote. The
    part itself declares the NDJSON content type; the multipart boundary header
    belongs to the client.
    """
    store, recorder = build()

    run(store.upsert([record("doc--0000"), record("doc--0001")]))

    request = recorder.requests[0]
    assert request.url.path.endswith(f"/vectorize/v2/indexes/{INDEX}/upsert")
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    assert [line["id"] for line in ndjson_parts(request)] == ["doc--0000", "doc--0001"]


def test_the_vectors_part_is_the_only_part_and_is_typed_as_ndjson():
    """The part name, its filename and its content type, checked on the wire."""
    store, recorder = build()

    run(store.upsert([record("doc--0000")]))

    part = vectors_part(recorder.requests[0])
    headers, _, payload = part.partition(b"\r\n\r\n")
    assert b'name="vectors"' in headers
    assert b'filename="vectors.ndjson"' in headers
    assert b"Content-Type: application/x-ndjson" in headers
    # `body` was the wrong name; it must not survive anywhere in the request.
    assert b'name="body"' not in recorder.requests[0].content
    assert json.loads(payload.strip())["id"] == "doc--0000"


def test_the_vectors_part_is_newline_delimited_json_not_an_array():
    """One complete vector object per line — not a JSON array, not pretty-printed."""
    store, recorder = build()

    run(store.upsert([record(f"doc--{index:04d}") for index in range(3)]))

    _, _, payload = vectors_part(recorder.requests[0]).partition(b"\r\n\r\n")
    lines = payload.strip().splitlines()
    assert len(lines) == 3
    assert not payload.strip().startswith(b"[")
    for line in lines:
        assert isinstance(json.loads(line), dict)


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({}, id="no result at all"),
        pytest.param({"count": 2}, id="a result without a mutation id"),
        pytest.param({"mutationId": ""}, id="an empty mutation id"),
        pytest.param({"mutationId": "   "}, id="a blank mutation id"),
        pytest.param({"mutationId": None}, id="a null mutation id"),
        pytest.param({"mutationId": 12345}, id="a mutation id of the wrong type"),
    ],
)
def test_an_upsert_without_a_confirmed_mutation_is_not_a_successful_write(result: Any):
    """The failure this guard exists for.

    A wrongly encoded upsert is answered `200 {"success": true}` with no
    changeset. Without this check the store returns cleanly, the indexing
    service counts the records as written, and the index stays empty — which is
    exactly what happened against the live index.
    """
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": result}))

    with pytest.raises(VectorStoreError):
        run(store.upsert([record("doc--0000")]))


def test_a_delete_without_a_confirmed_mutation_is_not_a_successful_write():
    store, _ = build(lambda _: httpx.Response(200, json={"success": True}))

    with pytest.raises(VectorStoreError):
        run(store.delete(["doc--0000"]))


def test_a_confirmed_mutation_id_is_accepted():
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": MUTATION}))

    run(store.upsert([record("doc--0000")]))
    run(store.delete(["doc--0000"]))


def test_the_mutation_error_names_the_operation_but_carries_no_payload():
    store, _ = build(
        lambda _: httpx.Response(
            200, json={"success": True, "result": {"note": "sk-live-not-a-real-token"}}
        )
    )

    with pytest.raises(VectorStoreError) as caught:
        run(store.upsert([record("doc--0000")]))

    assert "/upsert" in str(caught.value)
    assert "sk-live" not in str(caught.value)


def test_metadata_is_flattened_with_a_namespaced_prefix():
    """Prefixing keeps this application's fields from colliding with anything
    else stored in the same index."""
    store, recorder = build()
    metadata = make_metadata(
        "doc--0000", heading_path=("Backend", "APIs"), visibility="internal", topics=("a", "b")
    )

    run(store.upsert([record("doc--0000", metadata=metadata)]))

    payload = ndjson_parts(recorder.requests[0])[0]
    assert payload["metadata"]["meta_visibility"] == "internal"
    assert payload["metadata"]["meta_heading_path"] == "Backend > APIs"
    assert payload["metadata"]["meta_topics"] == ["a", "b"]
    assert payload["metadata"]["meta_embedding_space"] == SPEC.identity


def test_the_authorization_header_carries_the_token():
    store, recorder = build()

    run(store.delete(["doc--0000"]))

    assert recorder.requests[0].headers["Authorization"] == f"Bearer {API_CREDENTIAL}"


def test_a_delete_becomes_a_list_of_ids():
    store, recorder = build()

    run(store.delete(["doc--0000", "doc--0001"]))

    request = recorder.requests[0]
    assert request.url.path.endswith("/delete_by_ids")
    assert json.loads(request.content) == {"ids": ["doc--0000", "doc--0001"]}


def test_deleting_nothing_makes_no_request():
    store, recorder = build()

    run(store.delete([]))

    assert recorder.requests == []


def test_a_fetch_becomes_a_get_by_ids_call():
    store, recorder = build(lambda _: httpx.Response(200, json={"success": True, "result": []}))

    run(store.fetch(["doc--0000"]))

    assert recorder.requests[0].url.path.endswith("/get_by_ids")
    assert json.loads(recorder.requests[0].content) == {"ids": ["doc--0000"]}


# --- /get_by_ids takes at most 20 ids ---------------------------------------


@pytest.mark.parametrize(
    ("count", "requests", "last_batch"),
    [
        pytest.param(1, 1, 1, id="one id"),
        pytest.param(19, 1, 19, id="just under the limit"),
        pytest.param(20, 1, 20, id="exactly the limit"),
        pytest.param(21, 2, 1, id="one over the limit"),
        pytest.param(40, 2, 20, id="two full batches"),
        pytest.param(149, 8, 9, id="the whole corpus"),
    ],
)
def test_a_fetch_is_split_into_batches_of_twenty(count: int, requests: int, last_batch: int):
    """Cloudflare answers 21 ids with HTTP 400 / 40007, "max id count is 20".

    Indexing the corpus reads every chunk's state in one go, so an unbatched
    fetch fails the whole run at 149 ids. Verified live before this was written.
    """
    store, recorder = build(lambda _: httpx.Response(200, json={"success": True, "result": []}))

    run(store.fetch([f"doc--{index:04d}" for index in range(count)]))

    assert len(recorder.requests) == requests
    sizes = [len(json.loads(request.content)["ids"]) for request in recorder.requests]
    assert max(sizes) <= FETCH_BATCH_SIZE
    assert sizes[-1] == last_batch
    assert sum(sizes) == count


def test_every_batch_of_a_fetch_is_merged_into_one_result():
    """Each batch answers with its own vectors; the caller sees one flat list."""
    wanted = [f"doc--{index:04d}" for index in range(45)]

    def _respond(request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.content)["ids"]
        return httpx.Response(
            200,
            json={"success": True, "result": [_vector_payload(item) for item in asked]},
        )

    store, recorder = build(_respond)

    fetched = run(store.fetch(wanted))

    assert len(recorder.requests) == 3
    assert [item.id for item in fetched] == wanted


def test_ids_the_index_does_not_hold_stay_absent_across_batches():
    """A short batch is not an error — those ids simply are not in the index."""

    def _respond(request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.content)["ids"]
        known = [item for item in asked if item.endswith("0")]
        return httpx.Response(
            200, json={"success": True, "result": [_vector_payload(item) for item in known]}
        )

    store, _ = build(_respond)

    fetched = run(store.fetch([f"doc--{index:04d}" for index in range(25)]))

    assert [item.id for item in fetched] == ["doc--0000", "doc--0010", "doc--0020"]


def test_a_failure_in_a_later_batch_fails_the_whole_fetch():
    """No partial result: one bad batch is a failed read, not a short answer."""
    seen: list[int] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        if len(seen) == 1:
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(500, json={"errors": ["upstream detail"]})

    store, _ = build(_respond)

    with pytest.raises(VectorStoreError, match="HTTP 500"):
        run(store.fetch([f"doc--{index:04d}" for index in range(30)]))

    assert len(seen) == 2


def test_fetching_states_is_batched_the_same_way():
    """fetch_states goes through fetch, so it inherits the limit."""
    store, recorder = build(lambda _: httpx.Response(200, json={"success": True, "result": []}))

    run(store.fetch_states([f"doc--{index:04d}" for index in range(149)]))

    assert len(recorder.requests) == 8
    assert all(
        len(json.loads(request.content)["ids"]) <= FETCH_BATCH_SIZE for request in recorder.requests
    )


def test_a_query_maps_top_k_and_filters():
    store, recorder = build(
        lambda _: httpx.Response(200, json={"success": True, "result": {"matches": []}})
    )

    run(
        store.query(
            VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=7, filters={"visibility": "public"})
        )
    )

    payload = json.loads(recorder.requests[0].content)
    assert recorder.requests[0].url.path.endswith("/query")
    assert payload["topK"] == 7
    assert payload["vector"] == [1.0, 0.0, 0.0]
    assert payload["filter"] == {"meta_visibility": {"$eq": "public"}}
    assert payload["returnValues"] is False


def test_a_large_upsert_is_split_into_provider_sized_batches():
    store, recorder = build()
    store._batch_size = 2

    run(store.upsert([record(f"doc--{index:04d}") for index in range(5)]))

    assert len(recorder.requests) == 3


# --- response mapping -------------------------------------------------------


def test_a_fetched_record_is_rebuilt_completely():
    payload = _vector_payload()
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": [payload]}))

    (fetched,) = run(store.fetch(["doc--0000"]))

    assert fetched.id == "doc--0000"
    assert fetched.spec == SPEC
    assert fetched.metadata.visibility.value == "public"
    assert fetched.metadata.heading_path == ("Section",)
    assert fetched.embedding == record("doc--0000").embedding


def test_fetching_states_drops_the_vectors():
    payload = _vector_payload()
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": [payload]}))

    (state,) = run(store.fetch_states(["doc--0000"]))

    assert state.id == "doc--0000"
    assert "embedding" not in state.model_dump()


def test_query_matches_are_ordered_the_same_way_the_reference_store_orders_them():
    matches = [
        {**_vector_payload("doc--0002"), "score": 0.5},
        {**_vector_payload("doc--0000"), "score": 0.9},
        {**_vector_payload("doc--0001"), "score": 0.9},
    ]
    store, _ = build(
        lambda _: httpx.Response(200, json={"success": True, "result": {"matches": matches}})
    )

    results = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=3)))

    assert [match.record.id for match in results] == ["doc--0000", "doc--0001", "doc--0002"]


# --- validation and failures ------------------------------------------------


def test_a_record_from_another_embedding_space_is_refused_before_the_request():
    store, recorder = build()
    foreign = record("doc--0000")
    foreign = foreign.model_copy(update={"spec": SPEC.model_copy(update={"model": "other"})})

    with pytest.raises(VectorStoreError, match="embedding space"):
        run(store.upsert([foreign]))

    assert recorder.requests == []


def test_a_wrongly_sized_vector_is_refused_before_the_request():
    store, recorder = build()

    with pytest.raises(VectorStoreError, match="dimensions"):
        run(store.upsert([record("doc--0000", (1.0, 0.0))]))

    assert recorder.requests == []


def test_a_wrongly_sized_query_vector_is_refused():
    store, _ = build()

    with pytest.raises(VectorStoreError, match="dimensions"):
        run(store.query(VectorQuery(embedding=(1.0, 0.0))))


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
def test_http_failures_become_store_errors(status_code: int):
    store, _ = build(lambda _: httpx.Response(status_code, json={"errors": ["upstream detail"]}))

    with pytest.raises(VectorStoreError, match=f"HTTP {status_code}"):
        run(store.delete(["doc--0000"]))


def test_a_failure_body_is_not_echoed_into_the_error():
    store, _ = build(lambda _: httpx.Response(500, json={"errors": ["sensitive upstream detail"]}))

    with pytest.raises(VectorStoreError) as caught:
        run(store.delete(["doc--0000"]))

    assert "sensitive upstream detail" not in str(caught.value)
    assert API_CREDENTIAL not in str(caught.value)


def test_a_rejected_upsert_is_an_error_that_does_not_echo_cloudflare():
    """The exact live failure: HTTP 400 / 40045 for a wrongly named part.

    It must surface as a store error naming only the operation — Cloudflare's
    own message can quote request content, so none of it is repeated.
    """
    detail = "Got a multipart request without a vectors part in upsert operation"
    store, _ = build(
        lambda _: httpx.Response(
            400, json={"success": False, "errors": [{"code": 40045, "message": detail}]}
        )
    )

    with pytest.raises(VectorStoreError) as caught:
        run(store.upsert([record("doc--0000")]))

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "/upsert" in message
    assert detail not in message
    assert API_CREDENTIAL not in message


def test_an_unsuccessful_body_is_a_failure_even_with_status_200():
    store, _ = build(lambda _: httpx.Response(200, json={"success": False, "errors": []}))

    with pytest.raises(VectorStoreError, match="reported a failure"):
        run(store.delete(["doc--0000"]))


# --- a write whose outcome is unknown ---------------------------------------


def _read_timeout(_: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("the read operation timed out")


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda store: store.upsert([record("doc--0000")]), id="upsert"),
        pytest.param(lambda store: store.delete(["doc--0000"]), id="delete"),
    ],
)
def test_a_write_that_times_out_reading_is_unknown_rather_than_failed(write: Any):
    """The live failure this exists for.

    An upsert timed out client-side after 30s and the vector was in the index
    afterwards: Vectorize commits to a durable log before it responds. Calling
    that "failed" is what makes a caller write a second time, so the outcome is
    reported as unknown and the error is explicitly not retryable.
    """
    store, _ = build(_read_timeout)

    with pytest.raises(VectorStoreError) as caught:
        run(write(store))

    message = str(caught.value)
    assert "unknown, not failed" in message
    assert "sent in full" in message
    assert "before writing them again" in message
    assert caught.value.retryable is False


def test_a_read_that_times_out_stays_an_ordinary_retryable_failure():
    """Nothing is at stake in a read, so it carries no unknown-outcome warning."""
    store, _ = build(_read_timeout)

    with pytest.raises(VectorStoreError) as caught:
        run(store.fetch(["doc--0000"]))

    assert "unknown" not in str(caught.value)
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ConnectTimeout, id="connect"),
        pytest.param(httpx.WriteTimeout, id="write"),
        pytest.param(httpx.PoolTimeout, id="pool"),
    ],
)
def test_a_write_that_never_left_stays_retryable(failure: type[Exception]):
    """Connect, write and pool timeouts mean the request never arrived whole."""

    def _fail(_: httpx.Request) -> httpx.Response:
        raise failure("never got there")

    store, _ = build(_fail)

    with pytest.raises(VectorStoreError) as caught:
        run(store.upsert([record("doc--0000")]))

    assert "unknown" not in str(caught.value)
    assert caught.value.retryable is True


def test_an_unknown_outcome_error_carries_no_payload_or_credential():
    store, _ = build(_read_timeout)

    with pytest.raises(VectorStoreError) as caught:
        run(store.upsert([record("doc--0000")]))

    message = str(caught.value)
    assert API_CREDENTIAL not in message
    assert "doc--0000" not in message
    assert "vectors.ndjson" not in message


def test_a_timeout_becomes_a_store_error():
    def _timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow")

    store, _ = build(_timeout)

    with pytest.raises(VectorStoreError, match="timed out"):
        run(store.delete(["doc--0000"]))


def test_a_network_failure_becomes_a_store_error():
    def _fail(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    store, _ = build(_fail)

    with pytest.raises(VectorStoreError, match="could not be reached"):
        run(store.delete(["doc--0000"]))


def test_a_non_json_body_becomes_a_store_error():
    store, _ = build(lambda _: httpx.Response(200, content=b"not json"))

    with pytest.raises(VectorStoreError, match="non-JSON"):
        run(store.delete(["doc--0000"]))


def test_a_record_without_metadata_is_reported_rather_than_half_built():
    store, _ = build(
        lambda _: httpx.Response(200, json={"success": True, "result": [{"id": "doc--0000"}]})
    )

    with pytest.raises(VectorStoreError, match="no metadata"):
        run(store.fetch(["doc--0000"]))


def test_metadata_this_application_does_not_recognise_is_reported():
    payload = _vector_payload()
    payload["metadata"]["meta_visibility"] = "classified"
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": [payload]}))

    with pytest.raises(VectorStoreError, match="does not recognise"):
        run(store.fetch(["doc--0000"]))


def test_a_record_without_values_is_reported():
    payload = _vector_payload()
    payload.pop("values")
    store, _ = build(lambda _: httpx.Response(200, json={"success": True, "result": [payload]}))

    with pytest.raises(VectorStoreError, match="without values"):
        run(store.fetch(["doc--0000"]))
