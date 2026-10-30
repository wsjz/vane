# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import builtins
import datetime as dt
import os
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink as datasink
import vane.datasink.clickhouse as clickhouse
from vane import ClickHouseSink, EnvironmentSecret
from vane.datasink import BoundDataSink, DataSinkExecutionOptions, DataSinkWriteError, WriteContext, WriteOutcome

_REAL_LOAD_CLICKHOUSE_CONNECT = clickhouse._load_clickhouse_connect
_REAL_OPEN_CLIENT = clickhouse._open_client
_QUERY_ID = re.compile(r"vane_[0-9a-f]{16}_[0-9a-f]{32}_[0-9a-f]+\Z")


class _ClientError(Exception):
    pass


@dataclass(frozen=True)
class _Summary:
    summary: dict[str, str]


@dataclass(frozen=True)
class _Insert:
    table: str
    column_names: tuple[str, ...]
    body: Any
    settings: dict[str, Any]
    fmt: str

    def read_table(self) -> pa.Table:
        return pa.ipc.open_stream(pa.py_buffer(self.body)).read_all()


_ResponseFactory = Callable[[_Insert], _Summary]


def _success(insert: _Insert) -> _Summary:
    return _Summary({"query_id": insert.settings["query_id"], "written_rows": str(insert.read_table().num_rows)})


class _Client:
    responses: list[_Summary | _ResponseFactory | BaseException] = []
    instances: list[_Client] = []
    close_error: BaseException | None = None

    def __init__(self, endpoint: str, user: str, password: str, timeout: int) -> None:
        self.endpoint = endpoint
        self.user = user
        self.password = password
        self.timeout = timeout
        self.inserts: list[_Insert] = []
        self.close_calls = 0

    def raw_insert(
        self,
        table: str,
        column_names: tuple[str, ...],
        insert_block: Any,
        settings: dict[str, Any],
        fmt: str,
    ) -> _Summary:
        insert = _Insert(table, tuple(column_names), insert_block, dict(settings), fmt)
        self.inserts.append(insert)
        if not _Client.responses:
            raise AssertionError("fake ClickHouse response queue is empty")
        response = _Client.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response(insert) if callable(response) else response

    def close(self) -> None:
        self.close_calls += 1
        if _Client.close_error is not None:
            raise _Client.close_error


def _open_fake_client(endpoint: str, user: str, password: str, timeout: int) -> _Client:
    client = _Client(endpoint, user, password, timeout)
    _Client.instances.append(client)
    return client


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Client.responses = []
    _Client.instances = []
    _Client.close_error = None
    monkeypatch.setattr(clickhouse, "_open_client", _open_fake_client)


def _schema(*, vector_type: pa.DataType | None = None) -> pa.Schema:
    return pa.schema(
        [
            ("source_id", pa.int64()),
            ("embedding", pa.list_(pa.float32(), 3) if vector_type is None else vector_type),
            ("source_title", pa.string()),
        ]
    )


def _destination_schema(
    *,
    id_type: pa.DataType | None = None,
    title_type: pa.DataType | None = None,
    title_nullable: bool = False,
) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.uint32() if id_type is None else id_type, nullable=False),
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
            pa.field("title", pa.string() if title_type is None else title_type, nullable=title_nullable),
        ]
    )


def _table(*, vectors: list[list[float] | None] | None = None, schema: pa.Schema | None = None) -> pa.Table:
    values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] if vectors is None else vectors
    return pa.table(
        {"source_id": [1, 2], "embedding": values, "source_title": ["one", "two"]},
        schema=_schema() if schema is None else schema,
    )


def _sink(**overrides: object) -> ClickHouseSink:
    options: dict[str, object] = {
        "endpoint": "http://clickhouse.example:8123",
        "destination_schema": _destination_schema(),
        "field_mapping": {"source_id": "id", "source_title": "title"},
        "vector_dimensions": {"embedding": 3},
        "timeout": 5,
    }
    options.update(overrides)
    return ClickHouseSink("analytics", "items", **options)  # type: ignore[arg-type]


def _bound(*, schema: pa.Schema | None = None, **overrides: object) -> BoundDataSink:
    return _sink(**overrides).bind(_schema() if schema is None else schema)


def _worker(*, schema: pa.Schema | None = None, **overrides: object) -> Any:
    return _bound(schema=schema, **overrides).open_worker(WriteContext("clickhouse-test-operation"))


def test_clickhouse_sink_is_public_without_importing_clickhouse_connect() -> None:
    assert vane.ClickHouseSink is ClickHouseSink
    assert clickhouse.__all__ == ["ClickHouseSink"]


def test_clickhouse_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_client(name: str, *args: object, **kwargs: object) -> Any:
        if name == "clickhouse_connect":
            raise ModuleNotFoundError("No module named 'clickhouse_connect'", name="clickhouse_connect")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_client)
    with pytest.raises(ImportError, match=r"vane-ai\[clickhouse\]"):
        _REAL_LOAD_CLICKHOUSE_CONNECT()


def test_clickhouse_sink_opens_a_client_without_hidden_retries_or_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _Module:
        @staticmethod
        def get_client(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "client"

    monkeypatch.setattr(clickhouse, "_load_clickhouse_connect", lambda: _Module)

    assert _REAL_OPEN_CLIENT("https://ch.example", "writer", "secret", 42) == "client"
    assert _REAL_OPEN_CLIENT("http://ch.example:9123", "writer", "", 7) == "client"
    assert calls[0] == {
        "interface": "https",
        "host": "ch.example",
        "port": 8443,
        "username": "writer",
        "password": "secret",
        "compress": False,
        "query_retries": 0,
        "connect_timeout": 30,
        "send_receive_timeout": 42,
        "autogenerate_session_id": False,
        "autogenerate_query_id": False,
    }
    assert (calls[1]["interface"], calls[1]["port"], calls[1]["send_receive_timeout"]) == ("http", 9123, 7)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"endpoint": "ftp://clickhouse.example"}, ValueError, "HTTP or HTTPS"),
        ({"endpoint": "http://user:pw@clickhouse.example"}, ValueError, "must not contain credentials"),
        ({"endpoint": "http://clickhouse.example/?database=x"}, ValueError, "query parameters"),
        ({"endpoint": "http://clickhouse.example/proxy"}, ValueError, "must not contain a path"),
        ({"password": "plain-text"}, TypeError, "EnvironmentSecret"),
        ({"user": " default"}, ValueError, "surrounding whitespace"),
        ({"worker_count": 0}, ValueError, "positive"),
        ({"max_request_bytes": True}, TypeError, "positive integer"),
        ({"vector_dimensions": {"embedding": 0}}, ValueError, "positive"),
        ({"field_mapping": {"source_id": "id\n"}}, ValueError, "NUL, CR, or LF"),
        ({"destination_schema": pa.schema([])}, ValueError, "at least one field"),
        (
            {"destination_schema": pa.schema([("id", pa.int32()), ("id", pa.int64())])},
            ValueError,
            "unique",
        ),
        ({"destination_schema": pa.schema([("price", pa.decimal128(10, 2))])}, ValueError, "unsupported Arrow"),
        ({"destination_schema": pa.schema([("seen_at", pa.timestamp("us"))])}, ValueError, "without a time zone"),
        (
            {"destination_schema": pa.schema([("seen_at", pa.list_(pa.timestamp("ms")))])},
            ValueError,
            r"seen_at\[\]",
        ),
    ],
)
def test_clickhouse_sink_validates_constructor(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _sink(**overrides)


@pytest.mark.parametrize("name", ["", "analytics.items", "1items", "items`", "ítems", "a" * 257])
def test_clickhouse_sink_requires_plain_database_and_table_names(name: str) -> None:
    with pytest.raises(ValueError):
        ClickHouseSink(name, "items", endpoint="http://ch.example", destination_schema=_destination_schema())
    with pytest.raises(ValueError):
        ClickHouseSink("analytics", name, endpoint="http://ch.example", destination_schema=_destination_schema())


def test_clickhouse_sink_binds_mapping_vectors_and_execution_options() -> None:
    sink = _sink(worker_count=3, max_batch_rows=10, max_batch_bytes=1024)
    bound = sink.bind(_schema())

    assert bound.execution_options == DataSinkExecutionOptions(
        worker_count=3, batch_size=10, target_max_batch_bytes=1024, max_retries=0
    )
    assert sink.destination_schema == _destination_schema()
    cloudpickle.loads(cloudpickle.dumps(bound))


def test_clickhouse_sink_rejects_invalid_bound_schema_and_mappings() -> None:
    with pytest.raises(ValueError, match="unknown input columns"):
        _sink(field_mapping={"missing": "id"}).bind(_schema())
    with pytest.raises(ValueError, match="unknown input columns"):
        _sink(vector_dimensions={"missing": 3}).bind(_schema())
    with pytest.raises(ValueError, match="names and order"):
        _sink(field_mapping={"source_id": "id"}).bind(_schema())
    with pytest.raises(ValueError, match="unique ClickHouse column names"):
        _sink(field_mapping={"source_id": "id", "source_title": "id"}).bind(_schema())
    with pytest.raises(ValueError, match="fixed dimension 3"):
        _sink(vector_dimensions={"embedding": 4}).bind(_schema())
    with pytest.raises(ValueError, match="list, large_list, or fixed_size_list of float32"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float64())))
    with pytest.raises(ValueError, match="unique input column names"):
        _sink().bind(pa.schema([("a", pa.int32()), ("a", pa.int32())]))


@pytest.mark.parametrize(
    "source_type",
    [
        pa.timestamp("us"),
        pa.list_(pa.timestamp("ns")),
    ],
)
def test_clickhouse_sink_rejects_naive_timestamp_sources(source_type: pa.DataType) -> None:
    sink = ClickHouseSink(
        "analytics",
        "events",
        endpoint="http://ch.example",
        destination_schema=pa.schema([("seen_at", pa.timestamp("us", tz="UTC"))]),
    )
    with pytest.raises(ValueError, match="without a time zone"):
        sink.bind(pa.schema([("seen_at", source_type)]))


@pytest.mark.parametrize(
    ("source_type", "destination_type"),
    [
        (pa.timestamp("s", tz="UTC"), pa.date32()),
        (pa.int64(), pa.timestamp("us", tz="UTC")),
        (pa.string(), pa.timestamp("us", tz="UTC")),
        (pa.timestamp("us", tz="UTC"), pa.int64()),
        (pa.date32(), pa.int32()),
        (pa.list_(pa.timestamp("ms", tz="UTC")), pa.list_(pa.date32())),
    ],
)
def test_clickhouse_sink_writes_temporal_values_only_from_the_same_kind(
    source_type: pa.DataType, destination_type: pa.DataType
) -> None:
    sink = ClickHouseSink(
        "analytics", "events", endpoint="http://ch.example", destination_schema=pa.schema([("value", destination_type)])
    )
    with pytest.raises(ValueError, match="convert the value explicitly"):
        sink.bind(pa.schema([("value", source_type)]))


@pytest.mark.parametrize(
    "source_type",
    [
        pa.date64(),
        pa.dictionary(pa.int32(), pa.string()),
        pa.string_view(),
        pa.struct([("a", pa.int32())]),
        pa.decimal128(10, 2),
    ],
)
def test_clickhouse_sink_rejects_encoded_and_non_plain_sources_before_opening_a_client(
    source_type: pa.DataType,
) -> None:
    sink = ClickHouseSink(
        "analytics", "items", endpoint="http://ch.example", destination_schema=pa.schema([("title", pa.string())])
    )
    with pytest.raises(ValueError, match="unsupported Arrow type"):
        sink.bind(pa.schema([("title", source_type)]))
    assert _Client.instances == []


def test_clickhouse_sink_inserts_an_arrow_stream_with_strict_settings() -> None:
    _Client.responses = [_success, _success]
    worker = _worker()
    try:
        first = worker.write(_table())
        second = worker.write(_table())
    finally:
        worker.close()

    client = _Client.instances[0]
    assert (client.endpoint, client.user, client.password, client.timeout) == (
        "http://clickhouse.example:8123",
        "default",
        "",
        5,
    )
    assert client.close_calls == 1
    query_ids = [insert.settings["query_id"] for insert in client.inserts]
    assert len(set(query_ids)) == 2
    for insert, query_id in zip(client.inserts, query_ids, strict=True):
        assert _QUERY_ID.fullmatch(query_id)
        assert insert.table == "`analytics`.`items`"
        assert insert.column_names == ("`id`", "`embedding`", "`title`")
        assert insert.fmt == "ArrowStream"
        assert isinstance(insert.body, memoryview)
        assert insert.settings == {
            "async_insert": 0,
            "input_format_null_as_default": 0,
            "insert_deduplication_token": query_id,
            "query_id": query_id,
        }
        expected = pa.table(
            {"id": [1, 2], "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "title": ["one", "two"]},
            schema=_destination_schema(),
        )
        assert insert.read_table().equals(expected)

    assert first.rows_received == first.rows_affected == 2
    assert first.metadata["query_id"] == query_ids[0]
    assert first.metadata["provider"] == "clickhouse"
    assert first.metadata["wire_bytes"] == len(client.inserts[0].body)
    assert len(first.warnings) == 1 and "independently" in first.warnings[0]
    assert second.warnings == ()


def test_clickhouse_sink_query_ids_differ_between_workers_of_one_operation() -> None:
    _Client.responses = [_success, _success]
    bound = _bound()
    for _ in range(2):
        worker = bound.open_worker(WriteContext("shared-operation"))
        worker.write(_table())
        worker.close()

    first, second = (client.inserts[0].settings["query_id"] for client in _Client.instances)
    assert first != second
    assert first.split("_")[1] == second.split("_")[1]


def test_clickhouse_sink_quotes_column_names() -> None:
    _Client.responses = [_success]
    destination = pa.schema([pa.field("we`ird\\name", pa.int64()), pa.field("值", pa.string())])
    sink = ClickHouseSink("analytics", "items", endpoint="http://ch.example", destination_schema=destination)
    worker = sink.bind(destination).open_worker(WriteContext("quoting"))
    worker.write(pa.table({"we`ird\\name": [1], "值": ["v"]}))

    insert = _Client.instances[0].inserts[0]
    assert insert.column_names == ("`we\\`ird\\\\name`", "`值`")
    assert insert.read_table().column_names == ["we`ird\\name", "值"]


def test_clickhouse_sink_converts_unsigned_dates_and_zoned_timestamps() -> None:
    _Client.responses = [_success]
    instant = dt.datetime(2026, 9, 5, 1, 2, 3, 456789, tzinfo=dt.timezone.utc)
    source = pa.schema(
        [
            ("id", pa.int64()),
            ("day", pa.date32()),
            ("seen_at", pa.timestamp("ms", tz="Asia/Shanghai")),
        ]
    )
    destination = pa.schema(
        [
            pa.field("id", pa.uint64(), nullable=False),
            pa.field("day", pa.date32(), nullable=False),
            pa.field("seen_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    sink = ClickHouseSink("analytics", "events", endpoint="http://ch.example", destination_schema=destination)
    worker = sink.bind(source).open_worker(WriteContext("temporal"))
    worker.write(
        pa.table(
            {"id": [7], "day": [dt.date(2026, 9, 5)], "seen_at": [instant.replace(microsecond=456000)]},
            schema=source,
        )
    )

    row = _Client.instances[0].inserts[0].read_table().to_pylist()[0]
    assert row == {"id": 7, "day": dt.date(2026, 9, 5), "seen_at": instant.replace(microsecond=456000)}


@pytest.mark.parametrize(
    ("destination", "source", "values", "message"),
    [
        (pa.uint32(), pa.int64(), [-1], "cannot be safely cast"),
        (pa.int8(), pa.int64(), [300], "cannot be safely cast"),
        (pa.float32(), pa.float64(), [1e300], "floating-point overflow"),
        (pa.string(), pa.int64(), [1], "string or binary source"),
    ],
)
def test_clickhouse_sink_rejects_unsafe_values_before_the_request(
    destination: pa.DataType, source: pa.DataType, values: list[object], message: str
) -> None:
    sink = ClickHouseSink(
        "analytics", "items", endpoint="http://ch.example", destination_schema=pa.schema([("value", destination)])
    )
    worker = sink.bind(pa.schema([("value", source)])).open_worker(WriteContext("unsafe"))
    with pytest.raises(ValueError, match=message):
        worker.write(pa.table({"value": pa.array(values, type=source)}))
    assert _Client.instances[0].inserts == []


def test_clickhouse_sink_rejects_nulls_for_non_nullable_destinations_before_the_request() -> None:
    table = pa.table(
        {"source_id": [1, 2], "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "source_title": ["one", None]},
        schema=_schema(),
    )
    with pytest.raises(ValueError, match="'title' is non-nullable"):
        _worker().write(table)
    assert _Client.instances[0].inserts == []

    _Client.responses = [_success]
    _worker(destination_schema=_destination_schema(title_nullable=True)).write(table)
    assert _Client.instances[1].inserts[0].read_table().column("title").to_pylist() == ["one", None]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.1, 0.2, 0.3], None], "null vectors"),
        ([[0.1, 0.2], [0.4, 0.5, 0.6]], "invalid dimension"),
        ([[0.1, None, 0.3], [0.4, 0.5, 0.6]], "null elements"),
        ([[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]], "finite float32"),
    ],
)
def test_clickhouse_sink_validates_vector_values_before_the_request(
    vectors: list[list[float] | None], message: str
) -> None:
    schema = _schema(vector_type=pa.list_(pa.float32()))
    with pytest.raises(ValueError, match=message):
        _worker(schema=schema).write(_table(vectors=vectors, schema=schema))
    assert _Client.instances[0].inserts == []


def test_clickhouse_sink_enforces_schema_and_batch_limits_before_the_request() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    worker = _worker(max_batch_bytes=8)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    worker = _worker(max_request_bytes=64)
    with pytest.raises(ValueError, match="max_request_bytes"):
        worker.write(_table())
    worker = _worker()
    with pytest.raises(ValueError, match="does not match the bound input schema"):
        worker.write(pa.table({"source_id": [1]}))
    assert all(client.inserts == [] for client in _Client.instances)


def test_clickhouse_sink_skips_the_request_for_an_empty_batch() -> None:
    result = _worker().write(_table().slice(0, 0))
    assert result.rows_received == result.rows_affected == 0
    assert _Client.instances[0].inserts == []


def test_clickhouse_sink_defers_endpoint_and_password_resolution_to_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    bound = _bound(
        endpoint=EnvironmentSecret("VANE_TEST_CLICKHOUSE_ENDPOINT"),
        password=EnvironmentSecret("VANE_TEST_CLICKHOUSE_PASSWORD"),
    )
    assert "worker-only-password" not in repr(cloudpickle.dumps(bound))

    monkeypatch.setenv("VANE_TEST_CLICKHOUSE_ENDPOINT", "https://private-ch.example:8443/")
    monkeypatch.setenv("VANE_TEST_CLICKHOUSE_PASSWORD", "worker-only-password")
    bound.open_worker(WriteContext("secrets"))

    client = _Client.instances[0]
    assert (client.endpoint, client.password) == ("https://private-ch.example:8443", "worker-only-password")


def test_clickhouse_sink_rejects_an_invalid_secret_endpoint_before_opening_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VANE_TEST_CLICKHOUSE_ENDPOINT", "http://ch.example:8123/?user=x")
    bound = _bound(endpoint=EnvironmentSecret("VANE_TEST_CLICKHOUSE_ENDPOINT"))
    with pytest.raises(ValueError, match="query parameters"):
        bound.open_worker(WriteContext("secret-endpoint"))
    assert _Client.instances == []


def test_clickhouse_sink_never_retries_an_unknown_batch() -> None:
    _Client.responses = [_ClientError("connection reset"), _success]

    with pytest.raises(RuntimeError, match=r"query_id 'vane_.+'.*inspect that query_id.*connection reset"):
        _worker().write(_table())

    assert len(_Client.instances[0].inserts) == 1


def test_clickhouse_sink_rejects_a_response_for_another_query() -> None:
    _Client.responses = [_Summary({"query_id": "someone-else"})]
    with pytest.raises(RuntimeError, match="different query_id"):
        _worker().write(_table())

    _Client.responses = [_Summary({"written_rows": "2"})]
    assert _worker().write(_table()).rows_affected == 2


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, asyncio.CancelledError, SystemExit, GeneratorExit])
def test_clickhouse_sink_interruption_keeps_its_identity_and_the_query_id(error_type: type[BaseException]) -> None:
    interruption = error_type("planned write interruption")
    _Client.responses = [_success, interruption]
    worker = _worker()
    try:
        completed = worker.write(_table())
        with pytest.raises(error_type) as exc_info:
            worker.write(_table())

        assert exc_info.value is interruption
        assert datasink._is_execution_interruption(exc_info.value)
        query_id = _Client.instances[0].inserts[-1].settings["query_id"]
        assert query_id != completed.metadata["query_id"]
        assert f"query_id {query_id!r}" in exc_info.value.args[0]
        assert "planned write interruption" in exc_info.value.args[0]
        worker.abort(interruption)
    finally:
        worker.close()
    assert _Client.instances[0].close_calls == 1


def test_clickhouse_sink_retains_client_ownership_when_close_fails() -> None:
    worker = _worker()
    _Client.close_error = _ClientError("close failed")
    with pytest.raises(_ClientError):
        worker.close()
    _Client.close_error = None
    worker.close()
    worker.close()
    assert _Client.instances[0].close_calls == 2


@pytest.fixture(params=["local-fast", "local", pytest.param("ray", marks=pytest.mark.real_ray)])
def _clickhouse_runner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    runner_type = request.param
    runner = None
    if runner_type == "ray":
        request.getfixturevalue("ray_local")
        from vane.runners.ray.runner import RayRunner

        runner = RayRunner(address=None, max_task_backlog=None)
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    if runner is not None:
        monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: runner)
    try:
        yield runner_type
    finally:
        if runner_type == "ray":
            assert runner is not None
            runner.close()


def _live_client() -> Any:
    clickhouse_connect = pytest.importorskip("clickhouse_connect")
    endpoint = os.environ["VANE_TEST_CLICKHOUSE_ENDPOINT"]
    return clickhouse_connect.get_client(
        dsn=endpoint,
        username=os.environ.get("VANE_TEST_CLICKHOUSE_USER", "default"),
        password=os.environ.get("VANE_TEST_CLICKHOUSE_PASSWORD", ""),
    )


def _live_sink(database: str, table: str, **options: Any) -> ClickHouseSink:
    return ClickHouseSink(
        database,
        table,
        endpoint=os.environ["VANE_TEST_CLICKHOUSE_ENDPOINT"],
        user=os.environ.get("VANE_TEST_CLICKHOUSE_USER", "default"),
        password=(
            EnvironmentSecret("VANE_TEST_CLICKHOUSE_PASSWORD")
            if "VANE_TEST_CLICKHOUSE_PASSWORD" in os.environ
            else None
        ),
        timeout=60,
        **options,
    )


@pytest.fixture
def _live_database() -> Iterator[tuple[Any, str]]:
    if not os.environ.get("VANE_TEST_CLICKHOUSE_ENDPOINT"):
        pytest.skip("VANE_TEST_CLICKHOUSE_ENDPOINT is required for the external ClickHouse test")
    client = _live_client()
    database = f"vane_sink_{uuid.uuid4().hex}"
    client.command(f"CREATE DATABASE `{database}`")
    try:
        yield client, database
    finally:
        try:
            client.command(f"DROP DATABASE IF EXISTS `{database}` SYNC")
        finally:
            client.close()


@pytest.mark.external_service
def test_clickhouse_sink_live_arrow_insert(_live_database: tuple[Any, str], _clickhouse_runner: str) -> None:
    client, database = _live_database
    client.command(
        f"""
        CREATE TABLE `{database}`.`items` (
            `id` UInt32,
            `embedding` Array(Float32),
            `title` String,
            `seen_at` DateTime64(6, 'UTC')
        ) ENGINE = MergeTree ORDER BY `id`
        SETTINGS non_replicated_deduplication_window = 100
        """
    )
    destination = pa.schema(
        [
            pa.field("id", pa.uint32(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("seen_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    with vane.connect() as connection:
        # Rows 1 and 2 are identical apart from their id; rows 2 and 3 are fully
        # identical, so a deduplicating table would drop one of them.
        relation = connection.sql(
            """
            SELECT
                (CASE WHEN i = 1 THEN 1 ELSE 2 END)::BIGINT AS source_id,
                [0.1, 0.2, 0.3]::FLOAT[] AS embedding,
                'same' AS source_title,
                TIMESTAMPTZ '2026-09-05 01:02:03.456789+00' AS seen_at
            FROM range(1, 4) AS t(i)
            """
        )
        summary = relation.write_datasink(
            _live_sink(
                database,
                "items",
                destination_schema=destination,
                field_mapping={"source_id": "id", "source_title": "title"},
                vector_dimensions={"embedding": 3},
                worker_count=2,
                max_batch_rows=1,
            ),
            operation_id=f"clickhouse-live-{uuid.uuid4()}",
        )

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == summary.rows_affected == 3
    rows = client.query(
        f"SELECT id, title, length(embedding), embedding[1], toString(seen_at) FROM `{database}`.`items` ORDER BY id"
    ).result_rows
    assert [(row[0], row[1], row[2], row[4]) for row in rows] == [
        (1, "same", 3, "2026-09-05 01:02:03.456789"),
        (2, "same", 3, "2026-09-05 01:02:03.456789"),
        (2, "same", 3, "2026-09-05 01:02:03.456789"),
    ]
    assert rows[0][3] == pytest.approx(0.1)


@pytest.mark.external_service
def test_clickhouse_sink_live_rejects_null_for_a_non_nullable_column(
    _live_database: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    client, database = _live_database
    client.command(f"CREATE TABLE `{database}`.`items` (`id` UInt32, `title` String) ENGINE = MergeTree ORDER BY `id`")
    # The schema wrongly declares title nullable, so only the server can reject the NULL.
    destination = pa.schema([pa.field("id", pa.uint32(), nullable=False), pa.field("title", pa.string())])
    relation = vane.sql("SELECT 1::BIGINT AS id, NULL::VARCHAR AS title")
    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_live_sink(database, "items", destination_schema=destination))

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "inspect that query_id" in str(exc_info.value)
    assert client.query(f"SELECT count() FROM `{database}`.`items`").result_rows == [(0,)]


@pytest.mark.external_service
def test_clickhouse_sink_live_insert_is_not_replayed_after_a_remote_close(
    _live_database: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    urllib3_exceptions = pytest.importorskip("urllib3.exceptions")
    client, database = _live_database
    client.command(f"CREATE TABLE `{database}`.`items` (`id` UInt32) ENGINE = MergeTree ORDER BY `id`")
    destination = pa.schema([pa.field("id", pa.uint32(), nullable=False)])
    worker = (
        _live_sink(database, "items", destination_schema=destination)
        .bind(destination)
        .open_worker(WriteContext("remote-close"))
    )
    http = worker._client._backend.http
    real_request = http.request
    inserts: list[str] = []

    def request_then_drop_the_response(method: str, url: str, **kwargs: Any) -> Any:
        response = real_request(method, url, **kwargs)
        # Batch settings travel as URL parameters whichever way the client frames the body.
        if "query_id=vane_" not in url:
            return response
        inserts.append(url)
        response.release_conn()
        raise urllib3_exceptions.ProtocolError("Connection aborted.") from ConnectionResetError("reset by peer")

    monkeypatch.setattr(http, "request", request_then_drop_the_response)
    try:
        with pytest.raises(RuntimeError, match="inspect that query_id"):
            worker.write(pa.table({"id": pa.array([1, 2, 3], pa.uint32())}))
    finally:
        worker.close()

    assert len(inserts) == 1
    assert client.query(f"SELECT count() FROM `{database}`.`items`").result_rows == [(3,)]
