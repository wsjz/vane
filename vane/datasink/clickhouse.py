# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed ClickHouse Arrow inserts.

Each worker converts an Arrow table directly to an Arrow IPC stream and sends
one synchronous ``INSERT ... FORMAT ArrowStream`` request per batch through the
ClickHouse HTTP interface. ClickHouse commits worker batches independently;
Vane does not provide a transaction spanning the complete distributed relation.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
import pyarrow.compute as pc  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.datasink import (
    BoundDataSink,
    DataSink,
    DataSinkExecutionOptions,
    DataSinkWorker,
    EnvironmentSecret,
    WriteContext,
    WriteResult,
)
from vane.datasink._arrow_schema import same_input_type
from vane.execution._diagnostics import bounded_utf8_text, exception_message_from_args, safe_exception_type_name

if TYPE_CHECKING:
    from types import ModuleType

_MAX_INT32 = (1 << 31) - 1
_MAX_INT64 = (1 << 63) - 1
_DEFAULT_MAX_BATCH_ROWS = 1_000_000
_DEFAULT_MAX_BATCH_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_REQUEST_BYTES = 160 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 600
_CONNECT_TIMEOUT_SECONDS = 30
_DEFAULT_PORTS = {"http": 8123, "https": 8443}
_QUERY_ID_PREFIX = "vane"
_QUERY_ID_PATTERN = re.compile(r"[_A-Za-z0-9]+\Z")
_OBJECT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INSERT_FORMAT = "ArrowStream"
# ClickHouse otherwise stores column defaults for NULL input and may buffer
# inserts server-side; both would hide the batch outcome from the worker.
_INSERT_SETTINGS = {"async_insert": 0, "input_format_null_as_default": 0}
_PARTIAL_VISIBILITY_WARNING = (
    "ClickHouse commits worker batches independently; a later operation failure can leave this batch visible"
)


def _name(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip() or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must not contain surrounding whitespace, NUL, CR, or LF characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8") from error
    return value


def _clickhouse_object(name: str, value: object) -> str:
    identifier = _name(name, value)
    if len(identifier) > 256:
        raise ValueError(f"{name} must be at most 256 characters")
    if _OBJECT_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{name} must be an ASCII ClickHouse identifier of letters, digits, and underscores")
    return identifier


def _quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("\\", "\\\\").replace("`", "\\`") + "`"


def _endpoint(value: object) -> str:
    endpoint = _name("endpoint", value)
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("endpoint must be a valid HTTP or HTTPS ClickHouse endpoint") from error
    if parsed.scheme not in _DEFAULT_PORTS or hostname is None:
        raise ValueError("endpoint must be an absolute HTTP or HTTPS ClickHouse endpoint")
    if parsed.username is not None or parsed.password is not None or "?" in endpoint or "#" in endpoint:
        raise ValueError(
            "endpoint must not contain credentials, query parameters, or fragments; use password=EnvironmentSecret(...)"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("endpoint must not contain a path")
    return endpoint.rstrip("/")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a positive signed 64-bit integer")
    return value


def _field_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("field_mapping must be a mapping or None")
    return tuple(
        (_name("field_mapping source", source), _name("field_mapping target", target))
        for source, target in value.items()
    )


def _vector_dimensions(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("vector_dimensions must be a mapping or None")
    normalized: list[tuple[str, int]] = []
    for source, dimension in value.items():
        normalized_dimension = _positive_int("vector dimension", dimension)
        if normalized_dimension > _MAX_INT32:
            raise ValueError("vector dimension must fit in a signed 32-bit Arrow ListArray offset")
        normalized.append((_name("vector_dimensions source", source), normalized_dimension))
    return tuple(normalized)


def _naive_timestamp_error(kind: str, path: str) -> ValueError:
    return ValueError(
        f"{kind} field {path!r} is a timestamp without a time zone; ClickHouse reads Arrow timestamps as UTC "
        "instants, so declare the time zone explicitly, for example with a TIMESTAMPTZ value"
    )


def _validate_destination_type(data_type: pa.DataType, path: str) -> None:
    if pa.types.is_timestamp(data_type):
        if data_type.tz is None:
            raise _naive_timestamp_error("ClickHouse destination", path)
        return
    if (
        pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_float32(data_type)
        or pa.types.is_float64(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_date32(data_type)
    ):
        return
    if pa.types.is_list(data_type):
        _validate_destination_type(data_type.value_type, f"{path}[]")
        return
    raise ValueError(
        f"destination_schema field {path!r} uses unsupported Arrow type {data_type}; supported types are bool, "
        "signed and unsigned integers, float32, float64, string, date32, timestamps with a time zone, "
        "and list values composed from them"
    )


def _validate_source_type(data_type: pa.DataType, path: str) -> None:
    if pa.types.is_timestamp(data_type):
        if data_type.tz is None:
            raise _naive_timestamp_error("ClickHouse source", path)
        return
    if (
        pa.types.is_null(data_type)
        or pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
        or pa.types.is_date32(data_type)
    ):
        return
    if _is_list_type(data_type):
        _validate_source_type(data_type.value_type, f"{path}[]")
        return
    raise ValueError(
        f"ClickHouse source field {path!r} uses unsupported Arrow type {data_type}; "
        "use plain scalar or standard list arrays, and decode and rebatch encoded or view arrays before writing"
    )


def _is_list_type(data_type: pa.DataType) -> bool:
    return pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)


def _validate_temporal_pair(source_type: pa.DataType, target_type: pa.DataType, path: str) -> None:
    # Arrow's safe cast truncates timestamps to dates and turns numbers or
    # strings into instants, so temporal values only cross between equal kinds.
    if pa.types.is_null(source_type):
        return
    if _is_list_type(source_type) and pa.types.is_list(target_type):
        _validate_temporal_pair(source_type.value_type, target_type.value_type, f"{path}[]")
        return
    for kind, matches in (("date32", pa.types.is_date32), ("timestamp", pa.types.is_timestamp)):
        if matches(source_type) != matches(target_type):
            raise ValueError(
                f"ClickHouse field {path!r} writes Arrow {source_type} to {target_type}; {kind} values are "
                f"written only from {kind} input, so convert the value explicitly in the input relation"
            )


def _destination_schema(value: object) -> pa.Schema:
    if not isinstance(value, pa.Schema):
        raise TypeError("destination_schema must be pyarrow.Schema")
    if not value:
        raise ValueError("destination_schema must contain at least one field")
    fields: list[pa.Field] = []
    names: list[str] = []
    for field in value:
        name = _name("destination_schema field name", field.name)
        _validate_destination_type(field.type, name)
        names.append(name)
        fields.append(pa.field(name, field.type, nullable=field.nullable))
    if len(set(names)) != len(names):
        raise ValueError("destination_schema field names must be unique")
    return pa.schema(fields)


def _load_clickhouse_connect() -> ModuleType:
    try:
        import clickhouse_connect  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name != "clickhouse_connect":
            raise
        raise ImportError("ClickHouseSink requires clickhouse-connect; install vane-ai[clickhouse]") from error
    return clickhouse_connect


def _open_client(endpoint: str, user: str, password: str, timeout: int) -> Any:
    clickhouse_connect = _load_clickhouse_connect()
    parts = urlsplit(endpoint)
    return clickhouse_connect.get_client(
        interface=parts.scheme,
        host=parts.hostname,
        port=parts.port or _DEFAULT_PORTS[parts.scheme],
        username=user,
        password=password,
        compress=False,
        query_retries=0,
        connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        send_receive_timeout=timeout,
        autogenerate_session_id=False,
        autogenerate_query_id=False,
    )


@dataclass(frozen=True)
class _FieldBinding:
    source_name: str
    target_name: str
    target_type: pa.DataType
    target_nullable: bool
    vector_dimension: int | None


class ClickHouseSink(DataSink):
    """Append relation batches to a ClickHouse table with Arrow inserts.

    ``destination_schema`` is required and must declare the exact Arrow
    physical type for each ClickHouse column selected by this sink, in
    input-column order. Every input column is safely cast to that schema before
    upload; overflow and incompatible values fail the batch without a request.
    Worker batches may use equivalent 32-bit or 64-bit string, binary, and list
    offsets; logical input types must otherwise match the bound schema.
    Destination types are bool, signed and unsigned integers, float32, float64,
    string, date32, timestamps with a time zone, and lists composed from them.
    Input types are null, boolean, integer, floating-point, string, binary,
    date32, timestamps with a time zone, and standard lists recursively
    composed from them. Encoded/view arrays and other input types must be
    converted and rebatched before writing. String destinations require string
    or binary input; stringify other types in the input relation. date32 and
    timestamp destinations require input of the same temporal kind. Hidden
    children beneath null list slots are not converted. Timestamps without a
    time zone are rejected because ClickHouse reads every Arrow timestamp as a
    UTC instant. ``field_mapping`` maps input Arrow field names to ClickHouse
    columns. Unmapped input fields retain their names, and the resulting names
    must exactly match ``destination_schema``. ``vector_dimensions`` maps input
    fields to the exact dimensions of ClickHouse ``Array(Float32)`` vector
    columns. Declared vector columns must contain non-null, finite float32
    values and are encoded as Arrow ``ListArray`` columns without converting
    rows to Python objects.

    Inserts are synchronous and reject NULL input for non-Nullable ClickHouse
    columns instead of storing column defaults. Each batch carries a unique
    ``insert_deduplication_token``, so tables that deduplicate inserted blocks
    keep identical batches. The same value is the batch ``query_id``.

    ``max_batch_bytes`` limits input Arrow data and ``max_request_bytes`` is a
    separate hard limit on the encoded IPC request. Conversion buffers are
    conservatively budgeted before casting, and the complete IPC size is
    measured before allocating a fixed-size request buffer. Vane full-operation
    and client-level insert retries remain disabled because the current
    DataSink batch contract has no replay-stable batch identity. If a
    connection fails after upload, the reported outcome is UNKNOWN and the
    batch ``query_id`` must be inspected in ``system.query_log`` before any
    manual retry.
    """

    def __init__(
        self,
        database: str,
        table: str,
        *,
        endpoint: str | EnvironmentSecret,
        destination_schema: pa.Schema,
        user: str = "default",
        password: EnvironmentSecret | None = None,
        field_mapping: Mapping[str, str] | None = None,
        vector_dimensions: Mapping[str, int] | None = None,
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.database = _clickhouse_object("database", database)
        self.table = _clickhouse_object("table", table)
        self.endpoint: str | EnvironmentSecret
        if isinstance(endpoint, EnvironmentSecret):
            self.endpoint = endpoint
        else:
            self.endpoint = _endpoint(endpoint)
        self._destination_schema = _destination_schema(destination_schema)
        self.user = _name("user", user)
        if password is not None and not isinstance(password, EnvironmentSecret):
            raise TypeError("password must be an EnvironmentSecret or None")
        self._password = password
        self._field_mapping = _field_mapping(field_mapping)
        self._vector_dimensions = _vector_dimensions(vector_dimensions)
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        self.max_request_bytes = _positive_int("max_request_bytes", max_request_bytes)
        self.timeout = _positive_int("timeout", timeout)

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        if not schema:
            raise ValueError("ClickHouseSink requires at least one input column")
        if len(set(schema.names)) != len(schema.names):
            raise ValueError("ClickHouseSink requires unique input column names")
        for field in schema:
            _validate_source_type(field.type, field.name)

        mapping = dict(self._field_mapping)
        if len(mapping) != len(self._field_mapping):
            raise ValueError("field_mapping sources must be unique")
        vector_dimensions = dict(self._vector_dimensions)
        if len(vector_dimensions) != len(self._vector_dimensions):
            raise ValueError("vector_dimensions sources must be unique")
        unknown_mapping = set(mapping).difference(schema.names)
        if unknown_mapping:
            raise ValueError(f"field_mapping contains unknown input columns: {sorted(unknown_mapping)!r}")
        unknown_vectors = set(vector_dimensions).difference(schema.names)
        if unknown_vectors:
            raise ValueError(f"vector_dimensions contains unknown input columns: {sorted(unknown_vectors)!r}")

        target_names = [_name("ClickHouse target column", mapping.get(field.name, field.name)) for field in schema]
        if len(set(target_names)) != len(target_names):
            raise ValueError("field_mapping must produce unique ClickHouse column names")
        if target_names != self._destination_schema.names:
            raise ValueError(
                "mapped input columns must exactly match destination_schema names and order: "
                f"mapped={target_names!r}, destination={self._destination_schema.names!r}"
            )

        fields: list[_FieldBinding] = []
        for field, target_field in zip(schema, self._destination_schema, strict=True):
            target_name = target_field.name
            vector_dimension = vector_dimensions.get(field.name)
            if vector_dimension is not None:
                data_type = field.type
                if not (
                    pa.types.is_list(data_type)
                    or pa.types.is_large_list(data_type)
                    or pa.types.is_fixed_size_list(data_type)
                ) or not pa.types.is_float32(data_type.value_type):
                    raise ValueError(
                        f"ClickHouse vector source field {field.name!r} must be an Arrow list, large_list, "
                        "or fixed_size_list of float32"
                    )
                if pa.types.is_fixed_size_list(data_type) and data_type.list_size != vector_dimension:
                    raise ValueError(
                        f"ClickHouse vector source field {field.name!r} fixed dimension {data_type.list_size} "
                        f"does not match vector_dimensions value {vector_dimension}"
                    )
                if not pa.types.is_list(target_field.type) or not pa.types.is_float32(target_field.type.value_type):
                    raise ValueError(
                        f"ClickHouse vector destination field {target_name!r} must be Arrow list<float32> "
                        "for an Array(Float32) column"
                    )
            _validate_temporal_pair(field.type, target_field.type, field.name)
            fields.append(
                _FieldBinding(
                    source_name=field.name,
                    target_name=target_name,
                    target_type=target_field.type,
                    target_nullable=target_field.nullable,
                    vector_dimension=vector_dimension,
                )
            )
        return _BoundClickHouseSink(self, schema, self._destination_schema, tuple(fields))

    @property
    def destination_schema(self) -> pa.Schema:
        return self._destination_schema

    def _resolve_endpoint(self) -> str:
        endpoint = self.endpoint
        if isinstance(endpoint, EnvironmentSecret):
            return _endpoint(endpoint.resolve())
        return endpoint


class _BoundClickHouseSink(BoundDataSink):
    def __init__(
        self,
        sink: ClickHouseSink,
        schema: pa.Schema,
        destination_schema: pa.Schema,
        fields: tuple[_FieldBinding, ...],
    ) -> None:
        self._sink = sink
        self._schema = schema
        self._destination_schema = destination_schema
        self._fields = fields

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            worker_count=self._sink.worker_count,
            batch_size=self._sink.max_batch_rows,
            target_max_batch_bytes=self._sink.max_batch_bytes,
            max_retries=0,
        )

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _ClickHouseWorker(
            self._sink,
            self._schema,
            self._destination_schema,
            self._fields,
            context,
        )


class _ClickHouseWorker(DataSinkWorker):
    def __init__(
        self,
        sink: ClickHouseSink,
        schema: pa.Schema,
        destination_schema: pa.Schema,
        fields: tuple[_FieldBinding, ...],
        context: WriteContext,
    ) -> None:
        endpoint = sink._resolve_endpoint()
        password = "" if sink._password is None else sink._password.resolve()

        self._sink = sink
        self._schema = schema
        self._destination_schema = destination_schema
        self._fields = fields
        self._target = f"{_quote_identifier(sink.database)}.{_quote_identifier(sink.table)}"
        # clickhouse-connect sends identifiers that are already validly quoted unchanged.
        self._columns = tuple(_quote_identifier(field.target_name) for field in fields)
        operation_digest = hashlib.sha256(context.operation_id.encode("utf-8")).hexdigest()[:16]
        self._query_id_stem = f"{_QUERY_ID_PREFIX}_{operation_digest}_{uuid.uuid4().hex}"
        self._batch_number = 0
        self._warning_pending = True
        self._base_metadata = {
            "provider": "clickhouse",
            "database": sink.database,
            "table": sink.table,
            "format": _INSERT_FORMAT,
        }
        self._client: Any | None = _open_client(endpoint, sink.user, password, sink.timeout)

    def _client_or_raise(self) -> Any:
        if self._client is None:
            raise RuntimeError("ClickHouseSink worker is closed")
        return self._client

    def _next_query_id(self) -> str:
        self._batch_number += 1
        query_id = f"{self._query_id_stem}_{self._batch_number:x}"
        if _QUERY_ID_PATTERN.fullmatch(query_id) is None:
            raise AssertionError("generated ClickHouse query_id is invalid")
        return query_id

    def _validate_table(self, table: pa.Table) -> tuple[int, int]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"ClickHouseSink expected pyarrow.Table, got {type(table).__name__}")
        # Local/Ray workers enable arrow_large_buffer_size independently of the
        # binding connection. Keep logical types stable, then let the existing
        # destination safe cast normalize offsets directly to the wire schema.
        if len(table.schema) != len(self._schema) or any(
            batch_field.name != bound_field.name or not same_input_type(batch_field.type, bound_field.type)
            for batch_field, bound_field in zip(table.schema, self._schema, strict=True)
        ):
            raise ValueError("ClickHouse batch schema does not match the bound input schema")
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("ClickHouse batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("ClickHouse batch exceeds max_batch_bytes")
        return row_count, batch_bytes

    def _vector_column(self, column: pa.ChunkedArray, binding: _FieldBinding) -> pa.ChunkedArray:
        dimension = binding.vector_dimension
        assert dimension is not None
        chunks: list[pa.Array] = []
        for chunk in column.chunks:
            if chunk.null_count:
                raise ValueError(
                    f"ClickHouse vector source field {binding.source_name!r} must not contain null vectors"
                )
            if not pa.types.is_fixed_size_list(chunk.type):
                lengths = pc.list_value_length(chunk)
                wrong_length = pc.any(pc.not_equal(lengths, pa.scalar(dimension, type=lengths.type)))
                if wrong_length.as_py() is True:
                    raise ValueError(
                        f"ClickHouse vector source field {binding.source_name!r} contains a vector with an "
                        "invalid dimension"
                    )
            values = pc.list_flatten(chunk)
            if values.null_count:
                raise ValueError(
                    f"ClickHouse vector source field {binding.source_name!r} must not contain null elements"
                )
            finite = pc.all(pc.is_finite(values))
            if finite.as_py() is False:
                raise ValueError(
                    f"ClickHouse vector source field {binding.source_name!r} must contain only finite float32 values"
                )
            nested_count = len(chunk) * dimension
            if nested_count > _MAX_INT32:
                raise ValueError(
                    f"ClickHouse vector source field {binding.source_name!r} exceeds Arrow ListArray limits"
                )
            offsets = pa.array(
                np.arange(0, nested_count + 1, dimension, dtype=np.int32),
                type=pa.int32(),
            )
            chunks.append(pa.ListArray.from_arrays(offsets, values))
        return pa.chunked_array(chunks, type=pa.list_(pa.float32()))

    def _validate_nullability(
        self,
        array: pa.Array | pa.ChunkedArray,
        field: pa.Field,
        path: str,
    ) -> None:
        if not field.nullable and array.null_count:
            raise ValueError(
                f"ClickHouse destination field {path!r} is non-nullable but the batch contains null values"
            )
        if not pa.types.is_list(field.type):
            return
        child_field = field.type.value_field
        chunks = array.chunks if isinstance(array, pa.ChunkedArray) else (array,)
        for chunk in chunks:
            self._validate_nullability(pc.list_flatten(chunk), child_field, f"{path}[]")

    def _destination_column(self, column: pa.ChunkedArray, binding: _FieldBinding) -> pa.ChunkedArray:
        if column.type != binding.target_type:
            try:
                converted = pc.cast(column, binding.target_type, safe=True)
                self._validate_float_overflow(column, converted, binding.target_name)
                column = converted
            except pa.ArrowException as error:
                raise ValueError(
                    f"ClickHouse source field {binding.source_name!r} cannot be safely cast to destination field "
                    f"{binding.target_name!r} with Arrow type {binding.target_type}: {error}"
                ) from error
        target_field = pa.field(
            binding.target_name,
            binding.target_type,
            nullable=binding.target_nullable,
        )
        self._validate_nullability(column, target_field, binding.target_name)
        return column

    def _validate_float_overflow(
        self,
        source: pa.Array | pa.ChunkedArray,
        converted: pa.Array | pa.ChunkedArray,
        path: str,
    ) -> None:
        if source.type == converted.type or pa.types.is_null(source.type):
            return
        if pa.types.is_list(converted.type):
            self._validate_float_overflow(pc.list_flatten(source), pc.list_flatten(converted), f"{path}[]")
        elif pa.types.is_float32(converted.type):
            # Arrow's safe cast permits finite doubles to overflow to float infinity.
            # Inspect float64 source values only when the result contains a nonfinite
            # value, preserving existing infinities/NaNs and ordinary float rounding.
            nonfinite = pc.invert(pc.is_finite(converted))
            if pc.any(nonfinite).as_py() is True:
                source_double = pc.cast(source, pa.float64(), safe=True)
                overflow = pc.and_(pc.is_finite(source_double), nonfinite)
                if pc.any(overflow).as_py() is True:
                    raise ValueError(
                        f"ClickHouse destination field {path!r} has floating-point overflow: "
                        "a finite source value becomes nonfinite when cast to float32"
                    )

    def _visible_list_values(self, array: pa.Array) -> pa.Array:
        if not (
            pa.types.is_list(array.type)
            or pa.types.is_large_list(array.type)
            or pa.types.is_fixed_size_list(array.type)
        ):
            return array
        # Arrow's nested safe cast visits physical children even beneath null
        # list slots. Flatten first so only logically visible values are cast.
        values = self._visible_list_values(pc.list_flatten(array))
        if len(values) > _MAX_INT32:
            raise ValueError("ClickHouse list values exceed Arrow ListArray limits")
        lengths = pc.fill_null(pc.list_value_length(array), 0)
        offsets = pa.concat_arrays([pa.array([0], type=lengths.type), pc.cumulative_sum(lengths)])
        child = pa.field(array.type.value_field.name, values.type, nullable=array.type.value_field.nullable)
        mask = array.is_null() if array.null_count else None
        if pa.types.is_large_list(array.type):
            return pa.LargeListArray.from_arrays(offsets, values, type=pa.large_list(child), mask=mask)
        return pa.ListArray.from_arrays(offsets, values, type=pa.list_(child), mask=mask)

    def _conversion_bytes(self, array: pa.Array, target_type: pa.DataType) -> int:
        count = len(array)
        # Include validity storage even when a kernel can omit it. Arithmetic
        # stays in Python integers so a size estimate cannot silently overflow.
        validity_bytes = (count + 7) // 8
        if pa.types.is_list(target_type):
            size = validity_bytes + 4 * (count + 1)
            if pa.types.is_null(array.type):
                return size + self._conversion_bytes(array.slice(0, 0), target_type.value_type)
            if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
                raise ValueError("ClickHouse list destinations require a standard list source")
            return size + self._conversion_bytes(array.values, target_type.value_type)
        if pa.types.is_string(target_type):
            if count == 0 or pa.types.is_null(array.type):
                data_bytes = 0
            elif pa.types.is_fixed_size_binary(array.type):
                data_bytes = count * array.type.byte_width
            elif (
                pa.types.is_string(array.type)
                or pa.types.is_binary(array.type)
                or pa.types.is_large_string(array.type)
                or pa.types.is_large_binary(array.type)
            ):
                large = pa.types.is_large_string(array.type) or pa.types.is_large_binary(array.type)
                offsets = np.frombuffer(array.buffers()[1], dtype=np.int64 if large else np.int32)
                data_bytes = int(offsets[array.offset + count]) - int(offsets[array.offset])
            else:
                raise ValueError("ClickHouse string destinations require a string or binary source")
            return validity_bytes + 4 * (count + 1) + data_bytes
        return validity_bytes + (count * target_type.bit_width + 7) // 8

    def _wire_table(self, table: pa.Table) -> pa.Table:
        prepared: list[pa.ChunkedArray] = []
        conversion_bytes = 0
        for index, binding in enumerate(self._fields):
            column = table.column(index)
            if binding.vector_dimension is not None:
                column = self._vector_column(column, binding)
            else:
                column = pa.chunked_array([self._visible_list_values(chunk) for chunk in column.chunks])
            conversion_bytes += sum(self._conversion_bytes(chunk, binding.target_type) for chunk in column.chunks)
            if conversion_bytes > self._sink.max_request_bytes:
                raise ValueError("ClickHouse converted Arrow data exceeds max_request_bytes before casting")
            prepared.append(column)
        arrays = [
            self._destination_column(column, binding) for column, binding in zip(prepared, self._fields, strict=True)
        ]
        return pa.Table.from_arrays(arrays, schema=self._destination_schema)

    def _arrow_body(self, table: pa.Table) -> pa.Buffer:
        # Size the complete IPC stream, including schema/chunk metadata, before
        # allocating its body. The fixed-size writer cannot grow past this cap.
        with pa.MockOutputStream() as sizing_output:
            with pa.ipc.new_stream(sizing_output, table.schema) as writer:
                writer.write_table(table)
            wire_bytes = sizing_output.size()
        if wire_bytes > self._sink.max_request_bytes:
            raise ValueError(
                "ClickHouse Arrow IPC request exceeds max_request_bytes: "
                f"wire_bytes={wire_bytes}, max_request_bytes={self._sink.max_request_bytes}"
            )
        body = pa.allocate_buffer(wire_bytes)
        with pa.FixedSizeBufferWriter(body) as output:
            with pa.ipc.new_stream(output, table.schema) as writer:
                writer.write_table(table)
        return body

    def _insert(self, query_id: str, body: pa.Buffer) -> Any:
        return self._client_or_raise().raw_insert(
            self._target,
            self._columns,
            # clickhouse-connect replays a bytes body after a remote close; a
            # memoryview body is sent exactly once.
            memoryview(body),
            settings={**_INSERT_SETTINGS, "insert_deduplication_token": query_id, "query_id": query_id},
            fmt=_INSERT_FORMAT,
        )

    def _applied_result(
        self,
        *,
        summary: Any,
        query_id: str,
        row_count: int,
        batch_bytes: int,
        body_size: int,
    ) -> WriteResult:
        fields = getattr(summary, "summary", None)
        if not isinstance(fields, Mapping):
            raise RuntimeError("ClickHouse insert returned an invalid summary")
        # Proxies may drop X-ClickHouse-Query-Id; a different value means the
        # response does not belong to this batch.
        response_query_id = fields.get("query_id")
        if response_query_id not in (None, "", query_id):
            raise RuntimeError("ClickHouse insert returned a different query_id")
        warnings = (_PARTIAL_VISIBILITY_WARNING,) if self._warning_pending else ()
        result = WriteResult(
            rows_received=row_count,
            rows_affected=row_count,
            bytes_received=batch_bytes,
            metadata={**self._base_metadata, "query_id": query_id, "wire_bytes": body_size},
            warnings=warnings,
        )
        self._warning_pending = False
        return result

    def write(self, table: pa.Table) -> WriteResult:
        row_count, batch_bytes = self._validate_table(table)
        if row_count == 0:
            return WriteResult(
                rows_received=0,
                rows_affected=0,
                bytes_received=batch_bytes,
                metadata=self._base_metadata,
            )
        wire_table = self._wire_table(table)
        body = self._arrow_body(wire_table)
        query_id = self._next_query_id()
        try:
            summary = self._insert(query_id, body)
            return self._applied_result(
                summary=summary,
                query_id=query_id,
                row_count=row_count,
                batch_bytes=batch_bytes,
                body_size=body.size,
            )
        except BaseException as error:
            message = (
                f"ClickHouse insert query_id {query_id!r} did not produce an accepted response; "
                f"inspect that query_id in system.query_log before retrying: {safe_exception_type_name(error)}"
            )
            detail = exception_message_from_args(error)
            if detail:
                message = f"{message}: {bounded_utf8_text(detail, 512)}"
            if isinstance(error, Exception):
                raise RuntimeError(message) from error
            # Keep cancellation/interrupt identity and classification. DataSink
            # diagnostics read args, not exception notes.
            BaseException.__setattr__(error, "args", (message,))
            raise

    def abort(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        client.close()
        self._client = None


__all__ = ["ClickHouseSink"]
