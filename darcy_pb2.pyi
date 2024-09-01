from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Array(_message.Message):
    __slots__ = ("n_cols", "n_rows", "val")
    N_COLS_FIELD_NUMBER: _ClassVar[int]
    N_ROWS_FIELD_NUMBER: _ClassVar[int]
    VAL_FIELD_NUMBER: _ClassVar[int]
    n_cols: int
    n_rows: int
    val: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, n_cols: _Optional[int] = ..., n_rows: _Optional[int] = ..., val: _Optional[_Iterable[float]] = ...) -> None: ...
