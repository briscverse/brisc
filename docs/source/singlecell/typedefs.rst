:orphan:
:description: Type aliases used across the brisc API: SingleCellColumn, PseudobulkColumn, Scalar, UnsDict, UnsItem, Color and Indexer, with the types each one accepts.

.. _typedefs:

Type aliases
============

.. currentmodule:: brisc

SingleCellColumn
----------------

``str | pl.Expr | pl.Series | np.ndarray | Callable[[SingleCell], pl.Series | np.ndarray]``

A flexible column specification. Can be a column name (string), a Polars
expression, a Polars Series, a 1D NumPy array, or a function that takes a
SingleCell dataset and returns a Series or array.

Scalar
------

``str | int | float | Decimal | date | time | datetime | timedelta | bool | bytes | Expr | Series | Iterable[Scalar]``

Any primitive literal or sequence thereof. Accepted wherever a single
value (or a 1-D sequence of values) is expected -- filter thresholds,
default fills, single-column inputs, etc.

UnsDict
-------

``dict[str, str | int | float | bool | np.ndarray | 'UnsDict']``

A nested dictionary of unstructured metadata. Values can be scalars
(strings, numbers, Booleans), NumPy arrays, or nested dictionaries
of the same types. Stored in :attr:`SingleCell.uns`.

UnsItem
-------

``str | int | float | bool | np.ndarray``

A single value inside an :attr:`SingleCell.uns` dictionary: a scalar
(string, number, or Boolean) or a NumPy array.

PseudobulkColumn
----------------

``str | pl.Expr | pl.Series | np.ndarray | Callable[[Pseudobulk, str], pl.Series | np.ndarray]``

The :class:`Pseudobulk` analogue of `SingleCellColumn`_. A column name,
Polars expression, Polars Series, 1D NumPy array, or a function that takes a
Pseudobulk dataset and a cell-type name and returns a Series or array.

Color
-----

``str | float | tuple[int, int, int] | tuple[int, int, int, int] | tuple[float, float, float] | tuple[float, float, float, float]``

A color specification: a named or hex color string, a grayscale float, or an
RGB(A) tuple of three or four ints (0--255) or floats (0--1).

Indexer
-------

``int | str | slice | np.ndarray | pl.Series | list[int | str | bool]``

Anything that can index a dataset: an integer or string position/name, a
slice, or a Boolean/integer mask (NumPy array, Polars Series, or list).
