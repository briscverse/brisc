"""Integer columns used as numeric covariates in de() / regress_out()."""
import polars as pl
import pytest


def _pb(sc_orig, age_dtype):
    """Pseudobulk with an `age` covariate of the requested dtype.

    `age` is derived from the sample name so it is constant within a sample and
    varies across samples, i.e. a well-behaved continuous covariate. The map is
    built from the union of samples across cell types, since qc() can retain
    different samples for different cell types.
    """
    pb = sc_orig.qc(verbose=False, allow_float=True)\
                .pseudobulk("sample", "cell_type", verbose=False)\
                .qc("treatment", verbose=False)\
                .library_size()
    samples = sorted({s for _, (X, obs, var) in pb.items()
                      for s in obs["sample"].cast(pl.String).to_list()})
    ages = {s: 40 + 3 * i for i, s in enumerate(samples)}
    # `batch` is a low-cardinality integer factor keyed on donor, so it varies
    # independently of `treatment` (each donor appears under both treatments)
    donors = sorted({s.split("_")[0] for s in samples})
    batches = {s: donors.index(s.split("_")[0]) % 2 for s in samples}
    return pb.with_columns_obs(
        pl.col("sample").cast(pl.String)
          .replace_strict(ages, default=40, return_dtype=pl.Int64)
          .cast(age_dtype).alias("age"),
        pl.col("sample").cast(pl.String)
          .replace_strict(batches, default=0, return_dtype=pl.Int64)
          .alias("batch"))


@pytest.mark.parametrize("age_dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                       pl.UInt32, pl.UInt64])
def test_integer_covariate_matches_float(sc_orig, require_r, age_dtype):
    """An integer covariate must give the same DE as the same values as Float64.

    Regression test: `to_r` maps polars integers to R's `integer64` (bit64)
    class, which `model.matrix()` reinterprets bitwise -- turning an age of 76
    into 3.75e-322. That silently produced either a rank-deficient design
    ("rank 3 with 6 columns") or NaN coefficients, with nothing in the error
    naming the dtype as the cause.
    """
    formula = "~ treatment + age"
    de_int = _pb(sc_orig, age_dtype).de(formula, coefficient="treatmentcytokine",
                                        verbose=False, strict=True)
    de_float = _pb(sc_orig, pl.Float64).de(formula, coefficient="treatmentcytokine",
                                           verbose=False, strict=True)
    assert de_int == de_float


def test_integer_covariate_is_not_dropped(sc_orig, require_r):
    """The integer covariate must actually enter the model, not be ignored."""
    pb = _pb(sc_orig, pl.Int64)
    with_age = pb.de("~ treatment + age", coefficient="treatmentcytokine",
                     verbose=False, strict=True)
    without_age = pb.de("~ treatment", coefficient="treatmentcytokine",
                        verbose=False, strict=True)
    assert with_age != without_age


def test_integer_covariate_regress_out(sc_orig, require_r):
    """regress_out() shares _create_design_matrix(), so it needs the same fix."""
    a = _pb(sc_orig, pl.Int64).log_cpm().regress_out("~ age")
    b = _pb(sc_orig, pl.Float64).log_cpm().regress_out("~ age")
    for cell_type in a.keys():
        assert a.X[cell_type] == pytest.approx(b.X[cell_type], rel=1e-6)


def test_categorical_integer_column_becomes_a_factor(sc_orig, require_r):
    """An integer column named in `categorical_columns` must become a factor.

    Regression test: the levels were derived as strings but the column was left
    as an integer, so the Enum cast raised
    `InvalidOperationError: conversion from i64 to enum failed`, making
    `categorical_columns` unusable for any integer column. Also confirms the
    Float64 cast for numeric integer covariates does not capture these columns.
    """
    pb = _pb(sc_orig, pl.Int64)
    de = pb.de("~ treatment + batch", coefficient="treatmentcytokine",
               categorical_columns="batch", verbose=False, strict=True)
    assert de is not None
