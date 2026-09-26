"""hvg()."""
import pytest


@pytest.fixture(scope="session")
def golden_hvg(sc_orig):
    return sc_orig.qc(subset=True, allow_float=True, verbose=False)\
                  .hvg(num_threads=1)


@pytest.mark.parametrize("csc", [False, True], ids=["csr", "csc"])
@pytest.mark.parametrize("qc_column", [False, True], ids=["subset", "qc_col"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_hvg(sc_orig, golden_hvg, csc, qc_column, num_threads):
    sc = sc_orig.qc(allow_float=True, verbose=False, subset=not qc_column)
    if csc:
        sc = sc.tocsc()
    sc = sc.hvg(num_threads=num_threads)
    if qc_column:
        sc = sc.filter_obs("passed_QC")
    if csc:
        sc = sc.tocsr()
    assert sc.var.equals(golden_hvg.var)


def _all_zero_gene_pattern(sc):
    # Genes with no counts in any cell can never be highly variable and do not
    # enter the mean-variance fit, so excluding them must change nothing.
    import re
    import numpy as np
    zero = np.asarray(sc.X.sum(axis=0)).ravel() == 0
    names = sc.var_names.filter(zero).to_list()
    assert len(names) > 10, "fixture needs some all-zero genes for this test"
    return tuple(f"^{re.escape(name)}$" for name in names)


@pytest.mark.parametrize("multiple_datasets", [False, True],
                         ids=["one_dataset", "two_datasets"])
def test_hvg_exclude_keeps_genes_aligned(sc_orig, multiple_datasets):
    # Regression test: `exclude` used to renumber genes AFTER filtering, so
    # each gene received the statistics of a different column of X (and with a
    # single dataset, the gene list and the statistics had different lengths).
    import polars as pl
    sc = sc_orig.qc(subset=True, allow_float=True, verbose=False)
    exclude = _all_zero_gene_pattern(sc)
    if multiple_datasets:
        half = sc.shape[0] // 2
        a = sc.filter_obs(pl.int_range(pl.len()) < half)
        b = sc.filter_obs(pl.int_range(pl.len()) >= half)
        expected = a.hvg(b, num_threads=1)
        observed = a.hvg(b, exclude=exclude, num_threads=1)
        pairs = zip(expected, observed)
    else:
        pairs = [(sc.hvg(num_threads=1), sc.hvg(exclude=exclude, num_threads=1))]
    for e, o in pairs:
        assert o.var.equals(e.var)
