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


def test_hvg_min_cells_counts_detections_across_batches(sc_orig):
    # A gene that is highly variable in one dataset must not be dropped from
    # the joint selection just because another dataset (here, cells in which
    # it is never detected, like a sorted population) has fewer than
    # `min_cells` cells expressing it.
    import numpy as np
    import polars as pl
    sc = sc_orig.qc(subset=True, allow_float=True, verbose=False)
    ranks = sc.hvg(num_threads=1).var
    gene = ranks.filter(pl.col("highly_variable_rank") == 1)[:, 0].item()
    column = sc.var_names.to_list().index(gene)
    counts = np.asarray(sc.X[:, [column]].todense()).ravel()
    without_gene = sc.filter_obs(pl.Series(counts == 0))
    assert without_gene.shape[0] >= 1000
    joint = sc.hvg(without_gene, num_threads=1)[0].var
    assert joint.filter(pl.col(joint.columns[0]) == gene)["highly_variable"].item()
