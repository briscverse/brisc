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
