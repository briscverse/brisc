"""pacmap(), localmap(), umap()."""
import pytest


# pacmap() and localmap()

@pytest.fixture(scope="session")
def golden_pacmap(sc_orig):
    return sc_orig.qc(subset=True, allow_float=True, verbose=False)\
                  .hvg().normalize().pca()\
                  .neighbors().pacmap(num_threads=1, match_parallel=True)


@pytest.fixture(scope="session")
def golden_localmap(sc_orig):
    return sc_orig.qc(subset=True, allow_float=True, verbose=False)\
                  .hvg().normalize().pca()\
                  .neighbors().localmap(num_threads=1, match_parallel=True)


@pytest.mark.parametrize("qc_column", [False, True], ids=["subset", "qc_col"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_pacmap(sc_orig, golden_pacmap, qc_column, num_threads):
    sc = sc_orig.qc(allow_float=True, verbose=False, subset=not qc_column)\
                .hvg().normalize().pca().neighbors()
    sc = sc.pacmap(num_threads=num_threads, match_parallel=num_threads == 1)
    if qc_column:
        sc = sc.filter_obs("passed_QC").drop_obs("passed_QC")
    assert sc == golden_pacmap


@pytest.mark.parametrize("qc_column", [False, True], ids=["subset", "qc_col"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_localmap(sc_orig, golden_localmap, qc_column, num_threads):
    sc = sc_orig.qc(allow_float=True, verbose=False, subset=not qc_column)\
                .hvg().normalize().pca().neighbors()
    sc = sc.localmap(num_threads=num_threads, match_parallel=num_threads == 1)
    if qc_column:
        sc = sc.filter_obs("passed_QC").drop_obs("passed_QC")
    assert sc == golden_localmap


# umap()

@pytest.fixture(scope="session")
def golden_umap(sc_orig):
    return sc_orig.qc(subset=True, allow_float=True, verbose=False)\
                  .hvg().normalize().pca().neighbors().umap(num_threads=1)


@pytest.mark.parametrize("qc_column", [False, True], ids=["subset", "qc_col"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_umap(sc_orig, golden_umap, qc_column, num_threads):
    sc = sc_orig.qc(allow_float=True, verbose=False, subset=not qc_column)\
                .hvg().normalize().pca().neighbors()
    sc = sc.umap(num_threads=num_threads)
    if qc_column:
        sc = sc.filter_obs("passed_QC").drop_obs("passed_QC")
    assert sc == golden_umap
