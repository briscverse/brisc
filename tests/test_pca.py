"""pca()."""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def golden_pca(sc_orig):
    return sc_orig.qc(subset=True, allow_float=True, verbose=False)\
                  .hvg().normalize().pca(num_threads=1, match_parallel=True)


@pytest.mark.parametrize("csc", [False, True], ids=["csr", "csc"])
@pytest.mark.parametrize("qc_column", [False, True], ids=["subset", "qc_col"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_pca(sc_orig, golden_pca, csc, qc_column, num_threads):
    sc = sc_orig.qc(allow_float=True, verbose=False, subset=not qc_column)\
                .hvg().normalize()
    if csc:
        sc = sc.tocsc()
    sc = sc.pca(num_threads=num_threads, match_parallel=num_threads == 1)
    if qc_column:
        sc = sc.filter_obs("passed_QC")
    if csc:
        sc = sc.tocsr()
    assert np.array_equal(sc.obsm["pca"], golden_pca.obsm["pca"])


@pytest.mark.parametrize("csc", [False, True], ids=["csr", "csc"])
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("scale", [False, True], ids=["center", "scale"])
def test_pca_scale(csc, num_threads, scale):
    # PCs match a dense SVD of the centered (and, if `scale`, standardized)
    # matrix, up to sign and float32 error
    import polars as pl
    from scipy.sparse import diags_array, random_array
    from brisc import SingleCell
    rng = np.random.default_rng(0)
    X = random_array((500, 80), density=0.2, rng=rng, dtype=np.float32,
                     data_sampler=lambda size: rng.lognormal(size=size))
    X = (X @ diags_array(rng.uniform(0.2, 5, 80).astype(np.float32))).tocsr()  # unequal variances
    sc = SingleCell(X=X.tocsc() if csc else X.tocsr(),
                    obs=pl.DataFrame({"cell": [str(i) for i in range(500)]}),
                    var=pl.DataFrame({"gene": [str(i) for i in range(80)]}),
                    uns={"normalized": True})
    PCs = sc.pca(QC_column=None, hvg_column=None, num_PCs=5, scale=scale,
                 num_threads=num_threads).obsm["pca"]
    Z = X.toarray().astype(np.float64)
    Z -= Z.mean(axis=0)
    if scale:
        Z /= Z.std(axis=0, ddof=1)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    assert np.allclose(np.abs(PCs), np.abs(U[:, :5] * S[:5]), atol=1e-3 * S[0])


def test_pca_scale_type():
    import polars as pl
    from scipy.sparse import csr_array
    from brisc import SingleCell
    sc = SingleCell(X=csr_array(np.eye(10, dtype=np.float32)),
                    obs=pl.DataFrame({"cell": [str(i) for i in range(10)]}),
                    var=pl.DataFrame({"gene": [str(i) for i in range(10)]}),
                    uns={"normalized": True})
    with pytest.raises(TypeError):
        sc.pca(QC_column=None, hvg_column=None, num_PCs=2, scale=1)
