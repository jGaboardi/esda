"""
A-DBSCAN implementation
"""

__author__ = "Dani Arribas-Bel <daniel.arribas.bel@gmail.com>"

import warnings
from collections import Counter

import numpy as np
import pandas
from libpysal.cg.alpha_shapes import alpha_shape_auto
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import ClusterMixin as _ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["ADBSCAN", "remap_lbls", "ensemble", "get_cluster_boundary"]


class ADBSCAN(_ClusterMixin, _BaseEstimator):
    """
    A-DBSCAN, as introduced in :cite:`ab_gl_vm2020joue`.

    A-DSBCAN is an extension of the original DBSCAN algorithm that creates an
    ensemble of solutions generated by running DBSCAN on a random subset and
    "extending" the solution to the rest of the sample through
    nearest-neighbor regression.

    See the original reference (:cite:`ab_gl_vm2020joue`) for more details or
    the notebook guide for an illustration.
    ...

    Parameters
    ----------
    eps : float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int
        The number of samples (or total weight) in a neighborhood
        for a point to be considered as a core point. This includes the
        point itself.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    n_jobs : int
        [Optional. Default=1] The number of parallel jobs to run. If -1, then
        the number of jobs is set to the number of CPU cores.
    pct_exact : float
        [Optional. Default=0.1] Proportion of the entire dataset
        used to calculate DBSCAN in each draw
    reps : int
        [Optional. Default=100] Number of random samples to draw in order to
        build final solution
    keep_solus : bool
        [Optional. Default=False] If True, the `solus` and `solus_relabelled`
        objects are kept, else it is deleted to save memory
    pct_thr : float
        [Optional. Default=0.9] Minimum proportion of replications that
        a non-noise label need to be assigned to an observation for that
        observation to be labelled as such

    Attributes
    ----------
    labels_ : array
        [Only available after `fit`] Cluster labels for each point in the
        dataset given to fit().
        Noisy (if the proportion of the most common label is < pct_thr)
        samples are given the label -1.
    votes : DataFrame
        [Only available after `fit`] Table indexed on `X.index` with
        `labels_` under the `lbls` column, and the frequency across draws of
        that label under `pct`
    solus : DataFrame, shape = [n, reps]
        [Only available after `fit`] Each solution of labels for every draw
    solus_relabelled : DataFrame, shape = [n, reps]
        [Only available after `fit`] Each solution of labels for
        every draw, relabelled to be consistent across solutions

    Examples
    --------
    >>> import pandas
    >>> from esda.adbscan import ADBSCAN
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> db = pandas.DataFrame({'X': np.random.random(25), \
                               'Y': np.random.random(25) \
                              })

    ADBSCAN can be run following scikit-learn like API as:

    >>> np.random.seed(10)
    >>> clusterer = ADBSCAN(0.03, 3, reps=10, keep_solus=True)
    >>> _ = clusterer.fit(db)
    >>> clusterer.labels_
    array(['-1', '-1', '-1', '0', '-1', '-1', '-1', '0', '-1', '-1', '-1',
           '-1', '-1', '-1', '0', '0', '0', '-1', '0', '-1', '0', '-1', '-1',
           '-1', '-1'], dtype=object)

    We can inspect the winning label for each observation, as well as the
    proportion of votes:

    >>> print(clusterer.votes.head().to_string())
      lbls  pct
    0   -1  0.7
    1   -1  0.5
    2   -1  0.7
    3    0  1.0
    4   -1  0.7

    If you have set the option to keep them, you can even inspect each
    solution that makes up the ensemble:

    >>> print(clusterer.solus.head().to_string())
      rep-00 rep-01 rep-02 rep-03 rep-04 rep-05 rep-06 rep-07 rep-08 rep-09
    0      0      1      1      0      1      0      0      0      1      0
    1      1      1      1      1      0      1      0      1      1      1
    2      0      1      1      0      0      1      0      0      1      0
    3      0      1      1      0      0      1      1      1      0      0
    4      0      1      1      1      0      1      0      1      0      1



    If we select only one replication and the proportion of the entire dataset
    that is sampled to 100%, we obtain a traditional DBSCAN:

    >>> clusterer = ADBSCAN(0.2, 5, reps=1, pct_exact=1)
    >>> np.random.seed(10)
    >>> _ = clusterer.fit(db)
    >>> clusterer.labels_
    array(['0', '-1', '0', '0', '0', '-1', '-1', '0', '-1', '-1', '0', '-1',
           '-1', '-1', '0', '0', '0', '-1', '0', '0', '0', '-1', '-1', '0',
           '-1'], dtype=object)

    """

    def __init__(
        self,
        eps,
        min_samples,
        algorithm="auto",
        n_jobs=1,
        pct_exact=0.1,
        reps=100,
        keep_solus=False,
        pct_thr=0.9,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.reps = reps
        self.n_jobs = n_jobs
        self.pct_exact = pct_exact
        self.pct_thr = pct_thr
        self.keep_solus = keep_solus

    def fit(self, X, y=None, sample_weight=None, xy=["X", "Y"]):  # noqa: ARG002
        """
        Perform ADBSCAN clustering from fetaures
        ...

        Parameters
        ----------
        X               : DataFrame
                          Features
        sample_weight   : Series, shape (n_samples,)
                          [Optional. Default=None] Weight of each sample, such
                          that a sample with a weight of at least ``min_samples``
                          is by itself a core sample; a sample with negative
                          weight may inhibit its eps-neighbor from being core.
                          Note that weights are absolute, and default to 1.
        xy              : list
                          [Default=`['X', 'Y']`] Ordered pair of names for XY
                          coordinates in `xys`
        y               : Ignored
        """
        n = X.shape[0]
        zfiller = len(str(self.reps))
        solus = pandas.DataFrame(
            np.zeros((X.shape[0], self.reps), dtype=str),
            index=X.index,
            columns=[f"rep-{str(i).zfill(zfiller)}" for i in range(self.reps)],
        )
        # Multi-core implementation of parallel draws
        if (self.n_jobs == -1) or (self.n_jobs > 1):
            # Set different parallel seeds!!!
            warnings.warn(
                "Multi-core implementation only works on relabelling solutions. "
                "Execution of draws is still sequential.",
                UserWarning,
                stacklevel=2,
            )
            for i in range(self.reps):
                pars = (
                    n,
                    X,
                    sample_weight,
                    xy,
                    self.pct_exact,
                    self.eps,
                    self.min_samples,
                    self.algorithm,
                    self.n_jobs,
                )
                lbls_pred = _one_draw(pars)
                solus.iloc[:, i] = lbls_pred
        else:
            for i in range(self.reps):
                pars = (
                    n,
                    X,
                    sample_weight,
                    xy,
                    self.pct_exact,
                    self.eps,
                    self.min_samples,
                    self.algorithm,
                    self.n_jobs,
                )
                lbls_pred = _one_draw(pars)
                solus.iloc[:, i] = lbls_pred

        solus_relabelled = remap_lbls(solus, X, xy=xy, n_jobs=self.n_jobs)
        self.votes = ensemble(solus_relabelled)
        lbls = self.votes["lbls"].values
        lbl_type = type(solus.iloc[0, 0])
        lbls[self.votes["pct"] < self.pct_thr] = lbl_type(-1)
        self.labels_ = lbls
        if not self.keep_solus:
            del solus
            del solus_relabelled
        else:
            self.solus = solus
            self.solus_relabelled = solus_relabelled
        return self


def _one_draw(pars):
    n, X, sample_weight, xy, pct_exact, eps, min_samples, algorithm, n_jobs = pars
    rids = np.arange(n)
    np.random.shuffle(rids)
    rids = rids[: int(n * pct_exact)]

    X_thin = X.iloc[rids, :]

    thin_sample_weight = None
    if sample_weight is not None:
        thin_sample_weight = sample_weight.iloc[rids]

    min_samples = min_samples * pct_exact
    min_samples = 1 if min_samples < 1 else int(np.floor(min_samples))

    dbs = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm=algorithm,
        n_jobs=n_jobs,
    ).fit(X_thin[xy], sample_weight=thin_sample_weight)
    lbls_thin = pandas.Series(dbs.labels_.astype(str), index=X_thin.index)

    NR = KNeighborsClassifier(n_neighbors=1)
    NR.fit(X_thin[xy], lbls_thin)
    lbls_pred = pandas.Series(NR.predict(X[xy]), index=X.index)
    return lbls_pred


def remap_lbls(solus, xys, xy=["X", "Y"], n_jobs=1):
    """
    Remap labels in solutions so they are comparable (same label
    for same cluster)
    ...

    Parameters
    ----------
    solus       : DataFrame
                  Table with labels for each point (row) and solution (column)
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    n_jobs      : int
                  [Optional. Default=1] The number of parallel jobs to run. If
                  -1, then the number of jobs is set to the number of CPU
                  cores.

    Returns
    -------
    onel_solus  : DataFrame
                  Table with original solutions remapped to consolidated
                  labels across all the columns

    Examples
    --------

    >>> import pandas
    >>> db = pandas.DataFrame({"X": [0, 0.1, 4, 6, 5], \
                               "Y": [0, 0.2, 5, 7, 5] \
                              })
    >>> solus = pandas.DataFrame({"rep-00": [0, 0, 7, 7, -1], \
                                  "rep-01": [4, 4, -1, 6, 6], \
                                  "rep-02": [5, 5, 8, 8, 8] \
                                 })
    >>> print(remap_lbls(solus, db).to_string())
       rep-00  rep-01  rep-02
    0       0       0       0
    1       0       0       0
    2       7      -1       7
    3       7       7       7
    4      -1       7       7
    """
    # N. of clusters by solution
    ns_clusters = solus.apply(lambda x: x.unique().shape[0])
    # Pick reference solution as one w/ max N. of clusters
    ref = ns_clusters[ns_clusters == ns_clusters.max()].iloc[[0]].index[0]
    lbl_type = type(solus[ref].iloc[0])
    # Obtain centroids of reference solution
    ref_centroids = (
        xys.groupby(solus[ref])[xy]
        .apply(lambda xys: xys.mean())
        .drop(lbl_type(-1), errors="ignore")
    )
    # Only continue if any solution
    if ref_centroids.shape[0] > 0:
        # Build KDTree and setup results holder
        ref_kdt = cKDTree(ref_centroids)
        remapped_solus = pandas.DataFrame(
            np.zeros(solus.shape, dtype=lbl_type),
            index=solus.index,
            columns=solus.columns,
        )
        if (n_jobs == -1) or (n_jobs > 1):
            pool = _setup_pool(n_jobs)
            s_ids = solus.drop(ref, axis=1).columns.tolist()
            to_loop_over = [(solus[s], ref_centroids, ref_kdt, xys, xy) for s in s_ids]
            remapped = pool.map(_remap_n_expand, to_loop_over)
            remapped_df = pandas.concat(remapped, axis=1)
            remapped_solus.loc[:, s_ids] = remapped_df
        else:
            for s in solus.drop(ref, axis=1):
                # -
                pars = (solus[s], ref_centroids, ref_kdt, xys, xy)
                remap_ids = _remap_lbls_single(pars)
                # -
                remapped_solus.loc[:, s] = solus[s].map(remap_ids)
        remapped_solus.loc[:, ref] = solus.loc[:, ref]
        return remapped_solus.fillna(lbl_type(-1)).astype(lbl_type)
    else:
        warnings.warn("No clusters identified.", UserWarning, stacklevel=2)
        return solus


def _remap_n_expand(pars):
    solus_s, ref_centroids, ref_kdt, xys, xy = pars
    remap_ids = _remap_lbls_single(pars)
    expanded = solus_s.map(remap_ids)
    return expanded


def _remap_lbls_single(pars):
    new_lbls, ref_centroids, ref_kdt, xys, xy = pars
    lbl_type = type(ref_centroids.index[0])
    # Cross-walk to cluster IDs
    ref_centroids_ids = pandas.Series(ref_centroids.index.values)
    # Centroids for new solution
    solu_centroids = (
        xys.groupby(new_lbls)[xy]
        .apply(lambda xys: xys.mean())
        .drop(lbl_type(-1), errors="ignore")
    )
    # Remapping from old to new labels
    _, nrst_ref_cl = ref_kdt.query(solu_centroids.values)
    remap_ids = pandas.Series(nrst_ref_cl, index=solu_centroids.index).map(
        ref_centroids_ids
    )
    return remap_ids


def ensemble(solus_relabelled):
    """
    Generate unique class prediction based on majority/hard voting
    ...

    Parameters
    ----------
    solus_relabelled  : DataFrame
                        Table with labels for each point (row) and solution
                        (column). Labels are assumed to be consistent across
                        solutions.

    Returns
    -------
    pred              : DataFrame
                        Table with one row per observation, a `lbls` column with the
                        winning label, and a `pct` column with the proportion of
                        times the winning label was voted

    Examples
    --------

    >>> import pandas
    >>> db = pandas.DataFrame({"X": [0, 0.1, 4, 6, 5], \
                               "Y": [0, 0.2, 5, 7, 5] \
                              })
    >>> solus = pandas.DataFrame({"rep-00": [0, 0, 7, 7, -1], \
                                  "rep-01": [4, 4, -1, 6, 6], \
                                  "rep-02": [5, 5, 8, 8, 8] \
                                 })
    >>> solus_rl = remap_lbls(solus, db)
    >>> print(round(ensemble(solus_rl), 2).to_string())
       lbls   pct
    0     0  1.00
    1     0  1.00
    2     7  0.67
    3     7  1.00
    4     7  0.67

    """

    counts = np.array(
        list(  # noqa: C417
            map(lambda a: Counter(a).most_common(1)[0], solus_relabelled.values)
        )
    )
    winner = counts[:, 0]
    votes = counts[:, 1].astype(int) / solus_relabelled.shape[1]
    pred = pandas.DataFrame(
        {"lbls": winner, "pct": votes}, index=solus_relabelled.index
    )
    return pred


def _setup_pool(n_jobs):
    """
    Set pool for multiprocessing
    ...

    Parameters
    ----------
    n_jobs      : int
                  The number of parallel jobs to run. If -1, then the number
                  of jobs is set to the number of CPU cores.
    """
    import multiprocessing as mp

    return mp.Pool(mp.cpu_count()) if n_jobs == -1 else mp.Pool(n_jobs)


def get_cluster_boundary(labels, xys, xy=["X", "Y"], n_jobs=1, crs=None, step=1):
    """
    Turn a set of labels associated with 2-D points into polygon boundaries
    for each cluster using the auto alpha shape algorithm
    (`libpysal.cg.alpha_shapes.alpha_shape_auto`)
    ...

    Parameters
    ----------
    labels      : Series
                  Cluster labels for each point in the dataset (noise
                  samples expressed as -1), indexed as `xys`
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    n_jobs      : int
                  [Optional. Default=1] The number of parallel jobs to run
                  for remapping. If -1, then the number of jobs is set to
                  the number of CPU cores.
    crs         : str
                  [Optional] Coordinate system
    step        : int
                  [Optional. Default=1]
                  Number of points in `xys` to jump ahead in the alpha
                  shape stage after checking whether the largest possible
                  alpha that includes the point and all the other ones
                  with smaller radii

    Returns
    -------
    polys       : GeoSeries
                  GeoSeries with polygons for each cluster boundary, indexed
                  on the cluster label

    Examples
    --------
    >>> import pandas
    >>> from esda.adbscan import ADBSCAN, get_cluster_boundary
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> db = pandas.DataFrame({'X': np.random.random(25), \
                               'Y': np.random.random(25) \
                              })

    ADBSCAN can be run following scikit-learn like API as:

    >>> np.random.seed(10)
    >>> clusterer = ADBSCAN(0.03, 3, reps=10, keep_solus=True)
    >>> _ = clusterer.fit(db)
    >>> labels = pandas.Series(clusterer.labels_, index=db.index)
    >>> polys = get_cluster_boundary(labels, db)
    >>> polys[0].wkt
    'POLYGON ((0.7217553174317995 0.8192869956700687, 0.7605307121989587 0.9086488808086682, 0.9177741225129434 0.8568503024577332, 0.8126209616521135 0.6262871483113925, 0.6125260668293881 0.5475861559192435, 0.5425443680112613 0.7546476915298572, 0.7217553174317995 0.8192869956700687))'
    """  # noqa: E501

    try:
        from geopandas import GeoSeries
    except ModuleNotFoundError:

        def GeoSeries(data, index=None, crs=None):  # noqa: ARG001, N802
            return list(data)

    lbl_type = type(labels.iloc[0])
    noise = lbl_type(-1)
    ids_in_cluster = labels[labels != noise].index
    g = xys.loc[ids_in_cluster, xy].groupby(labels[ids_in_cluster])
    chunked_pts_step = []
    cluster_lbls = []
    for sub in g.groups:
        chunked_pts_step.append((xys.loc[g.groups[sub], xy].values, step))
        cluster_lbls.append(sub)
    if n_jobs == 1:
        polys = map(_asa, chunked_pts_step)
    else:
        pool = _setup_pool(n_jobs)
        polys = pool.map(_asa, chunked_pts_step)
        pool.close()
    polys = GeoSeries(polys, index=cluster_lbls, crs=crs)
    return polys


def _asa(pts_s):
    return alpha_shape_auto(pts_s[0], step=pts_s[1])
