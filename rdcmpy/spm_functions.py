import numpy as np
from scipy.linalg import issymmetric, svd


def _spm_dcm_fmri_prior(
        spm_a: np.ndarray, spm_b: np.ndarray, spm_c: np.ndarray) -> tuple[np.ndarray, ...]:
    n_region = spm_a.shape[0]
    a_mat = np.array(spm_a > 0, dtype=int)
    a_mat = a_mat - np.diag(np.diag(a_mat))

    # prior expectations
    pe_a = a_mat / (64 * n_region) - np.eye(n_region) / 2
    pe_a = np.zeros(pe_a.shape) + np.diag(np.diag(pe_a))
    pe_b = spm_b * 0
    prior_mean = np.hstack((pe_a, pe_b.reshape(n_region, n_region * spm_c.shape[1]), spm_c * 0))

    # prior precisions
    pc_a = a_mat * 8 / n_region + np.eye(n_region) / (8 * n_region)
    pc_a = 1 / pc_a
    with np.errstate(divide='ignore'):
        pc_b = 1 / spm_b
    with np.errstate(divide='ignore'):
        pc_c = 1 / spm_c
    if np.any(pc_c[:, -1]):
        pc_c[:, -1] = 1e-8
    prior_precision = np.hstack((pc_a, pc_b.reshape(n_region, n_region * spm_c.shape[1]), pc_c))

    return prior_mean, prior_precision


def _spm_logdet(in_mat: np.ndarray) -> np.ndarray:
    tol = 1e-16
    indices = np.where((np.diag(in_mat) > tol) & (np.diag(in_mat) < 1/tol))[0]
    diag_mat = in_mat[np.ix_(indices, indices)]
    log_det = np.log(np.diag(diag_mat)).sum()

    if not issymmetric(diag_mat):
        long_dim = np.max(in_mat.shape)
        with np.errstate(divide='ignore'):
            log_det = log_det + np.log(np.linalg.det(diag_mat / np.exp(log_det / long_dim)))

    if not np.isreal(log_det) or np.isinf(log_det):
        svd_s = svd(diag_mat, compute_uv=False, lapack_driver='gesvd')
        indices = np.where((np.diag(svd_s) > tol) & (np.diag(svd_s) < 1 / tol))[0]
        log_det = np.log(svd_s[indices]).sum()

    return log_det
