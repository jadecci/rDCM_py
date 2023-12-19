from typing import Optional, Literal, Union
import time
import logging

import numpy as np
from scipy.fft import fft, ifft
from scipy.special import psi, gammaln

import rdcmpy.spm_functions as spm_func


logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('rdcmpy')
log.setLevel(level=logging.INFO)


class DimensionError(Exception):
    """Raised when dimension of a variable is wrong"""
    pass


class RegressionDCM:
    """regression Dynamic Causal Modelling (rDCM)"""
    def __init__(
            self, data: np.ndarray, t_rep: float, drive_input: Optional[np.ndarray] = None,
            method: Literal['original', 'sparse'] = 'original', prior_a: Optional[np.array] = None,
            endo_shift: int = 0, padding: bool = False, snr_thresh_std: Union[int, None] = 1,
            debug: bool = False) -> None:
        self._data = data
        self._t_rep = t_rep
        self._samp_rate = t_rep / 16
        self._n_datapoint, self._n_region = data.shape
        self._conv_length = self._n_datapoint * 16
        self._prior_a = prior_a

        if drive_input is not None:
            self._add_drive(drive_input)
            self._n_endo = self._drive_input.shape[1]
            self._conf = np.ones((self._conv_length, 1))
        else:
            self._drive_input = None
            self._n_endo = 1
            self._conf = np.zeros((self._conv_length, 0))

        if method not in ['original', 'sparse']:
            msg = "method must be 'original' or 'sparse'"
            log.error(msg)
            raise ValueError(msg)

        self._method = method
        self._endo_shift = endo_shift
        self._padding = padding
        self._snr_thresh_std = snr_thresh_std

        if debug:
            log.setLevel(level=logging.DEBUG)

    def _add_drive(self, drive_input: np.ndarray) -> None:
        """Add endogenous input to the DCM model"""
        if drive_input.shape[0] == self._data.shape[0] * 16:
            log.debug('endo_input with dimenions (16xN)xU assigned directly')
            self._drive_input = drive_input

        elif drive_input.shape[0] == self._data.shape[0]:
            log.debug(
                'endo_input with dimensions NxU will be repeated '
                'to match the correct microtime resolution')
            drive_input_up = np.zeros((drive_input.shape[0] * 16, drive_input.shape[1]))
            for nr_input in range(drive_input_up.shape[1]):
                drive_curr = drive_input[:, nr_input].T
                drive_curr = np.tile(drive_curr, (16, 1))
                drive_input_up[:, nr_input] = drive_curr.flatten()
            self._drive_input = drive_input_up

        else:
            msg = (
                'First dimenion of endo_input must be the same as that of data, '
                'or 16 times that of data')
            log.error(msg)
            raise DimensionError(msg)

    def _convolution_bm(self) -> None:
        """Create a fixed hemodynamic response function (HRF) by convolving a single event (impulse)
        with the standard Balloon model from DCM"""
        rho_hemo = 0.32  # resting oxygen extraction fraction (hemodynamic model)
        alpha = 0.32  # stiffness exponent
        tau0 = 2  # mean transit time
        gamma = 0.32  # rate of flow-dependent elimination
        kappa = 0.64  # rate of signal decay
        epsilon = 1  # neuronal efficacy
        theta0 = 40.3  # frequency offset
        t_echo = 0.04
        rho_bold = 0.4  # resting oxygen extraction fraction (BOLD signal model)
        r0 = 25  # relaxation slope
        v0 = 4  # resting venous volume

        endo_dummy = np.zeros(self._conv_length)
        endo_dummy[0] = 1  # single impulse (r_dt = 1)

        # Balloon hemodynamic model
        x = np.zeros(self._conv_length)  # input stimulus
        s = np.zeros(self._conv_length)  # flow-induced signal
        f1 = np.zeros(self._conv_length)  # in-flow
        f1[0] = 1
        f1_old = 0
        v1 = np.zeros(self._conv_length)  # normalised venous volume
        v1[0] = 1
        v1_old = 0
        q1 = np.zeros(self._conv_length)  # normalised total deoxyhemoglobin voxel content
        q1[0] = 1
        q1_old = 0

        for step in range(self._conv_length - 1):
            x[step+1] = x[step] + self._samp_rate * (endo_dummy[step] - x[step])
            s[step+1] = s[step] + self._samp_rate * (x[step] - kappa * s[step] - gamma * (f1[step] - 1))
            f1_old = f1_old + self._samp_rate * (s[step] / f1[step])
            f1[step+1] = np.exp(f1_old)
            v1_old = v1_old + self._samp_rate * ((f1[step] - v1[step] ** (1 / alpha)) / (tau0 * v1[step]))
            v1[step+1] = np.exp(v1_old)
            q1_old = q1_old + self._samp_rate * ((f1[step] * (1 - (1 - rho_hemo) ** (1 / f1[step])) / rho_hemo
                                                  - v1[step] ** (1/alpha) * q1[step] / v1[step])
                                                 / (tau0*q1[step]))
            q1[step+1] = np.exp(q1_old)

        # Balloon BOLD signal model
        k1 = 4.3 * theta0 * t_echo * rho_bold
        k2 = epsilon * r0 * t_echo * rho_bold
        self.hrf = v0 * (k1 * (1 - q1) + k2 * (1 - q1 / v1))  # k3 = 0

    def _filter(
            self, data: np.ndarray, drive: np.ndarray, hrf: np.ndarray) -> (np.ndarray, np.ndarray):
        """Specifies informative frequencies and filters the Fourier-transformed signal"""
        precision = 1e-4

        data_real = np.real(data)
        data_imag = np.imag(data)

        if self._drive_input is not None:
            drive_indices = (np.abs(drive).sum(axis=1) > precision)
            hpf = 16
        else:
            drive_indices = np.full(drive.shape[0], True)
            hpf = np.maximum(16 + (self._snr_thresh_std - 1) * 4, 16)
        freq = int(np.round(7 * data.shape[0] / hpf))
        freq_indices = np.concatenate((
            np.full((freq+1, data.shape[1]), True),
            np.full((data.shape[0]-freq*2-1, data.shape[1]), False),
            np.full((freq, data.shape[1]), True)))

        hrf_indices = (np.abs(hrf) > precision)
        if self._padding:
            data_indices = np.concatenate((
                np.full(np.round(self._n_datapoint / 2) + 1, True),
                np.full(self._n_datapoint - self._data.shape[0] - 1, False),
                np.full(self._n_datapoint / 2, True)))
        else:
            data_indices = np.full(self._n_datapoint, True)
        noise_indices = np.array(
            ~np.logical_and(np.logical_and(hrf_indices, drive_indices), data_indices))

        if noise_indices.sum() > 1:
            std_real = np.tile(np.std(data_real[noise_indices, :]), (data.shape[0], 1))
            std_imag = np.tile(np.std(data_imag[noise_indices, :]), (data.shape[0], 1))
        else:
            std_real = np.zeros(data.shape)
            std_imag = np.zeros(data.shape)
        snr_indices = np.logical_or(
            (np.abs(data_real) > self._snr_thresh_std * std_real),
            (np.abs(data_imag) > self._snr_thresh_std * std_imag))

        hpf_indices = np.logical_and(
            np.logical_and(snr_indices, np.tile(
                ~noise_indices.reshape(noise_indices.shape[0], 1), (1, data.shape[1]))),
            freq_indices)
        hpf_indices[0, :] = 1  # constant frequency
        hpf_indices_flip = hpf_indices.copy()
        hpf_indices_flip[1:, :] = np.flipud(hpf_indices_flip[1:, :])
        hpf_indices = np.logical_or(hpf_indices, hpf_indices_flip)  # symmetry

        # include everything except padding for informative regions
        data[~hpf_indices] = 0
        for region in range(self._n_region):
            freq1 = hpf_indices[0:int(np.round(data.shape[0]/2)), region].nonzero()[0][-1]
            freq2 = (hpf_indices[int(np.round(data.shape[0]/2)):-1, region].nonzero()[0][0]+
                     int(np.round(data.shape[0]/2)))
            hpf_indices[0:freq1, region] = True
            hpf_indices[freq2:-1, region] = True

        return data, hpf_indices

    @staticmethod
    def _reduce_zeros(design_mat: np.ndarray, data: np.ndarray) -> np.ndarray:
        """If there are more zero-valued frequencies than informative ones,
        subsample those frequencies to balance dataset"""
        data_all = np.abs(np.hstack((design_mat, data))).sum(axis=1)
        freq0_indices = np.where(data_all == 0)[0]
        n_zero = np.sum(data_all == 0)
        n_inform = np.sum(data_all > 0)
        if n_zero > n_inform:
            del_indices = np.concatenate((np.full(n_inform, False), np.full(n_zero-n_inform, True)))
            del_indices = np.random.permutation(del_indices)
            data[freq0_indices[del_indices], :] = np.nan

        return data

    def _design_matrix(self) -> (np.ndarray, np.ndarray):
        """Transform initial DCM signals into a set of regressors X (design matrix) and Y (data)"""
        hrf_fft = fft(self.hrf, axis=0, norm='backward')
        data_fft = fft(self._data, axis=0, norm='backward')

        if self._drive_input is not None:
            drive_input = np.roll(self._drive_input, self._endo_shift, axis=0)
        else:
            drive_input = np.zeros((self._n_datapoint * 16, 1))
        drive_input = ifft(
            fft(drive_input, axis=0, norm='backward') *
            np.tile(hrf_fft.reshape(self._conv_length, 1), (1, self._n_endo)))
        drive_input = np.hstack((drive_input, self._conf))

        if self._padding:
            break_point = np.round(self._n_datapoint / 2)
            data_fft[break_point, :] = data_fft[break_point, :] / 2
            data_fft = 16 * np.vstack((
                data_fft[0:(break_point + 1), :],
                np.zeros((self._conv_length - self._n_datapoint - 1, self._n_region)),
                data_fft[break_point:-1, :]))
            self._n_datapoint = data_fft.shape[0]
            drive_fft = fft(drive_input / 16, axis=0, norm='backward')

        else:
            hrf_fft = fft(self.hrf[0:self._conv_length:16], axis=0, norm='backward')
            drive_fft = fft(drive_input[0:self._conv_length:16, :], axis=0, norm='backward')

        if self._snr_thresh_std is not None:
            data_fft, filter_indices = self._filter(data_fft, drive_fft, hrf_fft)
        else:
            filter_indices = np.ones(data_fft.shape)

        deriv_coef = np.exp(2 * np.pi * 1j * np.arange(self._n_datapoint) / self._n_datapoint) - 1
        data_deriv = np.tile(
            deriv_coef.reshape(self._n_datapoint, 1), (1, self._n_region)) * data_fft / self._t_rep
        data_deriv[~filter_indices] = np.nan

        bilinear_term = np.zeros((
            self._n_datapoint, self._n_region * (drive_fft.shape[1] + self._conf.shape[1])))
        design_mat = np.hstack((data_fft, bilinear_term, drive_fft))
        data_mat = self._reduce_zeros(design_mat, data_deriv)

        return design_mat, data_mat

    @staticmethod
    def _prior(
            spm_a: np.ndarray, spm_b: np.ndarray,
            spm_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Get the prior parameters on model parameters (theta) and noise precision (tau)
        for the particular connectivity pattern"""
        prior_mean, prior_prec = spm_func.spm_dcm_fmri_prior(spm_a, spm_b, spm_c)
        prior_a0 = 2
        prior_b0 = 1

        return prior_mean, prior_prec, prior_a0, prior_b0

    def _ridge(self, design_mat: np.ndarray, data_mat: np.ndarray) -> None:
        """Variational Bayesian inversion of a linear DCM with regression DCM.
        The function implements the VB udpate equations derived in Frässele et al. 2017."""
        precision = 1e-5

        if self._prior_a is not None:
            prior_a = self._prior_a
        else:
            prior_a = np.ones((self._n_region, self._n_region))
        prior_b = np.zeros((self._n_region, self._n_region, self._n_endo))
        prior_c = np.ones((self._n_region, self._n_endo))
        if self._drive_input is None: # resting-state
            prior_c = prior_c * 0
        for _ in range(self._conf.shape[1]):
            prior_b = np.dstack((prior_b, prior_b[:, :, 0]))
            prior_c = np.hstack((prior_c, np.ones((prior_c.shape[0], 1))))

        indices = (np.hstack((
            prior_a, prior_b.reshape(self._n_region, self._n_region * prior_c.shape[1]), prior_c)) > 0)
        prior_mean, prior_prec, prior_a0, prior_b0 = self._prior(prior_a, prior_b, prior_c)

        mean_all = np.zeros(indices.shape)
        cov = np.zeros((self._n_region, indices.shape[1], indices.shape[1]))
        alpha = np.zeros(self._n_region)
        beta = np.zeros(self._n_region)
        free_energy = np.zeros(self._n_region)

        for region in range(self._n_region):
            log.debug('Estimating parameters for region %i ...', region)
            data_indices = ~np.isnan(data_mat[:, region])
            design_region = design_mat[np.ix_(data_indices, indices[region, :])]
            data_region = data_mat[data_indices, region]
            n_effective = data_indices.sum() / 16
            dim_effective = indices[region, :].sum()
            pp_region = np.diag(prior_prec[region, indices[region, :]])  # precision
            pm_region = prior_mean[region, indices[region, :]]  # mean

            xtx = design_region.conj().T @ design_region
            xty = design_region.conj().T @ data_region

            tau_region = prior_a0 / prior_b0
            alpha_region = prior_a0 + n_effective / (2 * 16)
            free_energy_old = -np.inf

            for i in range(500):
                cov_region = np.linalg.inv(tau_region * xtx + pp_region)
                mean_region = cov_region @ (tau_region * xty + pp_region @ pm_region)
                post_rate = (
                        (data_region - design_region @ mean_region).conj().T @
                        (data_region - design_region @ mean_region) / 2 +
                        np.trace(xtx @ cov_region) / 2)
                beta_region = prior_b0 + post_rate
                tau_region = alpha_region / beta_region

                log_like = (
                        n_effective * (psi(alpha_region) - np.log(beta_region)) / 2 -
                        n_effective * np.log(2*np.pi) / 2 - post_rate * tau_region)
                log_p_weight = (
                        1 / 2 * spm_func.spm_logdet(pp_region) -
                        dim_effective * np.log(2*np.pi) / 2 -
                        (mean_region - pm_region).T @ pp_region @ (mean_region - pm_region) / 2 -
                        np.trace(pp_region @ cov_region) / 2)
                log_p_prec = (
                    prior_a0 * np.log(prior_b0) - gammaln(prior_a0) +
                    (prior_a0 - 1) * (psi(alpha_region) - np.log(beta_region)) -
                    prior_b0 * tau_region)
                log_q_weight = (
                        1 / 2 * spm_func.spm_logdet(cov_region) +
                        dim_effective * (1 + np.log(2*np.pi)) / 2)
                log_q_prec = (
                    alpha_region - np.log(beta_region) + gammaln(alpha_region) +
                    (1 - alpha_region) * psi(alpha_region))

                # check convergence
                free_energy_curr = (
                        log_like + log_p_weight + log_p_prec + log_q_weight + log_q_prec)
                log.debug('Iteration %i logF = %.4f', i, np.real(free_energy_curr))
                free_energy_diff = np.power(free_energy_old - np.real(free_energy_curr), 2)
                if free_energy_diff < np.power(precision, 2):
                    # store parameters
                    indices_curr = indices[region, :]
                    free_energy[region] = np.real(free_energy_curr)
                    mean_all[region, indices_curr] = np.real(mean_region)
                    cov[np.ix_([region], indices_curr, indices_curr)] = np.real(cov_region)
                    alpha[region] = np.real(alpha_region)
                    beta[region] = np.real(beta_region)

                    break

                free_energy_old = np.real(free_energy_curr)

        self._aggregate_params(
            prior_mean, prior_prec, prior_a0, prior_b0, mean_all, cov, alpha, beta, free_energy)

    def _aggregate_params(
            self, prior_mean: np.ndarray, prior_prec: np.ndarray, prior_a0: float, prior_b0: float,
            mean_all: np.ndarray, cov: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
            free_energy: np.ndarray) -> None:
        """Gather parameters in one output structure"""
        # TODO: check if min(size(indices)) == 1 ??
        indices_a = np.ix_(range(self._n_region), range(self._n_region))
        indices_b = np.ix_(
            range(self._n_region),
            range(self._n_region, (self._n_region + self._n_region * self._n_endo)))
        indices_c = np.ix_(
            range(self._n_region),
            range(
                self._n_region + self._n_region * (self._n_endo + self._conf.shape[1]),
                mean_all.shape[1] - self._conf.shape[1]))
        indices_baseline = np.ix_(
            range(self._n_region), range(mean_all.shape[1] - self._conf.shape[1], mean_all.shape[1]))

        # TODO: connection probabilities ??
        # TODO: components of free energy ??

        self.priors = {
            'mu_mean': prior_mean,
            'mu_covariance': prior_prec,
            'tau_alpha': prior_a0,
            'tau_beta': prior_b0}

        self.params = {
            'mu_connectivity': mean_all[indices_a],
            'mu_b': mean_all[indices_b].reshape(self._n_region, self._n_region, self._n_endo),
            'mu_driving_input': mean_all[indices_c] * 16,
            'mu_baseline': mean_all[indices_baseline],
            'mu_covariance': cov,
            'tau_alpha': alpha,
            'tau_beta': beta,
            'free_energy': free_energy.sum(),
            'regionwise_free_energy': free_energy}

    def estimate(self) -> None:
        """Estimate parameters of a DCM model"""
        t_start = time.time()
        log.info('Starting rDCM ...')
        # TODO: what is the use of DCM.M (model)??

        self._convolution_bm()
        design_mat, data_mat = self._design_matrix()

        log.info('Running model inversion ...')
        if self._method == 'original':
            self._ridge(design_mat, data_mat)

        else:
            # TODO: sparse model
            log.debug('not implemented yet')

        log.info('Finished estimation')
        t_end = time.time()
        log.info('Time taken: %s', time.strftime('%H:%M:%S', time.gmtime(t_end - t_start)))

    def get_priors(self) -> dict:
        return self.priors

    def get_params(self) -> dict:
        return self.params
