import os

import numpy as np
import pandas as pd

from .twinsemb import utils


def load_sne_measurements(file_path):
    names = [
        "sn",
        "salt2_x1",
        "salt2_x1_unc",
        "salt2_c",
        "salt2_c_unc",
        "delta_av",
        "delta_av_unc",
        "delta_m",
        "delta_m_unc",
        "xi1",
        "xi2",
        "xi3",
    ]
    dtypes = {key: np.float64 for key in names}
    dtypes["sn"] = str
    rbtl_data = pd.read_csv(
        file_path, sep=r"\s+", names=names, header=0, dtype=dtypes
    )
    return rbtl_data


def load_global_params(file_path):
    names = [
        "wavelength",
        "c1",
        "c2",
        "eps_neg_5",
        "eps_neg_2_5",
        "eps_2_5",
        "eps_5",
        "eta",
    ]
    dtypes = {key: np.float64 for key in names}
    dtypes["wavelength"] = np.int64
    global_params = pd.read_csv(
        file_path, sep=r"\s+", names=names, header=0, dtype=dtypes
    )
    return global_params


def load_flux(dir_path, sn_order):
    spectra = {}
    for spectrum_f_path in os.scandir(dir_path):
        sn = spectrum_f_path.name.split(".")[0]
        spectra[sn] = np.loadtxt(spectrum_f_path, skiprows=1)[None, ...]
    print(len(spectra))
    spectra = [spectra[sn] for sn in sn_order]
    print(len(spectra))
    spectra = np.concatenate(spectra, axis=0)
    return spectra[..., 0], spectra[..., 1], spectra[..., 2]


class TwinsEmbeddingDataset:
    def get_noise_mask(self, max_frac=0.1):
        intrinsic_dispersion = self.global_params["eta"].values
        intrinsic_power = np.sum(intrinsic_dispersion**2)
        maximum_uncertainty = utils.frac_to_mag(self.flux_err / self.flux)
        maximum_power = np.sum(maximum_uncertainty**2, axis=1)
        maximum_uncertainty_fraction = maximum_power / intrinsic_power
        return maximum_uncertainty_fraction < max_frac

    def get_extinction_mask(self, max_extinction=0.5):
        return np.abs(self.sne_measurements["delta_av"].values) < max_extinction

    def __init__(
        self,
        global_params_f_path,
        sne_measurements_f_path,
        spectra_dir_path,
        max_extinction_coefficient=0.5,
        max_uncertainty_fraction=0.1,
    ):
        self.global_params = load_global_params(global_params_f_path)
        self.sne_measurements = load_sne_measurements(sne_measurements_f_path)
        self.wavelengths, self.flux, self.flux_err = load_flux(
            spectra_dir_path, self.sne_measurements["sn"].values
        )
        self.noise_mask = self.get_noise_mask(max_uncertainty_fraction)
        self.extinction_mask = self.get_extinction_mask(
            max_extinction_coefficient
        )
        self.max_extinction_coefficient = max_extinction_coefficient
        self.max_uncertainty_fraction = max_uncertainty_fraction

    def _combine_noise_mask(self, apply_noise_mask, apply_extinction_mask):
        mask = np.ones(self.flux.shape[0], dtype=bool)
        if apply_noise_mask:
            mask &= self.noise_mask
        if apply_extinction_mask:
            mask &= self.extinction_mask
        return mask

    def get_spectra(self, apply_noise_mask=False, apply_extinction_mask=False):
        mask = self._combine_noise_mask(apply_noise_mask, apply_extinction_mask)
        return self.wavelengths[mask], self.flux[mask], self.flux_err[mask]

    def get_residuals_mask(self):
        return self.extinction_mask[self.noise_mask]

    def get_sne_param(
        self, param, apply_noise_mask=False, apply_extinction_mask=False
    ):
        mask = self._combine_noise_mask(apply_noise_mask, apply_extinction_mask)
        return self.sne_measurements[param].values[mask]

    def get_global_param(
        self, param, apply_noise_mask=False, apply_extinction_mask=False
    ):
        mask = self._combine_noise_mask(apply_noise_mask, apply_extinction_mask)
        return self.global_params[param].values[mask]

    def get_mean_spectrum(
        self, apply_noise_mask=False, apply_extinction_mask=False
    ):
        mask = self._combine_noise_mask(apply_noise_mask, apply_extinction_mask)
        return np.mean(self.flux[mask], axis=0)
