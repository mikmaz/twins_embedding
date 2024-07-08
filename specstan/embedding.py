import numpy as np
from sklearn.manifold import Isomap, TSNE
from umap import UMAP

from .math import nmad
from .plot import scatter_combined
from .twinsemb.manifold_gp import ManifoldGaussianProcess


def calculate_peculiar_velocity_uncertainties(
    redshifts=0.05, peculiar_velocity=300
):
    """Calculate dispersion added to the magnitude due to host galaxy
    peculiar velocity
    """
    pec_vel_dispersion = (5 / np.log(10)) * (
        peculiar_velocity / 3e5 / redshifts
    )

    return pec_vel_dispersion


class Embedding:
    def __init__(
        self,
        fluxes,
        num_components,
        magnitude_residuals,
        extinction_coefficients,
        mean_spectrum,
        n_jobs=-1,
        kind="umap",
        algorithm_kwargs=None,
        gp_mask=None,
    ):
        algorithm_map = {
            "umap": UMAP,
            "tsne": TSNE,
            "isomap": Isomap,
        }
        if kind not in algorithm_map:
            raise ValueError(f"Invalid embedding algorithm: {kind}")
        self.kind = kind
        self.algorithm_kwargs = algorithm_kwargs or {}
        if kind == "umap":
            self.algorithm_kwargs["n_components"] = num_components
        else:
            self.algorithm_kwargs["num_components"] = num_components
        self.algorithm_kwargs["n_jobs"] = n_jobs
        self.model = algorithm_map[kind](**self.algorithm_kwargs)

        self.fractional_differences = fluxes / mean_spectrum - 1
        self.embedding = self.model.fit_transform(self.fractional_differences)

        self.gp_mask = (
            np.ones(fluxes.shape[0], dtype=bool) if gp_mask is None else gp_mask
        )

        self.magnitude_residuals = magnitude_residuals
        self.extinction_coefficients = extinction_coefficients

        self.manifold_gp = None

    def _get_gp_data(self, kind="rbtl"):
        """Return the data needed for GP fits along with the corresponding
        masks.

        Parameters
        ----------
        kind : {'rbtl', 'salt', 'salt_raw'}
            The kind of magnitude data to return. The options are:
            - rbtl: RBTL magnitudes and colors.
            - salt: Corrected SALT2 magnitudes and colors.
            - salt_raw: Uncorrected SALT2 magnitudes and colors.

        Returns
        -------
        coordinates : numpy.array
            The coordinates to evaluate the GP over.
        mags : numpy.array
            A list of magnitudes for each supernova in the sample.
        mag_errs : numpy.array
            The uncertainties on the magnitudes. This only includes measurement
            uncertainties, not model ones (since the GP will handle that). Since
            we are dealing with high signal-to-noise light curves/spectra, the
            color and magnitude measurement errors are very small and difficult
            to propagate so I ignore them. This therefore only includes
            contributions from peculiar velocity.
        colors : numpy.array
            A list of colors for each supernova in the sample.
        condition_mask : numpy.array
            The mask that should be used for conditioning the GP.
        """
        if kind == "rbtl":
            mags = self.magnitude_residuals
            colors = self.extinction_coefficients
            condition_mask = self.gp_mask

            # Assume that we can ignore measurement uncertainties for the
            # magnitude errors, so the only contribution is from peculiar
            # velocities.
            mag_errs = calculate_peculiar_velocity_uncertainties(
                0.05 * np.ones_like(mags)
            )
        else:
            raise Exception("Unknown kind %s!" % kind)

        # Use the Isomap embedding for the GP coordinates.
        coordinates = self.embedding

        # If the analysis is blinded, only use the training data for
        # conditioning.
        # if self.settings['blinded']:
        #    condition_mask &= self.train_mask

        return coordinates, mags, mag_errs, colors, condition_mask

    def get_rbtl_residuals_stats(self):
        filtered_residuals = self.magnitude_residuals[self.gp_mask]
        return nmad(filtered_residuals, unbiased=False), np.std(
            filtered_residuals, ddof=1
        )

    def get_gp_residuals_stats(self):
        if self.manifold_gp is None:
            raise ValueError("Manifold GP has not been fit yet!")
        gp_residuals = self.manifold_gp.residuals[self.gp_mask]
        return nmad(gp_residuals, unbiased=False), np.std(gp_residuals, ddof=1)

    def fit_gp_magnitude_residuals(
        self, additional_covariates=None, verbosity=None
    ):
        """
        Calculate magnitude residuals using a GP over a given latent space.
        """
        if verbosity is None:
            verbosity = 1

        # Fit the hyperparameters on the full conditioning sample.
        coordinates, mags, mag_errs, colors, raw_mask = self._get_gp_data()

        # Build a list of linear covariates to use in the model that includes
        # the color and any user-specified covariates.
        covariates = [
            colors,
        ]
        additional_covariates = additional_covariates or []
        if additional_covariates:
            covariates.append(additional_covariates)

        covariates = np.vstack(covariates)

        # Apply the user-specified mask if one was given.
        if self.gp_mask is None:
            mask = raw_mask
        else:
            mask = self.gp_mask & raw_mask

        manifold_gp = ManifoldGaussianProcess(
            None,
            self.embedding,
            mags,
            mag_errs,
            covariates,
            mask,
        )

        manifold_gp.fit(verbosity=verbosity)

        self.manifold_gp = manifold_gp

        return self.get_gp_residuals_stats()

    def plot(self, save_f_path, masked=True, gp_corrected=False):
        if gp_corrected and self.manifold_gp is None:
            raise ValueError("Manifold GP has not been fit yet!")
        residuals = (
            self.magnitude_residuals
            if not gp_corrected
            else self.manifold_gp.residuals
        )
        embedding = self.embedding
        if masked:
            residuals = residuals[self.gp_mask]
            embedding = self.embedding[self.gp_mask]
        scatter_combined(embedding, residuals, save_f_path, vmin=-0.3, vmax=0.3)
