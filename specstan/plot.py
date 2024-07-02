import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

default_settings = {# Is the analysis being blinded?
    "blinded": False,

    # Location of the SNfactory dataset.
    "idr_directory": "/home/kyle/data/snfactory/idr/",

    # Data release to use.
    "idr": "BLACKSTON",

    # Range of phases to allow for spectra in the analysis in restframe days.
    "phase_range": 5.,

    # Bin the spectrum with equally space bins in velocity before running the analysis.
    "bin_min_wavelength": 3300., "bin_max_wavelength": 8600.,
    "bin_velocity": 1000.,

    # Verbosity.
    # 0 = suppress most output
    # 1 = normal output
    # 2 = debug
    "verbosity": 1,

    # Cut on signal-to-noise
    "s2n_cut_min_wavelength": 3300, "s2n_cut_max_wavelength": 3800,
    "s2n_cut_threshold": 100,

    # Parameters for the differential evolution model used to model spectra at maximum
    # light.
    "differential_evolution_num_phase_coefficients": 4,
    "differential_evolution_use_salt_x1": False,

    # Parameters for the read between the lines algorithm.
    "rbtl_fiducial_rv": 2.8,

    # For the manifold learning analysis, we reject spectra with too large of
    # uncertainties on the estimates of their spectra at maximum light. This sets the
    # threshold for "too large" as the ratio of total variance of the spectrum at
    # maximum light to the total intrinsic variance of Type Ia supernovae from the RBTL
    # analysis.
    "mask_uncertainty_fraction": 0.1,

    # Parameters for the Isomap algorithm
    "isomap_num_neighbors": 10, "isomap_num_components": 3,

    # The signs of Isomap components are arbitrary. Choose to flip some of them so that
    # they match up nicely with previously established observables.
    "isomap_flip_components": [1],

    # Peculiar velocity (in km/s)
    "peculiar_velocity": 300,

    # Figure parameters

    # Directory to save figures to
    "figure_directory": "./figures/",

    # Matplotlib settings for all figures.
    "matplotlib_settings": {"figure.figsize": (5., 4.),
        "figure.constrained_layout.use": True,
        "figure.max_open_warning": 1000, },

    # Colormap to use
    "colormap": plt.cm.coolwarm,

    # Size of full-page spectra figures
    "spectrum_plot_figsize": (9., 2.8),
    "spectrum_plot_figsize_double": (9., 5.5),
    "spectrum_plot_figsize_triple": (9., 8.),

    # Width of full-page combined component scatter plots
    "combined_scatter_plot_width": 7., "combined_scatter_plot_marker_size": 50.,

    # Scatter plot properties
    "scatter_plot_marker_size": 70.,

    # Choose how to plot spectra. Options are "f_nu" or "f_lambda". In this analysis, we
    # do everything in F_lambda, but plots of SNe Ia look a lot better in F_nu because
    # the overall spectrum is flatter so we do that by default. Note that the overall
    # scale of our spectra is arbitrary.
    "spectrum_plot_format": "f_nu",

    # Default labels for spectrum plots
    "spectrum_plot_xlabel": "Wavelength ($\\AA$)",
    "spectrum_plot_ylabel": "Normalized flux\n(erg/$cm^2$/s/Hz)",

    # Directory to save LaTeX output to
    "latex_directory": "./latex/",

    # Different tests to run
    "test_no_interpolation": False, }


def scatter_combined(embedding, variable, save_path, mask=None, label=None,
                     axis_1=0, axis_2=1, axis_3=2, vmin=None, vmax=None,
                     cmap=None, discrete_color_map=None, invert_colorbar=False,
                     **kwargs):
    """Scatter plot that shows three components simultaneously while preserving
    aspect ratios.

    The height of the figure will be adjusted automatically to produce the right
    aspect ratio.
    """
    use_embedding = embedding

    if np.ndim(variable) == 2:
        c12 = variable[0]
        c13 = variable[1]
        c32 = variable[2]
    else:
        c12 = c13 = c32 = variable

    if mask is not None:
        use_embedding = use_embedding[mask]
        c12 = c12[mask]
        c13 = c13[mask]
        c32 = c32[mask]

    if discrete_color_map is not None:
        cmap = ListedColormap(discrete_color_map.values())
        color_id_map = {j: i for i, j in enumerate(discrete_color_map)}
        c12 = [color_id_map[i] for i in c12]
        c13 = [color_id_map[i] for i in c13]
        c32 = [color_id_map[i] for i in c32]
    else:
        if cmap is None:
            cmap = default_settings['colormap']

        if invert_colorbar:
            cmap = cmap.reversed()

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []

        c12 = sm.to_rgba(c12)
        c13 = sm.to_rgba(c13)
        c32 = sm.to_rgba(c32)

    min_1 = np.min(use_embedding[:, axis_1])
    max_1 = np.max(use_embedding[:, axis_1])
    min_2 = np.min(use_embedding[:, axis_2])
    max_2 = np.max(use_embedding[:, axis_2])
    min_3 = np.min(use_embedding[:, axis_3])
    max_3 = np.max(use_embedding[:, axis_3])

    range_1 = max_1 - min_1
    range_2 = max_2 - min_2
    range_3 = max_3 - min_3

    border = 0.1

    min_1 -= border * range_1
    max_1 += border * range_1
    min_2 -= border * range_2
    max_2 += border * range_2
    min_3 -= border * range_3
    max_3 += border * range_3

    range_1 *= (1 + 2. * border)
    range_2 *= (1 + 2. * border)
    range_3 *= (1 + 2. * border)

    if discrete_color_map:
        # Don't show a colorbar
        ncols = 2
        width_ratios = [range_1, range_3]
    else:
        # Add axes for a colorbar
        colorbar_frac = 0.025

        plot_width = 1 - colorbar_frac
        width_1 = plot_width * range_1 / (range_1 + range_3)
        width_3 = plot_width * range_3 / (range_1 + range_3)

        ncols = 3
        width_ratios = [width_1, width_3, colorbar_frac]

    # Set the figure width. The height will be adjusted automatically to produce the
    # right aspect ratio.
    fig_width = default_settings['combined_scatter_plot_width']
    fig = plt.figure(figsize=(fig_width, fig_width))
    gs = GridSpec(2, ncols, figure=fig, height_ratios=[range_3, range_2],
        width_ratios=width_ratios, )

    ax12 = fig.add_subplot(gs[1, 0])
    ax13 = fig.add_subplot(gs[0, 0], sharex=ax12)
    ax32 = fig.add_subplot(gs[1, 1], sharey=ax12)

    if discrete_color_map:
        # Show the legend in the middle of the upper right open space.
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis('off')
    else:
        # Show the colorbar on the right side of everything.
        cax = fig.add_subplot(gs[:, 2])

    plot_kwargs = {'s': default_settings['combined_scatter_plot_marker_size'],
        'edgecolors': 'gray', }

    if discrete_color_map:
        plot_kwargs['cmap'] = cmap

    plot_kwargs.update(kwargs)

    scatter = ax12.scatter(use_embedding[:, axis_1], use_embedding[:, axis_2],
        c=c12, **plot_kwargs, )
    ax12.set_xlabel(f'$\\xi_{axis_1 + 1}$')
    ax12.set_ylabel(f'$\\xi_{axis_2 + 1}$')
    ax12.set_xlim(min_1, max_1)
    ax12.set_ylim(min_2, max_2)

    ax13.scatter(use_embedding[:, axis_1], use_embedding[:, axis_3], c=c13,
        **plot_kwargs)
    ax13.set_ylabel(f'$\\xi_{axis_3 + 1}$')
    ax13.tick_params(labelbottom=False)
    ax13.set_ylim(min_3, max_3)

    ax32.scatter(use_embedding[:, axis_3], use_embedding[:, axis_2], c=c32,
        **plot_kwargs)
    ax32.set_xlabel(f'$\\xi_{axis_3 + 1}$')
    ax32.tick_params(labelleft=False)
    ax32.set_xlim(min_3, max_3)

    if discrete_color_map:
        # Show a legend with the discrete colors
        legend_ax.legend(handles=scatter.legend_elements()[0],
                         labels=discrete_color_map.keys(), loc='center')
    else:
        # Show a colorbar
        if label is not None:
            cb = fig.colorbar(sm, cax=cax, label=label)
        else:
            cb = fig.colorbar(sm, cax=cax)

        if invert_colorbar:
            # workaround: in my version of matplotlib, the ticks disappear if
            # you invert the colorbar y-axis. Save the ticks, and put them back
            # to work around that bug.
            ticks = cb.get_ticks()
            cb.ax.invert_yaxis()
            cb.set_ticks(ticks)

    # Calculate the aspect ratio, and regenerate the figure a few times until we get
    # it right.
    while True:
        fig.canvas.draw()

        coord = ax12.get_position() * fig.get_size_inches()
        plot_width = coord[1][0] - coord[0][0]
        plot_height = coord[1][1] - coord[0][1]
        plot_ratio = plot_height / plot_width

        aspect_ratio = plot_ratio / ax12.get_data_ratio()

        if np.abs(aspect_ratio - 1) < 0.001:
            # Good enough
            break

        fig.set_size_inches(
            [fig_width, fig.get_size_inches()[1] / aspect_ratio])
    fig.savefig(save_path, dpi=300)
    return ax12, ax13, ax32
