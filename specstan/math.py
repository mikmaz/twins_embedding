import numpy as np
from scipy.linalg import pinvh, LinAlgError


def nmad(x, *args, unbiased=False, centered=False, **kwargs):
    x = np.asarray(x)
    if not centered:
        x = x - np.median(x, *args, **kwargs)

    nmad = 1.4826 * np.median(np.abs(x), *args, **kwargs)

    if unbiased:
        nmad = nmad * x.size / (x.size - 1)

    return nmad


def hessian_to_covariance(hessian, valid_parameter_mask=None):
    """Safely invert a Hessian matrix to get a covariance matrix.

    Sometimes, parameters can have wildly different scales from each other. What we
    actually care about is having the same relative precision on the error of each
    parameter rather than the absolute precision. In that case, we can normalize the
    Hessian prior to inverting it, and then renormalize afterwards. This deals with the
    problem of varying scales of parameters gracefully.

    If valid_parameter_mask is given, then the parameters whose values are False in
    the mask are ignored, and their variance is set to -1 with a covariance of 0 with
    every other parameter.
    """
    if valid_parameter_mask is not None:
        hessian = hessian[valid_parameter_mask][:, valid_parameter_mask]

    # Choose scales to set the diagonal of the hessian to 1.
    scales = np.sqrt(np.diag(hessian))
    norm_hessian = hessian / np.outer(scales, scales)

    # Now invert the scaled Hessian using a safe inversion algorithm
    inv_norm_hessian = pinvh(norm_hessian)

    # Add the scales back in.
    covariance = inv_norm_hessian / np.outer(scales, scales)

    if valid_parameter_mask is not None:
        full_cov = -1.0 * np.eye(len(valid_parameter_mask))
        valid_mask_2d = np.outer(valid_parameter_mask, valid_parameter_mask)
        full_cov[valid_mask_2d] = covariance.flat
        covariance = full_cov

    return covariance


class IdrToolsMathException(Exception):
    pass


def calculate_covariance_finite_difference(
    negative_log_likelihood,
    parameter_names,
    values,
    bounds,
    verbose=False,
    allow_no_effect=False,
):
    """Estimate the covariance of the parameters of negative log likelihood function
    numerically.

    We do a 2nd order finite difference estimate of the covariance matrix.
    For this, the formula is:
    d^2f(dx1*dx2) = ((f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2))
                     / 4*e1*e2)
    So we need to calculate all the f(x +/-e1 +/-e2) terms (where e1 and
    e2 are small steps in 2 possibly different directions).

    We use adaptive step sizes to build a robust estimate of the Hessian, and invert it
    to obtain the covariance matrix.

    Parameters
    ----------
    negative_log_likelihood : function
        Negative log-likelihood function to evaluate. This should take as input a list
        of parameter values.
    parameter_names : list of str
        Names of each of the parameters.
    values : list of floats
        Values of each of the parameters at the maximum likelihood location. The
        likelihood should be run through a minimizer before calling this function, and
        the resulting values should be passed into this function.
    bounds : list of tuples
        Bounds for each parameter. For each parameter, this should be a two parameter
        tuple with the first entry being the minimum bound and the second entry being
        the maximum bound.
    verbose : bool
        If True, output diagnostic messages. Default: False
    allow_no_effect : bool
        If True, keep parameters that don't affect the model. Their variance will be
        set to -1. If False, an exception will be raised.
    """
    # The three terms here are the corresponding weight, the sign of e1 and
    # the sign of e2.
    difference_info = [
        (+1 / 4.0, +1.0, +1.0),
        (-1 / 4.0, +1.0, -1.0),
        (-1 / 4.0, -1.0, +1.0),
        (+1 / 4.0, -1.0, -1.0),
    ]

    num_variables = len(parameter_names)

    # Determine good step sizes. Since we have a chi-square function, a 1-sigma
    # change in a parameter corresponds to a 1 unit change in the output
    # chi-square function. We want our steps to change the chi-square function
    # by an amount of roughly 1e-5 (far from machine precision, but small
    # enough to be probing locally). We start by guessing a step size of 1e-5
    # (which is typically pretty reasonable for parameters that are of order 1)
    # and then bisect to find the right value.
    steps = []
    ref_likelihood = negative_log_likelihood(values)

    valid_parameter_mask = np.ones(num_variables, dtype=bool)

    for parameter_idx in range(len(parameter_names)):
        step = 1e-5
        min_step = None
        max_step = None

        # Move away from the nearest bounds to avoid boundary issues.
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        if min_bound is None:
            if max_bound is None:
                # No bounds, doesn't matter what we pick.
                direction = +1.0
            else:
                # Max bound only
                direction = -1.0
        else:
            if max_bound is None:
                # Min bound only
                direction = +1.0
            else:
                # Both bounds, move away from the nearest bound.
                if value - min_bound > max_bound - value:
                    direction = -1.0
                else:
                    direction = 1.0

        while True:
            # Estimate the second derivative numerator for a finite difference
            # calculation. We want to choose a step size that sets this to a
            # reasonable value. Note that we move only in the direction away
            # from the nearest boundary, so this isn't centered at the correct
            # position, but this is only to get an initial estimate of the
            # scale so it doesn't matter.
            step_values = values.copy()
            step_values[parameter_idx] += step * direction
            step_1_likelihood = negative_log_likelihood(step_values)
            step_values[parameter_idx] += step * direction
            step_2_likelihood = negative_log_likelihood(step_values)
            diff = (
                0.25 * step_2_likelihood
                - 0.5 * step_1_likelihood
                + 0.25 * ref_likelihood
            )

            if diff < -1e-4:
                # We found a minimum that is better than the supposed true
                # minimum. This indicates that something is wrong because the
                # minimizer failed.
                raise IdrToolsMathException(
                    "Second derivative is negative when varying %s to "
                    "calculate covariance matrix! Something is very wrong! "
                    "(step=%f, second derivative=%f)"
                    % (parameter_names[parameter_idx], step, diff)
                )

            if diff < 1e-6:
                # Too small step size, increase it.
                min_step = step
                if max_step is not None:
                    step = (step + max_step) / 2.0
                else:
                    step = step * 2.0
            elif diff > 1e-4:
                # Too large step size, decrease it.
                max_step = step
                if min_step is not None:
                    step = (step + min_step) / 2.0
                else:
                    step = step / 2.0
            else:
                # Good step size, we're done.
                break

            if step > 1e9:
                # Shouldn't need steps this large. This only happens if one
                # parameter doesn't affect the model at all, in which case we
                # can't calculate the covariance.
                message = (
                    "Parameter %s doesn't appear to affect the model! Cannot "
                    "estimate its covariance." % parameter_names[parameter_idx]
                )
                if allow_no_effect:
                    if verbose:
                        print(f"WARNING: {message}")
                    valid_parameter_mask[parameter_idx] = False
                    step = 0.0
                    break
                else:
                    raise IdrToolsMathException(message)

        steps.append(step)

    steps = np.array(steps)
    if verbose:
        print("Finite difference covariance step sizes: %s" % steps)

    difference_matrices = []

    # If we are too close to a bound, then we can't calculate the covariance.
    for parameter_idx in range(len(parameter_names)):
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        step = steps[parameter_idx]

        if (min_bound is not None and value - 2 * step < min_bound) or (
            max_bound is not None and value + 2 * step > max_bound
        ):
            # Step would cross the bound.
            message = (
                f"WARNING: Parameter {parameter_names[parameter_idx]} is too close to "
                f"bound! Cannot estimate its covariance."
            )
            if allow_no_effect:
                if verbose:
                    print(f"WARNING: {message}")
                valid_parameter_mask[parameter_idx] = False
            else:
                raise IdrToolsMathException(message)

    # Calculate all of the terms that will be required to calculate the finite
    # differences. Note that there is a lot of reuse of terms, so here we
    # calculate everything that is needed and build a set of matrices for each
    # step combination.
    for weight, sign_e1, sign_e2 in difference_info:
        matrix = np.zeros((num_variables, num_variables))
        for i in range(num_variables):
            for j in range(num_variables):
                if i > j:
                    # Symmetric
                    continue

                if not (valid_parameter_mask[i] and valid_parameter_mask[j]):
                    continue

                step_values = values.copy()
                step_values[i] += sign_e1 * steps[i]
                step_values[j] += sign_e2 * steps[j]
                likelihood = negative_log_likelihood(step_values)
                matrix[i, j] = likelihood
                matrix[j, i] = likelihood

        difference_matrices.append(matrix)

    # Hessian
    hessian = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            if i > j:
                continue

            if not (valid_parameter_mask[i] and valid_parameter_mask[j]):
                continue

            val = 0.0

            for (weight, sign_e1, sign_e2), matrix in zip(
                difference_info, difference_matrices
            ):
                val += weight * matrix[i, j]

            val /= steps[i] * steps[j]

            hessian[i, j] = val
            hessian[j, i] = val

    # Invert the Hessian to get the covariance matrix
    try:
        cov = hessian_to_covariance(hessian, valid_parameter_mask)
    except LinAlgError:
        raise IdrToolsMathException("Covariance matrix is not well defined!")

    variance = np.diag(cov)

    if np.any(variance[valid_parameter_mask] < 0):
        raise IdrToolsMathException(
            "Covariance matrix is not well defined! "
            "Found negative variances."
        )

    return cov
