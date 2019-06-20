import numpy as np
from ligo.lw import ligolw
from ligo.lw.ligolw import LIGOLWContentHandler
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo.lw import lsctables
from ligo.segments import utils as segments_utils
from lal import rate

def p_astro_update(category, event_bayesfac_dict, mean_values_dict):
    """
    Compute `p_astro` for a new event using mean values of Poisson expected
    counts constructed from all the previous events. Invoked with every new
    GraceDB entry.

    Parameters
    ----------
    category : string
        source category
    event_bayesfac_dict : dictionary
        event Bayes factors
    mean_values_dict : dictionary
        mean values of Poisson counts

    Returns
    -------
    p_astro : float
        p_astro by source category
    """
    if category == "counts_Terrestrial":
        numerator = mean_values_dict["counts_Terrestrial"]
    else:
        numerator = \
            event_bayesfac_dict[category] * mean_values_dict[category]

    denominator = mean_values_dict["counts_Terrestrial"] + \
        np.sum([mean_values_dict[key] * event_bayesfac_dict[key]
                for key in event_bayesfac_dict.keys()])

    return numerator / denominator


def evaluate_p_astro_from_bayesfac(astro_bayesfac,
                                   mean_values_dict,
                                   mass1,
                                   mass2,
                                   spin1z=None,
                                   spin2z=None,
                                   num_bins=None,
                                   activation_counts=None):
    """
    Evaluates `p_astro` for a new event using Bayes factor, masses, and number
    of astrophysical categories. Invoked with every new GraceDB entry.

    Parameters
    ----------
    astro_bayesfac : float
        astrophysical Bayes factor
    mean_values_dict: dictionary
        mean values of Poisson counts
    mass1 : float
        event mass1
    mass2 : float
        event mass2
    spin1z : float
        event spin1z
    spin2z : float
        event spin2z
    url_weights_key: str
        url config key pointing to weights file

    Returns
    -------
    p_astro : dictionary
        p_astro for all source categories
    """

    a_hat_bns, a_hat_bbh, a_hat_nsbh, a_hat_mg, num_bins = \
        make_weights_from_histograms(mass1,
                                     mass2,
                                     spin1z,
                                     spin2z,
                                     num_bins,
                                     activation_counts)

    # Compute category-wise Bayes factors
    # from astrophysical Bayes factor
    rescaled_fb = num_bins * astro_bayesfac
    bns_bayesfac = a_hat_bns * rescaled_fb
    nsbh_bayesfac = a_hat_nsbh * rescaled_fb
    bbh_bayesfac = a_hat_bbh * rescaled_fb
    mg_bayesfac = a_hat_mg * rescaled_fb

    # Construct category-wise Bayes factor dictionary
    event_bayesfac_dict = {"counts_BNS": bns_bayesfac,
                           "counts_NSBH": nsbh_bayesfac,
                           "counts_BBH": bbh_bayesfac,
                           "counts_MassGap": mg_bayesfac}

    # Compute the p-astro values for each source category
    # using the mean values
    p_astro_values = {}
    for category in mean_values_dict:
        p_astro_values[category.split("_")[1]] = \
            p_astro_update(category=category,
                           event_bayesfac_dict=event_bayesfac_dict,
                           mean_values_dict=mean_values_dict)

    return p_astro_values


def make_weights_from_hardcuts(mass1, mass2):
    """
    Construct binary weights from component masses based on cuts in component
    mass space that define astrophysical source categories. To be used for
    MBTA, PyCBC and SPIIR.

    Parameters
    ----------
    mass1 : float
        heavier component mass of the event
    mass2 : float
        lighter component mass of the event

    Returns
    -------
    a_bns, a_bbh, a_nshb, a_mg : floats
        binary weights (i.e, 1 or 0)
    """

    a_hat_bns = int(mass1 <= 3 and mass2 <= 3)
    a_hat_bbh = int(mass1 > 5 and mass2 > 5)
    a_hat_nsbh = int(min(mass1, mass2) <= 3 and
                     max(mass1, mass2) > 5)
    a_hat_mg = int(3 < mass1 <= 5 or 3 < mass2 <= 5)
    num_bins = 4

    return a_hat_bns, a_hat_bbh, a_hat_nsbh, a_hat_mg, num_bins


def closest_template(params, params_array):
    """
    Associate event's template to a template in the template bank. The assumed
    bank is the one used by Gstlal. Hence, for Gstlal events, the association
    should be exact, up to rounding errors.

    Parameters
    ----------
    params : tuple of floats
        intrinsic params of event template
    params_array: array of arrays
        array of template bank's template params

    Returns
    -------
    idx : int
        index pertaining to params_array
        for matching template
    """
    idx = np.argmin(np.sum((params_array-params)**2,axis=1))

    return idx


def make_weights_from_histograms(mass1,
                                 mass2,
                                 spin1z,
                                 spin2z,
                                 num_bins=None,
                                 activation_counts=None):
    """
    Construct binary weights from bin number provided by GstLAL, and a weights
    matrix pre-constructed and stored in a file, to be read from a url. The
    weights are keyed on template parameters of Gstlal's template bank. If that
    doesn't work, construct binary weights.

    Parameters
    ----------
    mass1 : float
        heavier component mass of the event
    mass2 : float
        lighter component mass of the event
    spin1z : float
        z component spin of heavier mass
    spin2z : float
        z component spin of lighter mass
    num_bins : int
        number of bins for template weighting
    activation_counts : pandas dataframe
        data frame for template weights

    Returns
    -------
    a_hat_bns, a_hat_bbh, a_hat_nsbh, a_hat_mg : floats
        mass-based template weights
    """

    if activation_counts is None or num_bins is None:
        a_hat_bns, a_hat_bbh, a_hat_nsbh, a_hat_mg, num_bins = \
            make_weights_from_hardcuts(mass1, mass2)
    else:
        params = (mass1, mass2, spin1z, spin2z)
        params_names = ['mass1', 'mass2', 'spin1z', 'spin2z']
        params_array = \
            np.array([activation_counts[key][:] for key in params_names]).T
        idx = closest_template(params, params_array)
        a_hat_bns = activation_counts['bns'][:][idx]
        a_hat_mg = activation_counts['mg'][:][idx]
        a_hat_nsbh = activation_counts['nsbh'][:][idx]
        a_hat_bbh = activation_counts['bbh'][:][idx]

    return a_hat_bns, a_hat_bbh, a_hat_nsbh, a_hat_mg, num_bins


def _get_event_ln_likelihood_ratio_svd_endtime_mass(xmldoc):
    """
    Task to acquire event parameters

    Parameters
    ----------
    xmldoc : lsctables object
        Object from which event parameters can be extracted

    Returns
    -------
    event parameters : tuple
        Tuple of event parameters: lnl, component masses, snr, far
    """

    coinc_event, = lsctables.CoincTable.get_table(xmldoc)
    sngl_inspiral = lsctables.SnglInspiralTable.get_table(xmldoc)
    coinc_inspiral = lsctables.CoincInspiralTable.get_table(xmldoc)

    return (coinc_event.likelihood,
            sngl_inspiral[0].mass1,
            sngl_inspiral[0].mass2,
            coinc_inspiral.snr,
            coinc_inspiral.combined_far)


def _get_ln_f_over_b(rankingstatpdf,
                     ln_likelihood_ratios,
                     livetime):

    # affect the zeroing of the PDFs below threshold by hacking the
    # histograms. Do the indexing ourselves to not 0 the bin @ threshold
    noise_lr_lnpdf = rankingstatpdf.noise_lr_lnpdf
    signal_lr_lnpdf = rankingstatpdf.signal_lr_lnpdf
    zero_lag_lr_lnpdf = rankingstatpdf.zero_lag_lr_lnpdf
    ssorted = zero_lag_lr_lnpdf.array.cumsum()[-1] - 10000
    idx = zero_lag_lr_lnpdf.array.cumsum().searchsorted(ssorted)

    ln_likelihood_ratio_threshold = \
        zero_lag_lr_lnpdf.bins[0].lower()[idx]

    # Compute FAR for threshold, and estimate
    # terrestrial count consistent with threshold
    fapfar = \
        FAPFAR(rankingstatpdf.new_with_extinction())
    far_threshold = fapfar.far_from_rank(ln_likelihood_ratio_threshold)
    lam_0 = far_threshold*livetime
    rankingstatpdf.noise_lr_lnpdf.array[
        :noise_lr_lnpdf.bins[0][ln_likelihood_ratio_threshold]] \
        = 0.
    rankingstatpdf.noise_lr_lnpdf.normalize()

    rankingstatpdf.signal_lr_lnpdf.array[
        :signal_lr_lnpdf.bins[0][ln_likelihood_ratio_threshold]] \
        = 0.
    rankingstatpdf.signal_lr_lnpdf.normalize()

    rankingstatpdf.zero_lag_lr_lnpdf.array[
        :zero_lag_lr_lnpdf.bins[0][ln_likelihood_ratio_threshold]] \
        = 0.
    rankingstatpdf.zero_lag_lr_lnpdf.normalize()

    f = rankingstatpdf.signal_lr_lnpdf
    b = rankingstatpdf.noise_lr_lnpdf
    ln_f_over_b = \
        np.array([f[ln_lr, ] - b[ln_lr, ] for ln_lr in ln_likelihood_ratios])
    if np.isnan(ln_f_over_b).any():
        raise ValueError("NaN encountered in ranking statistic PDF ratios")
    if np.isinf(np.exp(ln_f_over_b)).any():
        raise ValueError(
            "infinity encountered in ranking statistic PDF ratios")
    return ln_f_over_b, lam_0


def compute_p_astro(event_ln_likelihood_ratio,
                    event_mass1,
                    event_mass2,
                    snr,
                    far,
                    livetime,
                    mean_values_dict):
    """
    Task to compute `p_astro` by source category.

    Parameters
    ----------
    event_ln_likelihood_ratio : float
        Event's log likelihood ratio value
    event_mass1 : float
        Event's heavier component mass value
    event_mass2 : float
        Event's lighter component mass value
    snr: float
        Event's snr value
    far: float
        Event's far value
    livetime: float
        Livetime consistent with estimate of Poisson counts. 
        Eg. If counts are from O1 and O2, livetime is O1-O2
        livetime in seconds.
    mean_values_dict : dictionary
        dictionary of source specific FGMC Poisson counts

    Returns
    -------
    p_astros : str
        JSON dump of the p_astro by source category

    Example
    -------
    >>> import json
    >>> p_astros = json.loads(compute_p_astro(files))
    >>> p_astros
    {'BNS': 0.999, 'BBH': 0.0, 'NSBH': 0.0, 'Terrestrial': 0.001}
    """

    # Using the zerolag log likelihood ratio value event,
    # and the foreground/background model information provided
    # in ranking_data.xml.gz, compute the ln(f/b) value for this event
    zerolag_ln_likelihood_ratios = np.array([event_ln_likelihood_ratio])
    try:
        ln_f_over_b, lam_0 = \
            _get_ln_f_over_b(rankingstatpdf,
                             zerolag_ln_likelihood_ratios,
                             livetime)
    except ValueError:
        return compute_p_astro_approx(snr,
                                      far,
                                      event_mass1,
                                      event_mass2,
                                      mean_values_dict)

    # Read mean values from url file
    mean_values_dict["counts_Terrestrial"] = lam_0

    # Compute categorical p_astro values
    p_astro_values = \
        evaluate_p_astro_from_bayesfac(np.exp(ln_f_over_b[0]),
                                       mean_values_dict,
                                       event_mass1,
                                       event_mass2)

    # Dump values in json file
    return json.dumps(p_astro_values)

def compute_p_astro_approx(snr, far, mass1, mass2, livetime, mean_values_dict):
    """
    Task to compute `p_astro` by source category.

    Parameters
    ----------
    snr : float
        event's SNR
    far : float
        event's cfar
    mass1 : float
        event's mass1
    mass2 : float
        event's mass2
    livetime: float
        Livetime consistent with estimate of Poisson counts. 
        Eg. If counts are from O1 and O2, livetime is O1-O2
        livetime in seconds. 
    mean_values_dict : dictionary
        dictionary of source specific FGMC Poisson counts

    Returns
    -------
    p_astros : str
        JSON dump of the p_astro by source category

    Example
    -------
    >>> import json
    >>> p_astros = json.loads(compute_p_astro(files))
    >>> p_astros
    {'BNS': 0.999, 'BBH': 0.0, 'NSBH': 0.0, 'Terrestrial': 0.001}
    """

    # Define constants to compute bayesfactors
    snr_star = 8.5
    far_star = 1 / (30 * 86400)

    # Compute astrophysical bayesfactor for
    # GraceDB event
    fground = 3 * snr_star**3 / (snr_choice**4)
    bground = far / far_star
    astro_bayesfac = fground / bground

    # Update terrestrial count based on far threshold
    lam_0 = far_star * livetime
    mean_values_dict["counts_Terrestrial"] = lam_0

    # Compute categorical p_astro values
    p_astro_values = \
        evaluate_p_astro_from_bayesfac(astro_bayesfac,
                                       mean_values_dict,
                                       mass1,
                                       mass2)
    # Dump mean values in json file
    return json.dumps(p_astro_values)
