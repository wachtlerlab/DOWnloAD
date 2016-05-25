# -------------------------------------------------------------------------------
# Copyright (c) 2016 Christian Garbers.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Simplified BSD License
# which accompanies this distribution
#
# Contributors:
#     Christian Garbers - initial API and implementation
# -------------------------------------------------------------------------------

import itertools

import scipy.optimize
import scipy.stats
import scipy.stats

from pylab import slopes
import pylab
from data import *

s_rh1 = slopes(wl, rh1)
s_rh3 = slopes(wl, rh3)
s_rh4 = slopes(wl, rh4)
s_rh5 = slopes(wl, rh5)
s_rh6 = slopes(wl, rh6)

receptors_int = {1: s_rh1, 3: s_rh3, 4: s_rh4, 5: s_rh5, 6: s_rh6, 0: numpy.zeros(len(s_rh1)),
                 -1: -s_rh1, -3: -s_rh3, -4: -s_rh4, -5: -s_rh5, -6: -s_rh6, }


def err_fun(weights, mechs, data):
    '''
    calculate the residulas for the model
    '''
    sensitivity = get_sensitivity(weights, mechs)
    sensitivity = interpolate.interp1d(wl, sensitivity)
    res = sensitivity(data[0, :]) - data[1, :]
    return res


def get_sensitivity(weights, mechs, to_delta_lambda=lambda x: x):
    '''
    Returns the sensitivity for the mech weights combination
    '''
    result = numpy.zeros(len(receptors_int[1]))
    for counter, e in enumerate(mechs):
        result += abs(weights[counter]) * ((receptors_int[e[0]] - receptors_int[e[1]]) ** 2)
    return to_delta_lambda(1 / result)


def fit_model(mechs, data):
    '''
    fits the model with mechs to the data
    '''
    best_fit = scipy.optimize.leastsq(err_fun, numpy.ones(len(mechs)),
                                      full_output=True, args=(mechs, data))
    mse = (best_fit[2]['fvec'] ** 2).mean()
    sensitivity = get_sensitivity(best_fit[0], mechs)
    sensitivity = interpolate.interp1d(wl, sensitivity)
    res = sensitivity(data[0, :]) - data[1, :]
    return best_fit, mse, res


def chi2_err(weights, mechs, data, yerrors, xerrors):
    sensitivity = get_sensitivity(weights, mechs)
    sensitivity = interpolate.interp1d(wl, sensitivity)
    final_ystd = yerrors
    if len(xerrors) > 1:
        additional_ystd = []
        x_std = xerrors
        for e, i in zip(data[0, :], x_std):
            errorrange = sensitivity(numpy.arange(e - i, e + i))
            add_error = errorrange.max() - errorrange.min()
            additional_ystd.append(add_error / 2.)
        final_ystd = yerrors + numpy.array(additional_ystd)
    res = sensitivity(data[0, :]) - data[1, :]
    res = res ** 2 / final_ystd ** 2
    return res


def leastsq_fit_model(mechs, data, yerrors, xerrors):
    best_fit = scipy.optimize.leastsq(err_fun, numpy.ones(len(mechs)),
                                      full_output=True, args=(mechs, data))
    sensitivity = get_sensitivity(best_fit[0], mechs)
    sensitivity = interpolate.interp1d(wl, sensitivity)
    mse = ((sensitivity(data[0, :]) - data[1, :]) ** 2).mean()
    chisq = chi2_err(best_fit[0], mechs, data, yerrors, xerrors).sum()
    dof = len(sal_spatz_data[0, :]) - 1 - len(mechs)
    chisq_corr = chisq / dof
    p_value = 1. - scipy.stats.chi2.cdf(chisq, dof)
    return best_fit, mse, p_value, chisq, chisq_corr

def find_permutations(symbols, orders):
    result = []
    for slots in orders:
        result.extend(itertools.combinations(symbols, slots))
    return result

def show_model(nr, sal_spatz_data, final_y_errors, all_models, p_values, fits,
               x_error_wl):
    """
    Plot the model indicated by nr
    """
    pylab.subplot(2, 1, 1)
    pylab.plot(wl, get_sensitivity(fits[nr][0][0], all_models[nr]))
    if len(x_error_wl) > 0:
        pylab.errorbar(sal_spatz_data[0, :], sal_spatz_data[1, :], yerr=final_y_errors,
                       xerr=x_error_wl, fmt=None, capsize=6, color='green')
    else:
        pylab.errorbar(sal_spatz_data[0, :], sal_spatz_data[1, :], yerr=final_y_errors,
                       fmt=None, capsize=6, color='green')
    pylab.plot(sal_spatz_data[0, :], sal_spatz_data[1, :], "g*")
    pylab.text(310, 1.8, '%s' % [all_models[nr]])
    pylab.text(310, 40, 'p:%s' % [p_values[nr]])
    pylab.xlabel('wavelength [nm]')
    pylab.ylabel('$\delta\lambda$[nm]')
    pylab.ylim(0, 100)
    pylab.subplot(2, 1, 2)
    pylab.bar(range(len(fits[nr][0][0])), abs(fits[nr][0][0]))
    pylab.yticks([], [])
    pylab.xticks(numpy.arange(len(fits[nr][0][0])) + 0.5, all_models[nr])
    pylab.ylabel('weight')
