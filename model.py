# -------------------------------------------------------------------------------
# Copyright (c) 2016 Christian Garbers.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Simplified BSD License
# which accompanies this distribution
#
# Contributors:
#     Christian Garbers - initial API and implementation
# -------------------------------------------------------------------------------
import pylab

import numpy

from utilities import leastsq_fit_model, find_permutations, get_sensitivity
from data import sal_spatz_data, final_y_errors, x_error_wl, wl


# receptors to include (number refers to the opsin 1 is rh1)
recs_to_use = [1, 4, 5, 6, 3]
# combination to include
combinations = [2, 3, 4, 5, 6, 7, 8]

# create a list off all opponnent mechanism
opponent_mechs = []
for counter, first_rec in enumerate(recs_to_use):
    for sec_rec in recs_to_use[counter + 1:]:
        opponent_mechs.append((first_rec, sec_rec))
opponent_mechs = filter(lambda x: not all(map(lambda y: y < 0, x)), opponent_mechs)
opponent_mechs = filter(lambda x: not (x[0] + x[1] == 0), opponent_mechs)

# get all poossible visual systems
all_models = find_permutations(opponent_mechs, combinations)

# do the fitting
fits = [leastsq_fit_model(mechs, sal_spatz_data, final_y_errors,
                          x_error_wl) for mechs in
        all_models]

# split fits into those with rh1 and those without
rh1_models = []
no_rh1_models = []
for fit, model in zip(fits, all_models):
    if sum(map(lambda x: x.count(1), model)) > 0:
        rh1_models.append(fit)
    elif sum(map(lambda x: x.count(-1), model)) > 0:
        rh1_models.append(fit)
    else:
        no_rh1_models.append(fit)

# plot the historgramm of chi2 vlaues
pylab.figure()
pylab.hist([e[3] for e in rh1_models], normed='True', label='with rh1',
           bins=numpy.arange(0, 200, 10))
pylab.hist([e[3] for e in no_rh1_models], normed='True', alpha=0.5, label='no rh1',
           bins=numpy.arange(0, 200, 10))
pylab.legend()
pylab.xlabel('$\chi^2$', fontsize=36)
pylab.ylabel('density', fontsize=36)

# get the best fitting model indices with and without rh1
p_values = numpy.array([e[2] for e in fits])
best_rh1 = p_values.argsort()[-1]
best_without = 0
for nr in p_values.argsort()[::-1]:
    if sum(map(lambda x: x.count(1) + x.count(-1), all_models[nr])) == 0:
        best_without = nr
        break

# plot the best fitting model
pylab.figure()
pylab.plot(wl, get_sensitivity(fits[best_rh1][0][0], all_models[best_rh1]),
           color='black')
pylab.errorbar(sal_spatz_data[0, :], sal_spatz_data[1, :], yerr=final_y_errors,
               xerr=x_error_wl, fmt='', capsize=6, color='green')
pylab.plot(sal_spatz_data[0, :], sal_spatz_data[1, :], '.', color="grey")
pylab.xlim(400, 610)
pylab.ylim(0, 110)
pylab.xlabel('wavelength [nm]', fontsize=36)
pylab.ylabel('$\delta\lambda$[nm]', fontsize=36)

# plot the best fitting model without rh1
pylab.figure()
pylab.plot(wl, get_sensitivity(fits[best_without][0][0], all_models[best_without]),
           color="black")
pylab.errorbar(sal_spatz_data[0, :], sal_spatz_data[1, :], yerr=final_y_errors,
               xerr=x_error_wl, fmt='', capsize=6, color='green')
pylab.plot(sal_spatz_data[0, :], sal_spatz_data[1, :], '.', color="grey")
pylab.xlim(400, 610)
pylab.ylim(0, 110)
pylab.xlabel('wavelength [nm]', fontsize=36)
pylab.ylabel('$\delta\lambda$ [nm]', fontsize=36)

pylab.show()
