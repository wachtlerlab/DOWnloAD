# -------------------------------------------------------------------------------
# Copyright (c) 2016 Christian Garbers.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Simplified BSD License
# which accompanies this distribution
#
# Contributors:
#     Christian Garbers - initial API and implementation
# -------------------------------------------------------------------------------
from scipy import interpolate

import numpy

from receptors import *
from d65 import d65, d65_wl
from tungsteen_spectrum import *




# These are pixel measurements from the paper figure
points_pix = [[201, 534],
              [257, 538],
              [332, 399],
              [382, 413],
              [466, 541],
              [473, 541],
              [536, 306],
              [650, 248],
              [745, 405],
              [753, 399]
              ]

y_errors_pix = [48,
                57,
                79,
                43,
                87,
                83,
                114,
                142,
                180,
                114

                ]

x_errors_pix = [19,
                21,
                29,
                17,
                32,
                20,
                58,
                50,
                70,
                47
                ]

# Facors and function relating pixel to wavelength
y_fac = 50 / 235.0  # 235 are 50nm
x_fac = 100 / 338.0  # 338 pixel are 100nm
get_xy = lambda x, y: (((x - 483) * x_fac) + 500, 50 - (y - 413) * y_fac)
points_wl = numpy.array([get_xy(*e) for e in points_pix]).T
deltas = points_wl[1, :]

points_wl = numpy.array([get_xy(*e) for e in points_pix]).T
y_error_wl = numpy.array([e * y_fac for e in y_errors_pix]) * 0.5
x_error_wl = numpy.array([e * x_fac for e in x_errors_pix]) * 0.5
sal_spatz_data = numpy.array([points_wl[0, :], deltas])
final_y_errors = y_error_wl

# receptor spectral sensitivities
rh1 = numpy.array(rh1)
rh3 = numpy.array(rh3)
rh4 = numpy.array(rh4)
rh5 = numpy.array(rh5)
rh6 = numpy.array(rh6)
recs_scaled = numpy.array([rh1, rh3, rh4, rh5, rh6])

tun = numpy.array(tungsten)
tun_ip = interpolate.interp1d(tun[:, 0], tun[:, 1])(wl)
recs_scaled_tungsten = recs_scaled.T * (1 / numpy.dot(recs_scaled, tun_ip))
tun_ip = interpolate.interp1d(d65_wl, d65)(wl)
recs_scaled_d65 = recs_scaled.T * (1 / numpy.dot(recs_scaled, tun_ip))
