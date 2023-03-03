#
#  This file is part of CRAPy
#
#  CRAPy is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  CRAPy is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Healpy; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#  For more information about Healpy, see http://code.google.com/p/healpy
#
"""
CRAPy is a package to extract, analyse and simulate cosmic ray anisotropies. It is applicable for TeV-ZeV data collected by ground-based cosmic ray observatories. 
"""

import warnings

from .version import __version__

from .coordinates import (
	Lambert_ang2pix,
	Lambert_pix2ang,
	LC2EQ_vector,
	EQ2LC_vector,
	Lambert_to_healpy,
)

from .likelihood import (
	INA,
	INA_Lambert,
	extract_allsky,
	extract_allsky_Lambert,
	extract_dipole,
	extract_dipole_Lambert,
	dipole_sigma,
	dipole_sigma_Lambert,
)

from .EastWest import (
	EWderivative,
	EWdipole,
)

from .datatools import (
	mockdata,
	mockdata_Lambert,
	Lambertview,
	smoothdata,
	rebindata,
)




