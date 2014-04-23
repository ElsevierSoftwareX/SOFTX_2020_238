# Copyright (C) 2014  Chris Pankow, Madeline Wade
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

from gstlal import datasource
from collections import defaultdict

def channel_dict_set_from_channel_list(channel_list):
	"""!
	Given a list of channels, produce a dictionary keyed by ifo of channel names:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> channel_dict_set_from_channel_list(["H1=LSC-DARM_CTRL", "H1=LSC-DARM_ERR"])
		{'H1': ('LSC-DARM_CTRL', 'LSC-DARM_ERR')}
        """
	channel_dict = defaultdict(set)
	for channel_name in channel_list:
		instrument, channel = channel_name.split("=")
		channel_dict[instrument].add(channel)
	return channel_dict

class GWDataSourceInfoMulti(datasource.GWDataSourceInfo):
	def __init__(self, options):
		super(GWDataSourceInfoMulti, self).__init__(options)
		self.channel_list = options.channel_name
		self.channel_dict = channel_dict_set_from_channel_list(options.channel_name)
		


