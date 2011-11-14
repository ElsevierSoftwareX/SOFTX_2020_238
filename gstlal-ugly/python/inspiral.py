# Chad Hanna
# FIXME add Copyright, license does this belong in its own module?
#


def channel_dict_from_channel_list(channel_list):
	"""
	given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""
	
	channel_dict = {"H1" : LSC-STRAIN, "H2" : LSC-STRAIN, "L1" : LSC-STRAIN, "V1" : LSC-STRAIN, "G1" : LSC-STRAIN, "T1" : LSC-STRAIN}
	
	for channel in channel_ist:
		ifo = channel.split("=")[0]
		chan = channel.split("=")[1:]
		channel_dict[ifo] = chan
	
	return channel_dict
