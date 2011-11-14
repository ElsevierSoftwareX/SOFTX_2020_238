# Chad Hanna
# FIXME add Copyright, license does this belong in its own module?
#


def channel_dict_from_channel_list(channel_list):
	"""
	given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""

	channel_dict = {"H1" : "LSC-STRAIN", "H2" : "LSC-STRAIN", "L1" : "LSC-STRAIN", "V1" : "LSC-STRAIN", "G1" : "LSC-STRAIN", "T1" : "LSC-STRAIN"}

	for channel in channel_list:
		ifo = channel.split("=")[0]
		chan = "".join(channel.split("=")[1:])
		channel_dict[ifo] = chan

	return channel_dict


def pipeline_channel_list_from_channel_dict(channel_dict):
	"""
	produce a string of channel name arguments suitable for a pipeline.py program that doesn't technically allow multiple options. For example --channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN
	"""

	outstr = ""
	for i, ifo in enumerate(channel_dict):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[ifo])
		else:
			outstr += "--channel-name=%s=%s " % (ifo, channel_dict[ifo])

	return outstr
