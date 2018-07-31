import math
# copied from inspiral_pipe.T050017_filename
def T050017_filename(instruments, description, seg, extension, path = None):
	"""!
	A function to generate a T050017 filename.
	"""
	if not isinstance(instruments, basestring):
		instruments = "".join(sorted(instruments))
	start, end = seg
	start = int(math.floor(start))
	duration = int(math.ceil(end)) - start
	extension = extension.strip('.')
	if path is not None:
		return '%s/%s-%s-%d-%d.%s' % (path, instruments, description, start, duration, extension)
	else:
		return '%s-%s-%d-%d.%s' % (instruments, description, start, duration, extension)


