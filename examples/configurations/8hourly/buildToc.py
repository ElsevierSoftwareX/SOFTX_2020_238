#!/usr/bin/env python

import glob
import os.path

print >>open('index.html', 'w'), """
<html>
<title>gstlal Online 8-Hourly Results Index</title>
</head>
<frameset cols="20%,80%">
<frame src="index_list.html" id="listframe"/>
<frame id="contentframe" />
</frameset>
</html>
"""

print >>open('index_list.html', 'w'), "<html><body>" + "".join([
	'<a target="contentframe" href="%(dir)s/index_sngl.html">%(dir)s</a><br />' % {'dir':subdirname} for subdirname in glob.glob('????-??-??T??')]
) + "</body><html>"

