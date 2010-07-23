#!/usr/bin/env python

for ra_deg in range(0,360,10):
	for codec_deg in range(0,180,10):
		print 'JOB testSkyMap_%(ra_deg)03d_%(codec_deg)03d testSkyMap.sub\nVARS testSkyMap_%(ra_deg)03d_%(codec_deg)03d ra_deg="%(ra_deg)03d" codec_deg="%(codec_deg)03d"' % {"ra_deg":ra_deg,"codec_deg":codec_deg}

