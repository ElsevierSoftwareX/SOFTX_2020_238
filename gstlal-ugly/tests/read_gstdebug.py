# 1. cat postcohspiir-24379115.err | grep ", ts " > postcohspiir-24379115-ts.err
# 2. python read_gstdebug.py postcohspiir-24379115-ts.err > table.txt
# 3. delete all "gstpad.c:4708:gst_pad_push:" in table.txt (optionally)
# 4. copy table.txt and paste to excel, sort by time_stamp (optionally)

import sys

with open(sys.argv[1], "r") as f:
	# read all lines
	lines = f.readlines()
	# tag to tag_id
	tag2id = {}
	# current tag_id
	curid = 0
	
	# tag list
	tags = []
	# fulltag list
	fulltags = []
	
	# data {time_stamp1: {tag1: xx1, tag2: xx2}}, time_stamp2: ...}
	data = {}
	
	for idx, line in enumerate(lines):
		splits = [x for x in line.split(' ') if len(x) > 0]
		
		# time from starting program
		time = splits[0]
		# LOG type
		type = splits[3]
		# LOG file
		file = splits[4]
		# tag
		tag = splits[5]
		# full tag
		fulltag = " ".join(splits[5:9])
		# time stamp
		ts = splits[splits.index("ts") + 1]
		# duration
		if "dur" in splits:
			dur = splits[splits.index("dur") + 1]
		elif "duration" in splits:
			dur = splits[splits.index("duration") + 1]
		else:
			raise "not found duration information"

		# filter some data
		if tag.split(":")[2] != "gst_pad_push":
			continue
		if dur.startswith("0:00:00.000000000"):
			continue

		# tag, tag_id
		if tag not in tag2id:
			tagid = curid
			tag2id[tag] = tagid
			tags.append(tag)  # tags[curid] == tagid
			fulltags.append(fulltag)  # fulltags[curid] == tagid
			curid += 1
		else:
			tagid = tag2id[tag]
			# make sure that one tag only has one fulltag
			assert fulltag == fulltags[tagid]

		if ts not in data:
			data[ts] = {}

		assert (fulltag not in data[ts]), "time stamp %s has two %s records" % (ts, fulltag)
		data[ts][fulltag] = (time, type, file)
		
	assert len(tag2id) == len(tags)
	assert len(tag2id) == len(fulltags)
	# print "num ts items: ", len(data)
	# print "num tags type: ", len(tag2id)
	# for tag in tags:
	# 	print tag

	# print table header
	print "\t",
	for fulltag in fulltags:
		print "%s\t" % (fulltag),
	print 

	# print table row
	# time_stamp(ts), dt[fulltag1], dt[fulltag2], ...
	for ts, dt in data.items():
		print "%s\t" % (ts),
		for fulltag in fulltags:
			if fulltag in dt:
				print "%s\t" % (dt[fulltag][0]),
			else:
				print "\t",
		print
