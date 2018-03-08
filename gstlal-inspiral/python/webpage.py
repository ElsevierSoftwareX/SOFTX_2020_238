import sys, os, glob, time

if "GSTLAL_WEBVIS_DIR" in os.environ:
	webvis_dir = os.environ["GSTLAL_WEBVIS_DIR"]
else:
	raise ValueError("Must set environment variable GSTLAL_WEBVIS_DIR to path containing gstlal.css and gstlal.js. This is typically the directory set during the configure step of gstlal-ugly. You can also set it to the static directories at each of the following: UWM - /home/gstlalcbc/public_html/share/vis/, CIT - /home/gstlalcbc/public_html/share/vis/, Atlas - /home/gstlalcbc/WWW/LSC/share/vis/")

class elem(object):
	def __init__(self, tag, content, attributes = ""):
		self.tag = tag; self.content = content; self. attributes = attributes;

	def __iadd__(self, content):
		try:
			self.content += content
		except TypeError:
			self.content += [content]
		return self

	def __str__(self):
		out = "<%s %s>\n" % (self.tag, self.attributes)
		for c in self.content:
			out += "%s\n" % str(c)
		out += "</%s>" % self.tag
		return out


class tabs(elem):
	def __init__(self, content = []):
		elem.__init__(self, tag="ul", content = [] + content, attributes = 'class="tab"')
		# self.content += [elem("li", [elem("img", [], """ style="width: 100px; margin: 5px 5px 5px 5px;" src="http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal/doc/gstlal.png" """)])]
		#self.content += [elem("li", [elem("div", [time.strftime("%Y-%m-%d %H:%M")])])]

	def __iadd__(self, content):
		try:
			self.content += content
		except TypeError:
			self.content += [content]
		return self
	

class tab(elem):
	def __init__(self, href, div, text, charts=[], active = False):
		self.href = href; self.div = div; self.text = text;
		if len(charts) > 0:
			elem.__init__(self, tag="li", content = [elem("a", [text], """ href=#%s class="tablinks" onclick="openGstlalTab(event, '%s',%s)" """ % (href, div, ",".join(charts)) )], attributes = "")
		else:
			elem.__init__(self, tag="li", content = [elem("a", [text], """ href=#%s class="tablinks" onclick="openGstlalTab(event, '%s')" """ % (href, div) )], attributes = "")
	
	def __call__(self, content=[]):
		return elem("div", content, """ id="%s" class="tabcontent" """ % self.div)

class image_glob(elem):
	def __init__(self, globpat, caption):
		self.globpat = globpat; self.caption = caption;
		elem.__init__(self, "table", [], """ style="width: 100%; table-layout: auto" """)
		cap = elem("caption", ["Table: " + caption], """ style="caption-side: bottom; text-align: left; font-size: 12px; font-style: italic; padding: 15px;" """)
		td = elem("td", [])
		tr = elem("tr", [td])
		for img in sorted(glob.glob(globpat)):
			td += [elem("a", [elem("img", [], """ src="%s" width=500 """ % img)], """ class="fancybox" href="%s" rel="group" """ % img)]
		self.content = [cap, tr]
			
class page(object):
	def __init__(self, title="cbc web page", path='./',
		css=["https://cdnjs.cloudflare.com/ajax/libs/fancybox/2.1.5/jquery.fancybox.css"
			], 
		script=["https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js",
			"https://cdnjs.cloudflare.com/ajax/libs/fancybox/2.1.5/jquery.fancybox.js",
			"https://www.gstatic.com/charts/loader.js"
			], 
		content = None, header_content = None, verbose=False):
		if content is None:
			content = []
		if header_content is None:
			header_content = ["""<script type="text/javascript">google.charts.load('current', {'packages':['table', 'timeline']});</script>"""]
		self.title = title; self.path = path; self.css = css; self.script = script; self.content = content; self.verbose = verbose; self.header_content = header_content

	def __iadd__(self, content):
		try:
			self.content += content
		except TypeError:
			self.content += [content]
		return self


	def write(self, f):
		gstlal_css_file = open(webvis_dir + 'gstlal.css')
		gstlal_js_file = open(webvis_dir + 'gstlal.js')
		gstlal_css = """<style>""" + gstlal_css_file.read() + """</style>"""
		gstlal_js = """<script>""" + gstlal_js_file.read()  + """</script>"""
		gstlal_list = [elem("head", [gstlal_css, gstlal_js])]
		css_list = [elem("link", [], """ rel="stylesheet" type="text/css" href="%s" """ % c) for c in self.css]
		script_list = [elem("script", [], """ type="text/javascript" src="%s" """ % s) for s in self.script]
		self.full_content = [elem("html",  gstlal_list + css_list + script_list + self.header_content + [elem("title", [self.title]), elem("body", self.content, "")])]
		for c in self.full_content:
			print >>f, c

def section(text):
	return elem("details", [elem("summary", [text])], "open")

def googleTableFromJson(fname, div_id = 'table_div', gpscolumns = [], scinotationcolumns = []):
	f = open(fname)
	gpsformatstr = ["""
		var gpsformatter = new google.visualization.NumberFormat(
			{groupingSymbol: ''});\n""" + '\n'.join(["""		gpsformatter.format(data, %d);""" % (gpscolumn,) for gpscolumn in gpscolumns]) if gpscolumns else ''][0]
	scinotationformatstr = ["""
		var formatter = new google.visualization.NumberFormat(
			{pattern: '0.###E0'});\n""" + '\n'.join(["""		formatter.format(data, %d);""" % (scinotationcolumn,) for scinotationcolumn in scinotationcolumns]) if scinotationcolumns else ''][0]
	out = """
		<script type="text/javascript">

		function draw_%s() {
		var data = new google.visualization.DataTable(%s);
		var table = new google.visualization.Table(document.getElementById('%s'));%s%s
		table.draw(data, {showRowNumber: true, width: '100%%', allowHtml: true, page: "enable"});
		}
		google.charts.setOnLoadCallback(draw_%s);
		</script>
	""" % (div_id, f.read(), div_id, gpsformatstr, scinotationformatstr, div_id)
	f.close()
	return out

def googleTimelineFromJson(fname, div_id = 'timeline_div'):
	f = open(fname)
	out = """
		<script type="text/javascript">

		var %s_wrapper;

		function draw_%s() {
			var data = new google.visualization.DataTable();

			data.addColumn('string', 'name');
			data.addColumn('string', 'label');
			data.addColumn({ type: 'string', role: 'tooltip' });
			data.addColumn('number', 'start');
			data.addColumn('number', 'end');
			data.addRows(%s);

			%s_wrapper = new google.visualization.ChartWrapper({
				chartType: 'Timeline',
				dataTable: data,
				options: {width:'95%%', height:400, textStyle: {color: '#ecf0f1'}},
				containerId: '%s'
			});
			%s_wrapper.draw();
		}

		google.charts.setOnLoadCallback(draw_%s);

		</script>
	""" % (div_id, div_id, f.read(), div_id, div_id, div_id, div_id)
	f.close()
	return out
