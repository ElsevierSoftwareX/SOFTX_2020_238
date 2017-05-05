import sys, os, glob, time

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
	def __init__(self, href, div, text, charts=[]):
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
		css=["//versions.ligo.org/cgit/gstlal/plain/gstlal-ugly/share/vis/gstlal.css", 
			"//versions.ligo.org/cgit/gstlal/plain/gstlal-ugly/share/vis/jquery.fancybox.css"
			], 
		script=["//versions.ligo.org/cgit/gstlal/plain/gstlal-ugly/share/vis/jquery-3.1.1.min.js",
			"//versions.ligo.org/cgit/gstlal/plain/gstlal-ugly/share/vis/jquery.fancybox.pack.js?v=2.1.5",
			"https://www.gstatic.com/charts/loader.js",
			"//versions.ligo.org/cgit/gstlal/plain/gstlal-ugly/share/vis/gstlal.js"
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
		css_list = [elem("link", [], """ rel="stylesheet" type="text/css" href="%s" """ % c) for c in self.css]
		script_list = [elem("script", [], """ type="text/javascript" src="%s" """ % s) for s in self.script]
		self.full_content = [elem("html",  css_list + script_list + self.header_content + [elem("title", [self.title]), elem("body", self.content, "")])]
		for c in self.full_content:
			print >>f, c

def section(text):
	return elem("details", [elem("summary", [text])], "")

def googleTableFromJson(fname, div_id = 'table_div'):
	f = open(fname)
	out = """
		<script type="text/javascript">

		function draw_%s() {
		var data = new google.visualization.DataTable(%s);
		var table = new google.visualization.Table(document.getElementById('%s'));
		table.draw(data, {showRowNumber: true, width: '100%%', allowHtml: true, page: "enable"});
		}
		google.charts.setOnLoadCallback(draw_%s);
		</script>
	""" % (div_id, f.read(), div_id, div_id)
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
