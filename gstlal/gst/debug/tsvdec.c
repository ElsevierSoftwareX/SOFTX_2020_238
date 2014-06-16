#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glib.h>
#include <gst/gst.h>

#include "tsvdec.h"

/* default field and record separators */
#define DEFAULT_FS "\t"
#define DEFAULT_RS "\n"

/* handles escaped control sequences */
static char * string_convert(const char *s)
{
	char *os = strdup(s);
	char *ret = os;
	int i;
	for (i = 0; s[i]; ++i) {
		switch (s[i]) {
		case '\\':	/* escaped character */
			switch (s[i+1]) {
			case '\\':
				*os++ = '\\';
				break;
			case 'a':
				*os++ = '\a';
				break;
			case 'b':
				*os++ = '\b';
				break;
			case 'f':
				*os++ = '\f';
				break;
			case 'n':
				*os++ = '\n';
				break;
			case 'r':
				*os++ = '\r';
				break;
			case 't':
				*os++ = '\t';
				break;
			case 'v':
				*os++ = '\v';
				break;
			case '\0': /* impossible! */
				g_assert_not_reached();
				break;
			default:
				*os++ = s[i+1];
				break;
			}
			++i;
			break;
		default:	/* non-escaped character */
			*os++ = s[i];
			break;
		}
	}
	*os = 0; /* nul-terminate */
	return ret;
}

/* my memmem (since memmem is not portable) */
/* locates a byte substring in a byte string */
static void * mymemmem(const void *haystack, size_t haystack_size, const void *needle, size_t needle_size)
{
	size_t nbytes;
	size_t bytes;

	if (!haystack || !needle || haystack_size < needle_size)
		return NULL;

	if (needle_size == 1) { /* fall-back to memchr */
		int c = *(const char *)(needle);
		return memchr(haystack, c, haystack_size);
	}

	/* max number of bytes to compare */
	nbytes = haystack_size - needle_size + 1;
	for (bytes = 0; bytes < nbytes; ++bytes)
		if (memcmp(haystack + bytes, needle, needle_size) == 0)
			return (void *)(size_t)(haystack) + bytes;

	return NULL;
}

GST_BOILERPLATE(
	GSTLALTSVDec,
	gstlal_tsvdec,
	GstElement,
	GST_TYPE_ELEMENT
);

static GstClockTime parse_line(int *chanc, double **chanv, char *line, const char *delim)
{
	GstClockTime t;
	double *x = NULL;
	int n = 0;
	char *s;

	/* get time stamp from first column */
	s = strsep(&line, delim);
	t = 1000000000 * strtol(s, &s, 10);	/* integer part */
	if (*s == '.')	/* fractional part; note: need long double precision */
		t += roundl(1e9L * strtold(s, NULL));

	while (line) {
		x = realloc(x, (n + 1) * sizeof(*x));
		x[n++] = strtod(strsep(&line, delim), NULL);
	}

	*chanc = n;
	*chanv = x;
	return t;
}

static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GstFlowReturn result = GST_FLOW_OK;
	GSTLALTSVDec *element = GSTLAL_TSVDEC(gst_pad_get_parent(pad));
	GstClockTime t = GST_CLOCK_TIME_NONE;
	GstBuffer *srcbuf = NULL;
	GstCaps *caps;
	guint nbytes;
	guint nlines = 0;
	char *bufdata = NULL;
	guint bufsize = 0;

	/* get copy of caps of srcpad */
	caps = gst_pad_get_fixed_caps_func(element->srcpad);
	caps = gst_caps_copy(caps);

	/* append sinkbuf data to data accumulated so far */
	nbytes = GST_BUFFER_SIZE(sinkbuf);
	element->data = realloc(element->data, element->size + nbytes);
	memcpy(element->data + element->size, GST_BUFFER_DATA(sinkbuf), nbytes);
	element->size += nbytes;

	/* loop while we have data */
	while (element->size) {
		const char *endp;
		char *line;
		GstBuffer *tmpbuf;
		double *chanv;
		int chanc;

		/* get a line of data */
		endp = mymemmem(element->data, element->size, element->RS, strlen(element->RS));
		if (endp == NULL) /* no newline found: wait for more data */
			break;

		/* if rate is unset, will need a second line */
		if (element->rate == 0 && nlines == 0)
			if (mymemmem(endp + strlen(element->RS), element->size - (endp - element->data + strlen(element->RS)), element->RS, strlen(element->RS)) == NULL)
				break;

		/* grab the line */
		++nlines;
		nbytes = endp - element->data; /* line length w/o RS */
		line = malloc(nbytes + 1); /* ensure line is nul-terminated */
		memcpy(line, element->data, nbytes);
		line[nbytes] = 0; /* ensure line is nul-terminated */

		/* drop the data points including RS from element data */
		nbytes += strlen(element->RS);
		element->size -= nbytes;
		memmove(element->data, element->data + nbytes, element->size);
		element->data = realloc(element->data, element->size);

		/* parse the line */
		t = parse_line(&chanc, &chanv, line, element->FS);
		/* fprintf(element->fp, "%d.%09d", t / 1000000000, t % 1000000000);
		int c;
		for (c = 0; c < chanc; ++c)
			fprintf(element->fp, "\t%.6e", chanv[c]);
		fprintf(element->fp, "\n");
		*/
		

		/* make sure the number of channels hasn't changed */
		if (element->channels)
			g_assert_cmpint(chanc, ==, element->channels);
		else {
			element->channels = chanc;
			gst_caps_set_simple(caps, "channels", G_TYPE_INT, chanc, NULL);
		}

		/* create a temporary buffer holding the data */
		tmpbuf = gst_buffer_new_and_alloc(chanc * sizeof(double));
		memcpy(GST_BUFFER_DATA(tmpbuf), chanv, GST_BUFFER_SIZE(tmpbuf));
		GST_BUFFER_TIMESTAMP(tmpbuf) = t;

		/* join this to the current buffer, if it exists */
		srcbuf = srcbuf ? gst_buffer_join(srcbuf, tmpbuf) : tmpbuf;

		free(line);
	}

	/* set buffer and caps on source pad */
	if (srcbuf && GST_BUFFER_SIZE(srcbuf)) {
		if (element->rate == 0) { /* if rate is unset, compute it */
			guint dt = t - GST_BUFFER_TIMESTAMP(srcbuf);
			element->rate = round((nlines - 1) * 1e9 / dt);
			gst_caps_set_simple(caps, "rate", G_TYPE_INT, element->rate, NULL);
			gst_pad_set_caps(element->srcpad, caps);
		}
		
		GST_BUFFER_OFFSET(srcbuf) = element->offset;
		GST_BUFFER_OFFSET_END(srcbuf) = element->offset += nlines;
		GST_BUFFER_DURATION(srcbuf) = round(nlines * 1e9 / element->rate);
		gst_buffer_set_caps(srcbuf, caps);
		result = gst_pad_push(element->srcpad, srcbuf);
	}

	return result;
}

enum property { ARG_FS = 1, ARG_RS = 2 };

static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTSVDec *element = GSTLAL_TSVDEC(object);
	GST_OBJECT_LOCK(element);
	switch (id) {
	case ARG_FS:
		free(element->FS);
		element->FS = string_convert(g_value_get_string(value));
		break;
	case ARG_RS:
		free(element->RS);
		element->RS = string_convert(g_value_get_string(value));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}

static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTSVDec *element = GSTLAL_TSVDEC(object);
	GST_OBJECT_LOCK(element);
	switch (id) {
	case ARG_FS:
		g_value_set_string(value, element->FS);
		break;
	case ARG_RS:
		g_value_set_string(value, element->RS);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}

static void finalize(GObject *object)
{
	GSTLALTSVDec *element = GSTLAL_TSVDEC(object);

	if (element->sinkpad) {
		gst_object_unref(element->sinkpad);
		element->sinkpad = NULL;
	}

	if (element->srcpad) {
		gst_object_unref(element->srcpad);
		element->srcpad = NULL;
	}

	if (element->data) {
		free(element->data);
		element->data = NULL;
		element->size = 0;
	}

	free(element->FS);
	free(element->RS);

	/* fclose(element->fp); */

	G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void gstlal_tsvdec_base_init(gpointer klass)
{
}

static void gstlal_tsvdec_class_init(GSTLALTSVDecClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"Tab separated value decoder",
		"Decoder",
		"Decodes a bytestream containing tab-separated values",
		"Jolien Creighton <jolien.creighton@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string("text/tab-separated-values")
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64; "
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_FS,
		g_param_spec_string(
			"FS",
			"FS",
			"Field separator",
			DEFAULT_FS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_RS,
		g_param_spec_string(
			"RS",
			"RS",
			"Record separator",
			DEFAULT_RS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}

static void gstlal_tsvdec_init(GSTLALTSVDec *element, GSTLALTSVDecClass *Klass)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	element->srcpad = pad;

	element->data = NULL;
	element->size = 0;
	element->rate = 0;
	element->channels = 0;
	element->offset = 0;
	element->FS = strdup(DEFAULT_FS);
	element->RS = strdup(DEFAULT_RS);

	/* element->fp = fopen("dump.out", "w"); */
}
