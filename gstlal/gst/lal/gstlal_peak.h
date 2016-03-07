/*
 * Copyright (C) 2011 Chad Hanna <chad.hanna@ligo.org>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef __GSTLAL_PEAK_H__
#define __GSTLAL_PEAK_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>

#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>

G_BEGIN_DECLS


#define GSTLAL_PEAK_TYPE \
	(gstlal_peak_get_type())
#define GSTLAL_PEAK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_PEAK_TYPE, GSTLALPeak))
#define GSTLAL_PEAK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_PEAK_TYPE, GSTLALPeakClass))
#define GST_IS_GSTLAL_PEAK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_PEAK_TYPE))
#define GST_IS_GSTLAL_PEAK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_PEAK_TYPE))


typedef struct _GSTLALPeak GSTLALPeak;
typedef struct _GSTLALPeakClass GSTLALPeakClass;


/**
 * GSTLALPeak:
 */


struct _GSTLALPeak {
	GstElement element;

	GstPad *sinkpad;
	GstPad *srcpad;

	gint rate;
	gint samples_since_last_discont;
	guint64 timestamp_at_last_discont;
	guint n;
	guint channels;
	GstAudioAdapter *adapter;
	gstlal_peak_type_specifier peak_type;
	struct gstlal_peak_state *maxdata;
	void *data;
	guint64 next_output_offset;
	GstClockTime next_output_timestamp;
};


/**
 * GSTLALPeakClass:
 * @parent_class:  the parent class
 */


struct _GSTLALPeakClass {
	GstElementClass parent_class;
};


GType gstlal_peak_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_PEAK_H__ */
