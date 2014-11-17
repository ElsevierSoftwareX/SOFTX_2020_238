/*
 * Copyright (C) 2011 Chad Hanna <chad.hanna@ligo.org>, Kipp Cannon <kipp.cannon@ligo.org>, Drew Keppel <drew.keppel@ligo.org>
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

#ifndef __GSTLAL_BURST_TRIGGERGEN_H__
#define __GSTLAL_BURST_TRIGGERGEN_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <lal/LIGOMetadataTables.h>
#include <gsl/gsl_matrix.h>

G_BEGIN_DECLS


#define GSTLAL_BURST_TRIGGERGEN_TYPE \
	(gstlal_burst_triggergen_get_type())
#define GSTLAL_BURST_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_BURST_TRIGGERGEN_TYPE, GSTLALBurst_Triggergen))
#define GSTLAL_BURST_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_BURST_TRIGGERGEN_TYPE, GSTLALBurst_TriggergenClass))
#define GST_IS_GSTLAL_BURST_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_BURST_TRIGGERGEN_TYPE))
#define GST_IS_GSTLAL_BURST_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_BURST_TRIGGERGEN_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALBurst_TriggergenClass;


typedef struct {

	GstElement element;

	GstPad *sinkpad;
	GstPad *srcpad;
	GstAudioAdapter *adapter;
	
	SnglBurst *event_buffer;

	gint rate;
	guint n;
	guint channels;
	guint64 count;
	guint64 total_offset;
	gdouble snr_thresh;
	/* TODO: Right now we're just copying all the data structures */
	struct gstlal_peak_state *maxdata;
	struct gstlal_peak_state *maxdatad;
	double complex *data;
	double *datad;
	enum {
		GSTLAL_BURSTTRIGGEN_DOUBLE,
		GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE,
		GSTLAL_BURSTTRIGGEN_DOUBLE_NO_PEAK,
		GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE_NO_PEAK
	} data_type;
	guint64 next_output_offset;
	GstClockTime next_output_timestamp;
	SnglBurst *bankarray;
	char * bank_filename;
	char * instrument;
	char * channel_name;
	gboolean last_gap;
	gboolean EOS;
	
	GMutex *bank_lock;
} GSTLALBurst_Triggergen;


GType gstlal_burst_triggergen_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_BURST_TRIGGERGEN_H__ */
