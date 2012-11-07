/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna
 * <chad.hanna@ligo.caltech.edu>
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifndef __GSTLAL_TRIGGERGEN_H__
#define __GSTLAL_TRIGGERGEN_H__


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gstlal/gstlalcollectpads.h>
#include <lal/LIGOMetadataTables.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                             Trigger Generator
 *
 * ============================================================================
 */


#define GSTLAL_TRIGGERGEN_TYPE \
	(gstlal_triggergen_get_type())
#define GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGen))
#define GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGenClass))
#define GST_IS_GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TRIGGERGEN_TYPE))
#define GST_IS_GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TRIGGERGEN_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALTriggerGenClass;


typedef struct {
	GstElement element;

	GstCollectPads *collect;
	GstPadEventFunction collect_event;

	GstPad *snrpad;
	GstLALCollectData *snrcollectdata;
	GstPad *chisqpad;
	GstLALCollectData *chisqcollectdata;
	GstPad *srcpad;

	gboolean segment_pending;
	gboolean flush_stop_pending;
	GstSegment segment;
	guint64 next_output_offset;
	guint64 next_output_timestamp;

	int rate;

	GMutex *bank_lock;
	char *bank_filename;
	gchar *instrument;
	gchar *channel_name;
	SnglInspiralTable *bank;
	gint num_templates;
	double snr_thresh;
	double max_gap;
	SnglInspiralTable *last_event;
	LIGOTimeGPS *last_time;
} GSTLALTriggerGen;


GType gstlal_triggergen_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TRIGGERGEN_H__ */
