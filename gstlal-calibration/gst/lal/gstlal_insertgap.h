/*
 * Copyright (C) 2016 Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_INSERTGAP_H__
#define __GSTLAL_INSERTGAP_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS
#define GSTLAL_INSERTGAP_TYPE \
	(gstlal_insertgap_get_type())
#define GSTLAL_INSERTGAP(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_INSERTGAP_TYPE, GSTLALInsertGap))
#define GSTLAL_INSERTGAP_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_INSERTGAP_TYPE, GSTLALInsertGapClass))
#define GST_IS_GSTLAL_INSERTGAP(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_INSERTGAP_TYPE))
#define GST_IS_GSTLAL_INSERTGAP_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_INSERTGAP_TYPE))


typedef struct _GSTLALInsertGap GSTLALInsertGap;
typedef struct _GSTLALInsertGapClass GSTLALInsertGapClass;


/**
 * GSTLALInsertGap:
 */


struct _GSTLALInsertGap {
	GstElement element;

	/* pads */
	GstPad *sinkpad;
	GstPad *srcpad;

	/* stream parameters */
	gint rate;
	gint channels;
	gint unit_size;
	enum gstlal_insertgap_data_type {
		GSTLAL_INSERTGAP_U32 = 0,
		GSTLAL_INSERTGAP_F32,
		GSTLAL_INSERTGAP_F64,
		GSTLAL_INSERTGAP_Z64,
		GSTLAL_INSERTGAP_Z128
	} data_type;

	guint64 last_sinkbuf_ets;
	guint64 last_sinkbuf_offset_end;
	guint64 discont_offset;
	guint64 discont_time;
	guint64 empty_bufs;
	GMutex mutex;
	gboolean finished_running;

	/* timestamp bookkeeping */
	GstClockTime t0;

	/* properties */
	gboolean insert_gap;
	gboolean remove_gap;
	gboolean remove_nan;
	gboolean remove_inf;
	gboolean fill_discont;
	double replace_value;
	double *bad_data_intervals;
	gint array_length;
	guint64 chop_length;
	GstClockTime block_duration;
	guint64 wait_time;
};


/**
 * GSTLALInsertGapClass:
 * @parent_class:  the parent class
 */


struct _GSTLALInsertGapClass {
	GstElementClass parent_class;
};


GType gstlal_insertgap_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_INSERTGAP_H__ */
