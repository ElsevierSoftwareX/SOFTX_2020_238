/*
 * Copyright (C) 2009, 2016 Kipp Cannon <kipp.cannon@ligo.org>, Madeline Wade * <madeline.wade@ligo.org>
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


#ifndef __GST_LAL_SMOOTHKAPPAS_H__
#define __GST_LAL_SMOOTHKAPPAS_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_SMOOTHKAPPAS_TYPE \
	(gstlal_smoothkappas_get_type())
#define GSTLAL_SMOOTHKAPPAS(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SMOOTHKAPPAS_TYPE, GSTLALSmoothKappas))
#define GSTLAL_SMOOTHKAPPAS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SMOOTHKAPPAS_TYPE, GSTLALSmoothKappasClass))
#define GST_IS_GSTLAL_SMOOTHKAPPAS(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SMOOTHKAPPAS_TYPE))
#define GST_IS_GSTLAL_SMOOTHKAPPAS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SMOOTHKAPPAS_TYPE))


typedef struct _GSTLALSmoothKappas GSTLALSmoothKappas;
typedef struct _GSTLALSmoothKappasClass GSTLALSmoothKappasClass;


/**
 * GSTLALSmoothKappas:
 */


struct _GSTLALSmoothKappas {
	GstBaseTransform element;

	/* Pads */
	GstPad *srcpad;

	/* stream information */
	gint unit_size;
	gint rate;
	enum gstlal_smoothkappas_data_type {
		GSTLAL_SMOOTHKAPPAS_F32 = 0,
		GSTLAL_SMOOTHKAPPAS_F64,
		GSTLAL_SMOOTHKAPPAS_Z64,
		GSTLAL_SMOOTHKAPPAS_Z128
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* filter memory */
	double current_median_re;
	double current_median_im;
	double *fifo_array_re;
	double *fifo_array_im;
	double *avg_array_re;
	double *avg_array_im;
	int index_re;
	int index_im;
	int avg_index_re;
	int avg_index_im;
	int num_bad_in_avg_re;
	int num_bad_in_avg_im;
	int samples_in_filter;

	/* properties */
	int array_size;
	int avg_array_size;
	double default_kappa_re;
	double default_kappa_im;
	double maximum_offset_re;
	double maximum_offset_im;
	gboolean default_to_median;
	gboolean track_bad_kappa;
	double filter_latency;
};


/**
 * GSTLALSmoothKappasClass:
 * @parent_class:  the parent class
 */


struct _GSTLALSmoothKappasClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_smoothkappas_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_SMOOTHKAPPAS_H__ */
