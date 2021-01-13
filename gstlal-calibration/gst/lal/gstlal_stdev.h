/*
 * Copyright (C) 2021 Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_STDEV_H__
#define __GSTLAL_STDEV_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_STDEV_TYPE \
	(gstlal_stdev_get_type())
#define GSTLAL_STDEV(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_STDEV_TYPE, GSTLALStDev))
#define GSTLAL_STDEV_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_STDEV_TYPE, GSTLALStDevClass))
#define GST_IS_GSTLAL_STDEV(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_STDEV_TYPE))
#define GST_IS_GSTLAL_STDEV_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_STDEV_TYPE))


typedef struct _GSTLALStDev GSTLALStDev;
typedef struct _GSTLALStDevClass GSTLALStDevClass;


/*
 * gstlal_stdev_mode enum
 */


enum gstlal_stdev_mode {
	GSTLAL_STDEV_ABSOLUTE = 0,
	GSTLAL_STDEV_RELATIVE
};


#define GSTLAL_STDEV_MODE \
	(gstlal_stdev_mode_get_type())


GType gstlal_stdev_mode_get_type(void);


/**
 * GSTLALStDev:
 */


struct _GSTLALStDev {
	GstBaseTransform element;

	/* Pads */
	GstPad *srcpad;

	/* stream information */
	gint unit_size;
	gint rate;
	enum gstlal_stdev_data_type {
		GSTLAL_STDEV_F32 = 0,
		GSTLAL_STDEV_F64,
		GSTLAL_STDEV_Z64,
		GSTLAL_STDEV_Z128
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 total_insamples;
	guint64 next_out_offset;
	gboolean need_discont;

	/* filter memory */
	union {
		struct {
			float current_stdev;
			float *array;
		} typef;  /* real float */
		struct {
			double current_stdev;
			double *array;
		} type;  /* real double */
		struct {
			float current_stdev;
			complex float *array;
		} ctypef;  /* complex float */
		struct {
			double current_stdev;
			complex double *array;
		} ctype;  /* complex double */
	} workspace;

	guint64 start_index;
	guint64 array_index;
	guint64 buffer_index;
	guint64 samples_in_array;

	/* properties */
	guint64 array_size;
	guint64 coherence_length;
	enum gstlal_stdev_mode mode;
	double filter_latency;
};


/**
 * GSTLALStDevClass:
 * @parent_class:  the parent class
 */


struct _GSTLALStDevClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_stdev_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_STDEV_H__ */
