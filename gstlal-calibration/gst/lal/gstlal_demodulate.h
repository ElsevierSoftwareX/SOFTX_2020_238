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


#ifndef __GST_LAL_DEMODULATE_H__
#define __GST_LAL_DEMODULATE_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_DEMODULATE_TYPE \
	(gstlal_demodulate_get_type())
#define GSTLAL_DEMODULATE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_DEMODULATE_TYPE, GSTLALDemodulate))
#define GSTLAL_DEMODULATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_DEMODULATE_TYPE, GSTLALDemodulateClass))
#define GST_IS_GSTLAL_DEMODULATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_DEMODULATE_TYPE))
#define GST_IS_GSTLAL_DEMODULATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_DEMODULATE_TYPE))


typedef struct _GSTLALDemodulate GSTLALDemodulate;
typedef struct _GSTLALDemodulateClass GSTLALDemodulateClass;


/*
 * GSTLALDemodulate:
 */


struct _GSTLALDemodulate {
	GstBaseTransform element;

	/* stream info */
	gint unit_size;
	gint rate;
	enum gstlal_demodulate_data_type {
		GSTLAL_DEMODULATE_F32 = 0,
		GSTLAL_DEMODULATE_F64,
		GSTLAL_DEMODULATE_Z64,
		GSTLAL_DEMODULATE_Z128
	} data_type;

	/* timestamp book-keeping */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* properties */
	int line_frequency; /* in centihertz */
	double prefactor_real;
	double prefactor_imag;
};


/*
 * GSTLALDemodulateClass:
 * @parent_class:  the parent class
 */


struct _GSTLALDemodulateClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_demodulate_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_DEMODULATE_H__ */
