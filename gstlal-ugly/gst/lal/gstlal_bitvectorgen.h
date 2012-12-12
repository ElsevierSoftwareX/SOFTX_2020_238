/*
 * Copyright (C) 2012 Kipp Cannon <kipp.cannon@ligo.org>
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


#ifndef __GST_LAL_BITVECTORGEN_H__
#define __GST_LAL_BITVECTORGEN_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_BITVECTORGEN_TYPE \
	(gstlal_bitvectorgen_get_type())
#define GSTLAL_BITVECTORGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_BITVECTORGEN_TYPE, GSTLALBitVectorGen))
#define GSTLAL_BITVECTORGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_BITVECTORGEN_TYPE, GSTLALBitVectorGenClass))
#define GST_IS_GSTLAL_BITVECTORGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_BITVECTORGEN_TYPE))
#define GST_IS_GSTLAL_BITVECTORGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_BITVECTORGEN_TYPE))


typedef struct _GSTLALBitVectorGenClass GSTLALBitVectorGenClass;
typedef struct _GSTLALBitVectorGen GSTLALBitVectorGen;


struct _GSTLALBitVectorGenClass {
	GstBaseTransformClass parent_class;

	void (*rate_changed)(GSTLALBitVectorGen *, gint, void *);
	void (*start)(GSTLALBitVectorGen *, guint64, void *);
	void (*stop)(GSTLALBitVectorGen *, guint64, void *);
};


struct _GSTLALBitVectorGen {
	GstBaseTransform element;

	gint rate;
	guint64 mask;
	gdouble (*get_input_func)(void **);
	void (*set_output_func)(void **, guint64);

	gboolean emit_signals;
	gint last_state;
	gdouble threshold;
	gint64 attack_length;
	gint64 hold_length;
	gboolean invert_control;
	gboolean nongap_is_control;
	guint64 bit_vector;
};


GType gstlal_bitvectorgen_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_BITVECTORGEN_H__ */
