/*
 * Copyright (C) 2011 Kipp Cannon <kipp.cannon@ligo.org>
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


#ifndef __GST_LAL_STATEVECTOR_H__
#define __GST_LAL_STATEVECTOR_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_STATEVECTOR_TYPE \
	(gstlal_statevector_get_type())
#define GSTLAL_STATEVECTOR(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_STATEVECTOR_TYPE, GSTLALStateVector))
#define GSTLAL_STATEVECTOR_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_STATEVECTOR_TYPE, GSTLALStateVectorClass))
#define GST_IS_GSTLAL_STATEVECTOR(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_STATEVECTOR_TYPE))
#define GST_IS_GSTLAL_STATEVECTOR_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_STATEVECTOR_TYPE))


typedef struct _GSTLALStateVector GSTLALStateVector;
typedef struct _GSTLALStateVectorClass GSTLALStateVectorClass;


/**
 * GSTLALStateVector:
 */


struct _GSTLALStateVector {
	GstBaseTransform element;

	guint required_on;
	guint required_off;
	guint mask;
	guint64 on_samples;
	guint64 off_samples;
	guint64 gap_samples;

	guint (*get_input)(void **);
};


/**
 * GSTLALStateVectorClass:
 */


struct _GSTLALStateVectorClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_statevector_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_STATEVECTOR_H__ */
