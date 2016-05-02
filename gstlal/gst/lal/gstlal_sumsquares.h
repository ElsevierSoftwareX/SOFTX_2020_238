/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>
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


#ifndef __GST_LAL_SUMSQUARES_H__
#define __GST_LAL_SUMSQUARES_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_SUMSQUARES_TYPE \
	(gstlal_sumsquares_get_type())
#define GSTLAL_SUMSQUARES(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SUMSQUARES_TYPE, GSTLALSumSquares))
#define GSTLAL_SUMSQUARES_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SUMSQUARES_TYPE, GSTLALSumSquaresClass))
#define GST_IS_GSTLAL_SUMSQUARES(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SUMSQUARES_TYPE))
#define GST_IS_GSTLAL_SUMSQUARES_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SUMSQUARES_TYPE))


typedef struct _GSTLALSumSquares GSTLALSumSquares;
typedef struct _GSTLALSumSquaresClass GSTLALSumSquaresClass;


/**
 * GSTLALSumSquares:
 */


struct _GSTLALSumSquares {
	GstBaseTransform element;

	gint channels;

	GMutex weights_lock;
	double *weights;
	void *weights_native;

	void *(*make_weights_native_func)(GSTLALSumSquares *);
	GstFlowReturn (*sumsquares_func)(GSTLALSumSquares *, GstBuffer *, GstBuffer *);
};


/**
 * GSTLALSumSquaresClass:
 * @parent_class:  the parent class
 */


struct _GSTLALSumSquaresClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_sumsquares_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_SUMSQUARES_H__ */
