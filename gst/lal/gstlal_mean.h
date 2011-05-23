/*
 * Copyright (C) 2010 Kipp Cannon <kipp.cannon@ligo.org>
 * Copyright (C) 2010 Chad Hanna <chad.hanna@ligo.org>
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

#ifndef __GST_LAL_MEAN_H__
#define __GST_LAL_MEAN_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS


#define GSTLAL_MEAN_TYPE \
	(gstlal_mean_get_type())
#define GSTLAL_MEAN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MEAN_TYPE, GSTLALMean))
#define GSTLAL_MEAN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MEAN_TYPE, GSTLALMeanClass))
#define GST_IS_GSTLAL_MEAN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MEAN_TYPE))
#define GST_IS_GSTLAL_MEAN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MEAN_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALMeanClass;


typedef struct {
	GstBaseTransform element;

	/*
	 * input stream
	 */

	gint rate;
	gint channels;
	GstAdapter *adapter;

	/*
	 * averaging parameters
	 */

	guint32 n;
	guint32 type;
	guint32 moment;
	double *sum1;
	double *sum2;
	double *max;
	guint64 *lastmax;
	guint64 *lastcross;
	double thresh;
	gboolean invert_thresh;

	/*
	 * timestamp book-keeping
	 */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_out_offset;
	gboolean need_discont;

	/*
	 * process function
	 * FIXME support other caps
	 */
	
	int (*process)();

} GSTLALMean;


GType gstlal_mean_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_MEAN_H__ */
