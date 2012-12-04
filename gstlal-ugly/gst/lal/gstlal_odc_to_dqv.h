/*
 * Copyright (C) 2012 Kipp Cannon <kipp.cannon@ligo.org>, Chris Pankow <chris.pankow@ligo.org>
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


#ifndef __GST_LAL_ODC_TO_DQV_H__
#define __GST_LAL_ODC_TO_DQV_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_ODC_TO_DQV_TYPE \
	(gstlal_odc_to_dqv_get_type())
#define GSTLAL_ODC_TO_DQV(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_ODC_TO_DQV_TYPE, GSTLALODCtoDQV))
#define GSTLAL_ODC_TO_DQV_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_ODC_TO_DQV_TYPE, GSTLALODCtoDQVClass))
#define GST_IS_GSTLAL_ODC_TO_DQV(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_ODC_TO_DQV_TYPE))
#define GST_IS_GSTLAL_ODC_TO_DQV_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_ODC_TO_DQV_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALODCtoDQVClass;


typedef struct GSTLALODCtoDQV {
	GstBaseTransform element;

	guint required_on;
	guint status_out;
	guint64 gap_samples;
} GSTLALODCtoDQV;


GType gstlal_odc_to_dqv_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_ODC_TO_DQV_H__ */
