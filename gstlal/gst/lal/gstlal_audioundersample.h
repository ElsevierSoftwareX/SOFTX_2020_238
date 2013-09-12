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


#ifndef __GST_LAL_AUDIOUNDERSAMPLE_H__
#define __GST_LAL_AUDIOUNDERSAMPLE_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_AUDIOUNDERSAMPLE_TYPE \
	(gstlal_audioundersample_get_type())
#define GSTLAL_AUDIOUNDERSAMPLE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_AUDIOUNDERSAMPLE_TYPE, GSTLALAudioUnderSample))
#define GSTLAL_AUDIOUNDERSAMPLE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_AUDIOUNDERSAMPLE_TYPE, GSTLALAudioUnderSampleClass))
#define GST_IS_GSTLAL_AUDIOUNDERSAMPLE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_AUDIOUNDERSAMPLE_TYPE))
#define GST_IS_GSTLAL_AUDIOUNDERSAMPLE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_AUDIOUNDERSAMPLE_TYPE))


typedef struct _GSTLALAudioUnderSample GSTLALAudioUnderSample;
typedef struct _GSTLALAudioUnderSampleClass GSTLALAudioUnderSampleClass;


/**
 * GSTLALAudioUnderSample:
 */


struct _GSTLALAudioUnderSample {
	GstBaseTransform element;

	/*
	 * stream info
	 */

	gint rate_in;
	gint rate_out;
	gint unit_size;

	/*
	 * timestamp book-keeping
	 */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/*
	 * internal state
	 */

	guint64 remainder;
};


/**
 * GSTLALAudioUnderSampleClass
 */


struct _GSTLALAudioUnderSampleClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_audioundersample_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_AUDIOUNDERSAMPLE_H__ */
