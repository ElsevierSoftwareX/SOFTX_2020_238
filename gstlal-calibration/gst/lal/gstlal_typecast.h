/*
 * Copyright (C) 2019 Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_TYPECAST_H__
#define __GSTLAL_TYPECAST_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_TYPECAST_TYPE \
	(gstlal_typecast_get_type())
#define GSTLAL_TYPECAST(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TYPECAST_TYPE, GSTLALTypeCast))
#define GSTLAL_TYPECAST_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TYPECAST_TYPE, GSTLALTypeCastClass))
#define GST_IS_GSTLAL_TYPECAST(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TYPECAST_TYPE))
#define GST_IS_GSTLAL_TYPECAST_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TYPECAST_TYPE))


typedef struct _GSTLALTypeCast GSTLALTypeCast;
typedef struct _GSTLALTypeCastClass GSTLALTypeCastClass;


/*
 * GSTLALTypeCast:
 */


struct _GSTLALTypeCast {
	GstBaseTransform element;

	/* stream info */
	int unit_size_in;
	int unit_size_out;
	int channels;
	int rate;
	gboolean complex_in;
	gboolean complex_out;
	gboolean float_in;
	gboolean float_out;
	gboolean sign_in;
	gboolean sign_out;

	/* timestamp book-keeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;
};


/*
 * GSTLALTypeCastClass:
 * @parent_class:  the parent class
 */


struct _GSTLALTypeCastClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_typecast_get_type(void);


G_END_DECLS


#endif  /* __GSTLAL_TYPECAST_H__ */
