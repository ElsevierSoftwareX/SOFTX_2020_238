/*
 * Copyright (C) 2018 Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_DQTUKEY_H__
#define __GSTLAL_DQTUKEY_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GSTLAL_DQTUKEY_TYPE \
	(gstlal_dqtukey_get_type())
#define GSTLAL_DQTUKEY(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_DQTUKEY_TYPE, GSTLALDQTukey))
#define GSTLAL_DQTUKEY_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_DQTUKEY_TYPE, GSTLALDQTukeyClass))
#define GST_IS_GSTLAL_DQTUKEY(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_DQTUKEY_TYPE))
#define GST_IS_GSTLAL_DQTUKEY_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_DQTUKEY_TYPE))


typedef struct _GSTLALDQTukey GSTLALDQTukey;
typedef struct _GSTLALDQTukeyClass GSTLALDQTukeyClass;


/*
 * GSTLALDQTukey:
 */


struct _GSTLALDQTukey {
	GstBaseTransform element;

	/* stream info */

	gint rate_in;
	gint rate_out;
	gint unit_size_in;
	gint unit_size_out;
	gboolean sign;
	gint num_cycle_in;
	gint num_cycle_out;

	/* timestamp book-keeping */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* internal state */

	enum gstlal_dqtukey_state {
		START = 0,
		ONES,
		ZEROS,
		RAMP_UP,
		RAMP_DOWN,
		DOUBLE_RAMP
	} state;
	gint64 ramp_up_index;
	gint64 ramp_down_index;
	gint64 num_leftover;
	int remainder;
	gint64 num_since_bad;
	void *ramp;

	/* properties  */

	guint32 required_on;
	guint32 required_off;
	guint32 required_on_xor_off;
	gint64 transition_samples;
	gboolean invert_window;
	gboolean invert_control;
	gboolean planck_taper;
};


/*
 * GSTLALDQTukeyClass:
 * @parent_class:  the parent class
 */


struct _GSTLALDQTukeyClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_dqtukey_get_type(void);


G_END_DECLS


#endif  /* __GSTLAL_DQTUKEY_H__ */
