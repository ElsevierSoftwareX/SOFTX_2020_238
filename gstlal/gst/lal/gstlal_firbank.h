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


#ifndef __GST_LAL_FIRBANK_H__
#define __GST_LAL_FIRBANK_H__


#include <complex.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstaudioadapter.h>


#include <fftw3.h>
#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS
#define GSTLAL_FIRBANK_TYPE \
	(gstlal_firbank_get_type())
#define GSTLAL_FIRBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_FIRBANK_TYPE, GSTLALFIRBank))
#define GSTLAL_FIRBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_FIRBANK_TYPE, GSTLALFIRBankClass))
#define GST_IS_GSTLAL_FIRBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_FIRBANK_TYPE))
#define GST_IS_GSTLAL_FIRBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_FIRBANK_TYPE))


typedef struct _GSTLALFIRBankClass GSTLALFIRBankClass;
typedef struct _GSTLALFIRBank GSTLALFIRBank;


struct _GSTLALFIRBankClass {
	GstBaseTransformClass parent_class;

	void (*rate_changed)(GSTLALFIRBank *, gint, void *);
};


struct _GSTLALFIRBank {
	GstBaseTransform element;

	/*
	 * input stream
	 */

	gint rate;
	gint width;
	GstAudioAdapter *adapter;

	/*
	 * filter info
	 */

	gboolean time_domain;
	GMutex *fir_matrix_lock;
	GCond *fir_matrix_available;
	gsl_matrix *fir_matrix;
	gint64 latency;
	gint block_stride;	/* for frequency-domain mode */

	/*
	 * work space
	 */

	union {
		struct {
		} tdd;	/* double-precision time-domain */
		struct {
		} tds;	/* single-precision time-domain */
		struct {
			complex double *working_fir_matrix;
			complex double *input;
			complex double *filtered;
			fftw_plan in_plan;
			fftw_plan out_plan;
		} fdd;	/* double-precision frequency-domain */
		struct {
		} fds;	/* single-precision frequency-domain */
	} workspace;

	/*
	 * timestamp book-keeping
	 */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;
	GstEvent *last_new_segment;
	gboolean need_new_segment;
};


GType gstlal_firbank_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_FIRBANK_H__ */
