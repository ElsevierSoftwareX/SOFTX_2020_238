/*
 * Copyright (C) 2011 Chad Hanna <chad.hanna@ligo.org>, Kipp Cannon <kipp.cannon@ligo.org>, Drew Keppel <drew.keppel@ligo.org>, 2018 Cody Messick <cody.messick@ligo.org>, Alex Pace <email here>
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

#ifndef __GSTLAL_ITACAC_H__
#define __GSTLAL_ITACAC_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstaggregator.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LALDatatypes.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_float.h>

G_BEGIN_DECLS

#define GSTLAL_ITACAC_PAD_TYPE \
	(gstlal_itacac_pad_get_type())
#define GSTLAL_ITACAC_PAD(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_ITACAC_PAD_TYPE, GSTLALItacacPad))
#define GSTLAL_ITACAC_PAD_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_ITACAC_PAD_TYPE, GSTLALItacacPadClass))
#define GSTLAL_ITACAC_PAD_GET_CLASS(obj) \
	 (G_TYPE_INSTANCE_GET_CLASS ((obj),GSTLAL_ITACAC_PAD_TYPE,GSTLALItacacPadClass))
#define GST_IS_GSTLAL_ITACAC_PAD(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_ITACAC_PAD_TYPE))
#define GST_IS_GSTLAL_ITACAC_PAD_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_ITACAC_PAD_TYPE))


#define GSTLAL_ITACAC_TYPE \
	(gstlal_itacac_get_type())
#define GSTLAL_ITACAC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_ITACAC_TYPE, GSTLALItacac))
#define GSTLAL_ITACAC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_ITACAC_TYPE, GSTLALItacacClass))
#define GSTLAL_ITACAC_GET_CLASS(obj) \
	 (G_TYPE_INSTANCE_GET_CLASS ((obj),GSTLAL_ITACAC_TYPE,GSTLALItacacClass))
#define GST_IS_GSTLAL_ITACAC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_ITACAC_TYPE))
#define GST_IS_GSTLAL_ITACAC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_ITACAC_TYPE))

struct data_container {
	gsl_matrix *duration_dataoffset_trigwindowoffset_peakfindinglength_matrix;
	void *data;
};


typedef struct {
	GstAggregatorPadClass parent_class;
} GSTLALItacacPadClass;

typedef struct {
	GstAggregatorPad aggpad;

	GstAudioAdapter *adapter;
	guint adapter_unit_size;
	struct data_container *data;
	void *chi2;
	void *tmp_chi2;
	char *bank_filename;
	char *instrument;
	char *channel_name;
	GMutex bank_lock;
	guint n;
	struct gstlal_peak_state *maxdata;
	struct gstlal_peak_state *tmp_maxdata;
	gdouble snr_thresh;
	gsl_matrix_complex *autocorrelation_matrix;
	gsl_matrix_int *autocorrelation_mask;
	gsl_vector *autocorrelation_norm;
	void *snr_mat;
	void *tmp_snr_mat;
	gsl_matrix_complex_float_view snr_matrix_view;
	gsl_matrix_complex_float_view tmp_snr_matrix_view;
	SnglInspiralTable *bankarray;
	gboolean EOS;
	gboolean waiting;

	GstClockTime initial_timestamp;
	guint samples_available_for_padding;

} GSTLALItacacPad;

typedef struct {
	GstAggregatorClass parent_class;
} GSTLALItacacClass;

typedef struct {
	// Required by base class
	GstAggregator aggregator;

	// itacac's members
	guint rate;
	guint channels;
	gstlal_peak_type_specifier peak_type;
	unsigned autocorrelation_length;
	GstClockTime initial_output_timestamp;
	guint64 next_output_offset;
	GstClockTime next_output_timestamp;
	GstClockTimeDiff difftime;
	gboolean EOS;
	GSTLALItacacPad *H1_itacacpad, *K1_itacacpad, *L1_itacacpad, *V1_itacacpad, *G1_itacacpad;
	GMutex caps_lock;
} GSTLALItacac;

GType gstlal_itacac_get_type(void);
GType gstlal_itacac_pad_get_type(void);

G_END_DECLS

#endif	/* __GSTLAL_ITACAC_H__ */
