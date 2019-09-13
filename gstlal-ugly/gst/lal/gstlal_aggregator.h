/*
 * Copyright (C) 2019 Patrick Godwin, Chad Hanna
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

#ifndef __GSTLAL_AGGREGATOR_H__
#define __GSTLAL_AGGREGATOR_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

#include <gstlal/gstlal.h>
#include <gstlal/gstaudioadapter.h>

G_BEGIN_DECLS


#define GSTLAL_AGGREGATOR_TYPE \
	(gstlal_aggregator_get_type())

#define GSTLAL_AGGREGATOR(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_AGGREGATOR_TYPE, GSTLALAggregator))

#define GSTLAL_AGGREGATOR_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_AGGREGATOR_TYPE, GSTLALAggregatorClass))

#define GST_IS_GSTLAL_AGGREGATOR(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_AGGREGATOR_TYPE))

#define GST_IS_GSTLAL_AGGREGATOR_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_AGGREGATOR_TYPE))


typedef struct _GSTLALAggregator GSTLALAggregator;
typedef struct _GSTLALAggregatorClass GSTLALAggregatorClass;


/**
 * GSTLALAggregator:
 */


struct _GSTLALAggregator {
	GstBaseTransform element;

	GstAudioInfo audio_info;
	GstAudioAdapter *adapter;

	gint inrate;
	gint outrate;
	guint channels;
	gint width;
	
	/* Timestamp and offset bookeeping */
	guint64 t0;
	guint64 offset0;
	guint64 next_input_offset;
	GstClockTime next_input_timestamp;
	guint64 next_output_offset;
	GstClockTime next_output_timestamp;
	gboolean need_discont;
	gboolean need_pretend;
	gboolean last_gap_state;

	/* Variables to control the size of transforms */
	gsize unitsize;
	guint blocksampsin;
	guint blocksampsout;
	guint blockstridein;
	guint blockstrideout;
	gsl_matrix_float *workspace32;
	gsl_matrix *workspace64;
};


/**
 * GSTLALAggregatorClass:
 * @parent_class:  the parent class
 */


struct _GSTLALAggregatorClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_aggregator_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_AGGREGATOR_H__ */
