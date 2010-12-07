#ifndef __GST_LAL_SPECGRAM_H__
#define __GST_LAL_SPECGRAM_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <fftw3.h>

G_BEGIN_DECLS


#define GSTLAL_SPECGRAM_TYPE \
	(gstlal_specgram_get_type())
#define GSTLAL_SPECGRAM(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SPECGRAM_TYPE, GSTLALSpecgram))
#define GSTLAL_SPECGRAM_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SPECGRAM_TYPE, GSTLALSpecgramClass))
#define GST_IS_GSTLAL_SPECGRAM(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SPECGRAM_TYPE))
#define GST_IS_GSTLAL_SPECGRAM_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SPECGRAM_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALSpecgramClass;


typedef struct {
	GstBaseTransform element;

	GMutex *fir_matrix_lock;

	/* input stream */

	guint32 rate;
	GstAdapter *adapter;

	/* output stream */

	guint32 outrate;
	
	/* FFT parameters */

	guint32 n;

	double * infft;
	fftw_complex *outfft;
	double * workspacefft;
	fftw_plan fftplan;

	/* timestamp book-keeping */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_out_offset;
	gboolean need_discont;
} GSTLALSpecgram;


GType gstlal_specgram_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_SPECGRAM_H__ */
