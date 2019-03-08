#ifndef __GST_LAL_STRING_TRIGGERGEN_H__
#define __GST_LAL_STRING_TRIGGERGEN_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstaudioadapter.h>
#include <lal/LIGOMetadataTables.h>
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_matrix_float.h>


G_BEGIN_DECLS


#define GSTLAL_STRING_TRIGGERGEN_TYPE \
	(gstlal_string_triggergen_get_type())
#define GSTLAL_STRING_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_STRING_TRIGGERGEN_TYPE, GSTLALStringTriggergen))
#define GSTLAL_STRING_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_STRING_TRIGGERGEN_TYPE, GSTLALStringTriggergenClass))
#define GST_IS_GSTLAL_STRING_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_STRING_TRIGGERGEN_TYPE))
#define GST_IS_GSTLAL_STRING_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_STRING_TRIGGERGEN_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALStringTriggergenClass;


typedef struct {
	GstBaseTransform element;

	/*
	 * properties
	 */

	float threshold;
	float cluster;
	char *bank_filename;
	gsl_matrix_float *autocorrelation_matrix;
	gsl_vector_float *autocorrelation_norm;

	/*
	 * input stream
	 */
	
	GstAudioInfo audio_info;
	GstAudioAdapter *adapter;
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/*
	 * trigger state
	 */

	GMutex bank_lock;
	SnglBurst *bank;
	gint num_templates;
	LIGOTimeGPS *last_time;
} GSTLALStringTriggergen;


GType gstlal_string_triggergen_get_type(void);


G_END_DECLS


#endif  /* __GST_LAL_STRING_TRIGGERGEN_H__ */
