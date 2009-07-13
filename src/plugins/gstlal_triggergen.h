/*
 * An SNR time series sink that produces LIGOLwXML files of triggers.
 *
 * Copyright (C) 2008  Chad Hanna, Kipp Cannon
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_TRIGGERGEN_H__
#define __GSTLAL_TRIGGERGEN_H__


#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/base/gstcollectpads.h>
#include <gstlalcollectpads.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOMetadataTables.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                             Trigger Generator
 *
 * ============================================================================
 */


#define GSTLAL_TRIGGERGEN_TYPE \
	(gstlal_triggergen_get_type())
#define GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGen))
#define GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGenClass))
#define GST_IS_GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TRIGGERGEN_TYPE))
#define GST_IS_GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TRIGGERGEN_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALTriggerGenClass;


typedef struct {
	GstElement element;

	GstCollectPads *collect;

	GstPad *snrpad;
	GstLALCollectData *snrcollectdata;
	GstPad *chisqpad;
	GstLALCollectData *chisqcollectdata;
	GstPad *srcpad;

	gboolean segment_pending;
	GstSegment segment;
	guint64 segment_position;

	int rate;

	char *bank_filename;
	SnglInspiralTable *bank;
	int num_templates;
	double snr_thresh;
} GSTLALTriggerGen;


GType gstlal_triggergen_get_type(void);


/*
 * ============================================================================
 *
 *                             Trigger XML Writer
 *
 * ============================================================================
 */


#define GSTLAL_TRIGGERXMLWRITER_TYPE \
	(gstlal_triggerxmlwriter_get_type())
#define GSTLAL_TRIGGERXMLWRITER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TRIGGERXMLWRITER_TYPE, GSTLALTriggerXMLWriter))
#define GSTLAL_TRIGGERXMLWRITER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TRIGGERXMLWRITER_TYPE, GSTLALTriggerXMLWriterClass))
#define GST_IS_GSTLAL_TRIGGERXMLWRITER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TRIGGERXMLWRITER_TYPE))
#define GST_IS_GSTLAL_TRIGGERXMLWRITER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TRIGGERXMLWRITER_TYPE))


typedef struct {
	GstBaseSinkClass parent_class;
} GSTLALTriggerXMLWriterClass;


typedef struct {
	GstBaseSink element;
	char *location;
	LIGOLwXMLStream *xml;
} GSTLALTriggerXMLWriter;


GType gstlal_triggerxmlwriter_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TRIGGERGEN_H__ */
