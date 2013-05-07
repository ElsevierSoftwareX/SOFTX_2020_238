/*
 * GstFrPad
 *
 * Copyright (C) 2012  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <string.h>


/*
 * stuff from GObject/GStreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_tags.h>
#include <gstfrpad.h>



/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


GST_BOILERPLATE(GstFrPad, gst_frpad, GstPad, GST_TYPE_PAD);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_PAD_TYPE GST_FRPAD_TYPE_FRPROCDATA
#define DEFAULT_COMMENT ""
#define DEFAULT_INSTRUMENT NULL
#define DEFAULT_CHANNEL_NAME NULL
#define DEFAULT_CHANNEL_GROUP 0
#define DEFAULT_CHANNEL_NUMBER 0
#define DEFAULT_NBITS 1	/* FIXME:  is there a "not set" value?  -1? */
#define DEFAULT_UNITS ""


/*
 * ============================================================================
 *
 *                               Pad Type Enum
 *
 * ============================================================================
 */


GType gst_frpad_type_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GST_FRPAD_TYPE_FRADCDATA, "FrAdcData", "Pad is an FrAdcData stream"},
			{GST_FRPAD_TYPE_FRPROCDATA, "FrProcData", "Pad is an FrProcData stream"},
			{GST_FRPAD_TYPE_FRSIMDATA, "FrSimData", "Pad is an FrSimData stream"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GST_FRPAD_TYPE", values);
	}

	return type;
}


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


/*
 * tags
 */


static gint get_bitrate(GstFrPad *pad)
{
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstStructure *s;
	gint samples_per_second, channels, bits_per_sample;
	gboolean success = caps != NULL;

	if(success)
		success = (s = gst_caps_get_structure(caps, 0)) != NULL;
	if(success) {
		success &= gst_structure_get_int(s, "rate", &samples_per_second);
		success &= gst_structure_get_int(s, "channels", &channels);
		if(!gst_structure_get_int(s, "depth", &bits_per_sample))
			success &= gst_structure_get_int(s, "width", &bits_per_sample);
	}

	return success ? bits_per_sample * samples_per_second * channels : -1;
}


static gboolean update_tag_list(GstFrPad *pad)
{
	GstTagList *new_tags = gst_tag_list_new_full(
		GST_TAG_CODEC, "RAW",
		GST_TAG_TITLE, GST_PAD_NAME(GST_PAD_CAST(pad)),
		GSTLAL_TAG_INSTRUMENT, pad->instrument && strcmp(pad->instrument, "") ? pad->instrument : " ",
		GSTLAL_TAG_CHANNEL_NAME, pad->channel_name && strcmp(pad->channel_name, "") ? pad->channel_name : " ",
		/*GST_TAG_GEO_LOCATION_NAME, observatory,
		GST_TAG_GEO_LOCATION_SUBLOCATION, pad->instrument,*/
		GSTLAL_TAG_UNITS, pad->units && strcmp(pad->units, "") ? pad->units : " ",
		NULL
	);
	gint bitrate = get_bitrate(pad);

	if(!new_tags) {
		GST_ERROR_OBJECT(pad, "failed to update tags");
		g_assert_not_reached();	/* can be compiled out */
		return FALSE;
	} else if(bitrate >= 0)
		gst_tag_list_add(new_tags, GST_TAG_MERGE_REPLACE, GST_TAG_BITRATE, bitrate, NULL);

	gst_tag_list_free(pad->tags);
	pad->tags = new_tags;
	return TRUE;
}


static void caps_notify_handler(GObject *object, GParamSpec *pspec, gpointer user_data)
{
	gboolean got_new_tags;

	GST_OBJECT_LOCK(object);
	got_new_tags = update_tag_list(GST_FRPAD(object));
	GST_OBJECT_UNLOCK(object);

	if(got_new_tags)
		g_object_notify(object, "tags");
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * GstFrPad version of gst_pad_new_from_template()
 */


GstFrPad *gst_frpad_new_from_template(GstPadTemplate *templ, const gchar *name)
{
	g_return_val_if_fail(GST_IS_PAD_TEMPLATE(templ), NULL);

	return g_object_new(GST_FRPAD_TYPE, "name", name, "direction", templ->direction, "template", templ, NULL);
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	PROP_PAD_TYPE = 1,
	PROP_COMMENT,
	PROP_INSTRUMENT,
	PROP_CHANNEL_NAME,
	PROP_CHANNEL_GROUP,
	PROP_CHANNEL_NUMBER,
	PROP_NBITS,
	PROP_UNITS,
	PROP_TAGS,
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);
	gboolean got_new_tags = FALSE;

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_PAD_TYPE:
		pad->pad_type = g_value_get_enum(value);
		break;

	case PROP_COMMENT:
		g_free(pad->comment);
		pad->comment = g_value_dup_string(value);
		break;

	case PROP_INSTRUMENT:
		g_free(pad->instrument);
		pad->instrument = g_value_dup_string(value);
		got_new_tags = update_tag_list(pad);
		break;

	case PROP_CHANNEL_NAME:
		g_free(pad->channel_name);
		pad->channel_name = g_value_dup_string(value);
		got_new_tags = update_tag_list(pad);
		break;

	case PROP_CHANNEL_GROUP:
		pad->channel_group = g_value_get_uint(value);
		break;

	case PROP_CHANNEL_NUMBER:
		pad->channel_number = g_value_get_uint(value);
		break;

	case PROP_NBITS:
		pad->nbits = g_value_get_uint(value);
		break;

	case PROP_UNITS:
		g_free(pad->units);
		pad->units = g_value_dup_string(value);
		got_new_tags = update_tag_list(pad);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);

	if(got_new_tags)
		g_object_notify(G_OBJECT(pad), "tags");
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_PAD_TYPE:
		g_value_set_enum(value, pad->pad_type);
		break;

	case PROP_COMMENT:
		g_value_set_string(value, pad->comment);
		break;

	case PROP_INSTRUMENT:
		g_value_set_string(value, pad->instrument);
		break;

	case PROP_CHANNEL_NAME:
		g_value_set_string(value, pad->channel_name);
		break;

	case PROP_CHANNEL_GROUP:
		g_value_set_uint(value, pad->channel_group);
		break;

	case PROP_CHANNEL_NUMBER:
		g_value_set_uint(value, pad->channel_number);
		break;

	case PROP_NBITS:
		g_value_set_uint(value, pad->nbits);
		break;

	case PROP_UNITS:
		g_value_set_string(value, pad->units);
		break;

	case PROP_TAGS:
		g_value_set_boxed(value, pad->tags);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);
}


static void finalize(GObject *object)
{
	GstFrPad *pad = GST_FRPAD(object);

	g_free(pad->comment);
	pad->comment = NULL;
	g_free(pad->instrument);
	pad->instrument = NULL;
	g_free(pad->channel_name);
	pad->channel_name = NULL;
	g_free(pad->units);
	pad->units = NULL;
	gst_tag_list_free(pad->tags);
	pad->tags = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void gst_frpad_base_init(gpointer klass)
{
	/* no-op */
}


static void gst_frpad_class_init(GstFrPadClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		PROP_PAD_TYPE,
		g_param_spec_enum(
			"pad-type",
			"Pad type",
			"Pad type.",
			GST_FRPAD_TYPE_TYPE,
			DEFAULT_PAD_TYPE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_COMMENT,
		g_param_spec_string(
			"comment",
			"Comment",
			"Comment field.  Validity:  FrAdcData, FrProcData, FrSimData.",
			DEFAULT_COMMENT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_INSTRUMENT,
		g_param_spec_string(
			"instrument",
			"Instrument",
			"Instrument name.  Not used for frame metadata.",
			DEFAULT_INSTRUMENT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Channel name.  Not used for frame metadata.",
			DEFAULT_CHANNEL_NAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_CHANNEL_GROUP,
		g_param_spec_uint(
			"channel-group",
			"Channel group",
			"Channel group.  Validity:  FrAdcData.",
			0, G_MAXUINT, DEFAULT_CHANNEL_GROUP,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_CHANNEL_NUMBER,
		g_param_spec_uint(
			"channel-number",
			"Channel number",
			"Channel number.  Validity:  FrAdcData.",
			0, G_MAXUINT, DEFAULT_CHANNEL_NUMBER,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_NBITS,
		g_param_spec_uint(
			"nbits",
			"Number of bits",
			"Number of bits in A/D output.  Validity:  FrAdcData.",
			1, G_MAXUINT, DEFAULT_NBITS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_UNITS,
		g_param_spec_string(
			"units",
			"Units",
			"Units.  Validity:  FrAdcData, FrProcData, FrSimData.",
			DEFAULT_UNITS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_TAGS,
		g_param_spec_boxed(
			"tags",
			"Tag list",
			"Tag list.",
			GST_TYPE_TAG_LIST,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void gst_frpad_init(GstFrPad *pad, GstFrPadClass *klass)
{
	pad->tags = gst_tag_list_new();
	g_signal_connect_after(pad, "notify::caps", (GCallback) caps_notify_handler, NULL);
}
