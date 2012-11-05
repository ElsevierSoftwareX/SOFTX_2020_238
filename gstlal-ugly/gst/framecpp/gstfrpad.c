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


#include <glib.h>
#include <gst/gst.h>


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
			{GST_FRPAD_TYPE_FRADCDATA, "GST_FRPAD_TYPE_FRADCDATA", "Pad is an FrAdcData stream"},
			{GST_FRPAD_TYPE_FRPROCDATA, "GST_FRPAD_TYPE_FRPROCDATA", "Pad is an FrProcData stream"},
			{GST_FRPAD_TYPE_FRSIMDATA, "GST_FRPAD_TYPE_FRSIMDATA", "Pad is an FrSimData stream"},
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
	PROP_COMMENT
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);

	switch(id) {
	case PROP_PAD_TYPE:
		pad->pad_type = g_value_get_enum(value);
		break;

	case PROP_COMMENT:
		g_free(pad->comment);
		pad->comment = g_value_dup_string(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);

	switch(id) {
	case PROP_PAD_TYPE:
		g_value_set_enum(value, pad->pad_type);
		break;

	case PROP_COMMENT:
		g_value_set_string(value, pad->comment);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void finalize(GObject *object)
{
	GstFrPad *pad = GST_FRPAD(object);

	g_free(pad->comment);
	pad->comment = NULL;

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
}


static void gst_frpad_init(GstFrPad *pad, GstFrPadClass *klass)
{
}
