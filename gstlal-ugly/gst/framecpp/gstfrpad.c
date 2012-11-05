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
	PROP_DUMMY = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);

	switch(id) {
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstFrPad *pad = GST_FRPAD(object);

	switch(id) {
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void gst_frpad_base_init(gpointer klass)
{
	/* no-op */
}


static void gst_frpad_class_init(GstFrPadClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
}


static void gst_frpad_init(GstFrPad *pad, GstFrPadClass *klass)
{
}
