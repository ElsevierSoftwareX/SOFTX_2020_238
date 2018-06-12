/*
 * Copyright (C) 2010 Leo Singer
 * Copyright (C) 2016 Aaron Viets
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *				     Preamble
 *
 * ============================================================================
 */


/*
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/*
 * Our own stuff
 */


#include <cmath_base.h>


/*
 * ============================================================================
 *
 *				Plugin Entry Point
 *
 * ============================================================================
 */


GType cmath_cabs_get_type(void);
GType cmath_creal_get_type(void);
GType cmath_cimag_get_type(void);
GType cmath_cexp_get_type(void);
GType cmath_cln_get_type(void);
GType cmath_clog_get_type(void);
GType cmath_clog10_get_type(void);
GType cmath_cpow_get_type(void);
GType cmath_lpshiftfreq_get_type(void);


static gboolean
plugin_init (GstPlugin *plugin)
{
	struct
	{
		const gchar *name;
		GType type;
	} *element, elements[] = {
		{
		"cmath_base", CMATH_BASE_TYPE}, {
		"cabs", cmath_cabs_get_type()}, {
		"creal", cmath_creal_get_type()}, {
		"cimag", cmath_cimag_get_type()}, {
		"cexp", cmath_cexp_get_type()}, {
		"cln", cmath_cln_get_type()}, {
		"clog", cmath_clog_get_type()}, {
		"clog10", cmath_clog10_get_type()}, {
		"cpow", cmath_cpow_get_type()}, {
		"lpshiftfreq", cmath_lpshiftfreq_get_type()}, {
		NULL, 0},};

	/*
	 * Tell GStreamer about the elements.
	 */

	for (element = elements; element->name; element++)
		if (!gst_element_register (plugin, element->name, GST_RANK_NONE,
			element->type))
			return FALSE;

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, cmath,
	"Complex arithmetic elements", plugin_init, PACKAGE_VERSION, "GPL",
	PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")
