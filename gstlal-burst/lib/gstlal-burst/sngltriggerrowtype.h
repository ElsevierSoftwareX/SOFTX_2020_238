/*
 * Copyright (C) 2016  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifndef __SNGLTRIGGERROWTYPE_H__
#define __SNGLTRIGGERROWTYPE_H__


#include <glib.h>
#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>

#include <gstlal-burst/gstlal_sngltrigger.h>

G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                       Custom sngl_inspiral row type
 *
 * ============================================================================
 */


/*
 * adds room for an SNR snippet
 */


struct GSTLALSnglTrigger {
	SnglTriggerTable parent;
	LIGOTimeGPS epoch;
	double deltaT;
	size_t length;
	float complex snr[];
};


struct GSTLALSnglTrigger *gstlal_sngltrigger_new(size_t length);
void gstlal_sngltrigger_free(struct GSTLALSnglTrigger *row);


G_END_DECLS
#endif	/* __SNGLTRIGGERROWTYPE_H__ */
