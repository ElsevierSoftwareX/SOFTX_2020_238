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


#ifndef __SNGLINSPIRALROWTYPE_H__
#define __SNGLINSPIRALROWTYPE_H__


#include <glib.h>
#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>


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


struct GSTLALSnglInspiral {
	SnglInspiralTable parent;
	LIGOTimeGPS epoch;
	double deltaT;
	size_t H1_length;
	size_t K1_length;
	size_t L1_length;
	size_t V1_length;
	float complex snr[];
};


struct GSTLALSnglInspiral *gstlal_snglinspiral_new(size_t H1_length, size_t K1_length, size_t L1_length, size_t V1_length);
void gstlal_snglinspiral_free(struct GSTLALSnglInspiral *row);


G_END_DECLS
#endif	/* __SNGLINSPIRALROWTYPE_H__ */
