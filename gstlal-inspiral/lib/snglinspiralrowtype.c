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


#include <stdlib.h>


#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/TimeSeries.h>


#include <snglinspiralrowtype.h>



/**
 * Allocate a new struct GSTLALSnglInspiral
 *
 * returns:  pointer to newly allocated struct GSTLALSnglInspiral
 */


struct GSTLALSnglInspiral *gstlal_snglinspiral_new(size_t H1_length, size_t K1_length, size_t L1_length, size_t V1_length)
{
	struct GSTLALSnglInspiral *row = calloc(1, sizeof(*row) + (H1_length + K1_length + L1_length + V1_length) * sizeof(row->snr[0]));
	if (row) {
		row->H1_length = H1_length;
		row->K1_length = K1_length;
		row->L1_length = L1_length;
		row->V1_length = V1_length;
	}


	return row;
}


/**
 * Deallocate a struct GSTLALSnglInspiral and the snr COMPLEX8TimeSeries
 * it carries.
 */


void gstlal_snglinspiral_free(struct GSTLALSnglInspiral *row)
{
	free(row);
}
