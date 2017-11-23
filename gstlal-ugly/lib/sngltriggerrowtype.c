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


#include <sngltriggerrowtype.h>


/**
 * Allocate a new struct GSTLALSnglTrigger
 *
 * returns:  pointer to newly allocated struct GSTLALSnglTrigger
 */


struct GSTLALSnglTrigger *gstlal_sngltrigger_new(size_t length)
{
	struct GSTLALSnglTrigger *row = calloc(1, sizeof(*row) + length * sizeof(row->snr[0]));
	if (row)
		row->length = length;

	return row;
}


/**
 * Deallocate a struct GSTLALSnglTrigger and the snr COMPLEX8TimeSeries
 * it carries.
 */


void gstlal_sngltrigger_free(struct GSTLALSnglTrigger *row)
{
	free(row);
}
