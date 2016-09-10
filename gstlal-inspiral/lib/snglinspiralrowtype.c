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


#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/TimeSeries.h>


#include <snglinspiralrowtype.h>


/**
 * Allocate a new struct GSTLALSnglInspiral
 *
 * returns:  pointer to newly allocated struct GSTLALSnglInspiral
 */


struct GSTLALSnglInspiral *gstlal_snglinspiral_new(void)
{
	struct GSTLALSnglInspiral *row = calloc(1, sizeof(*row));

	return row;
}


/**
 * Deallocate a struct GSTLALSnglInspiral and the snr COMPLEX8TimeSeries
 * it carries.
 */


void gstlal_snglinspiral_free(struct GSTLALSnglInspiral *row)
{
	if(row) {
		XLALDestroyCOMPLEX8TimeSeries(row->snr);
		row->snr = NULL;
	}
	free(row);
}


/**
 * Set the snr pointer of a struct GSTLALSnglInspiral.  Deallocates any
 * previously set snr COMPLEX8TimeSeries and takes ownership of the
 * COMPLEX8TimeSeries passed to this function.  Do not free the
 * COMPLEX8TimeSeries afterwards.  Passing NULL for the snr pointer is OK.
 *
 * returns:  pointer to the struct GSTLALSnglInspiral.
 */


struct GSTLALSnglInspiral *gstlal_snglinspiral_set_snr(struct GSTLALSnglInspiral *row, COMPLEX8TimeSeries *snr)
{
	XLALDestroyCOMPLEX8TimeSeries(row->snr);
	row->snr = snr;
	return row;
}
