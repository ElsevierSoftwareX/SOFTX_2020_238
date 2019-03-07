/*
 * Copyright (C) 2019  Kipp Cannon
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


#include <gstlal/ezligolw.h>
#include <gstlal_snglburst.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/SnglBurstUtils.h>
#include <string.h>


static int sngl_burst_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	int result_code;
	SnglBurst **head = data;
	SnglBurst *new = LALCalloc(1, sizeof(*new));
	struct ligolw_unpacking_spec spec[] = {
		{&new->process_id, "process_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{&new->event_id, "event_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "ifo", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "search", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "channel", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->start_time.gpsSeconds, "start_time", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->start_time.gpsNanoSeconds, "start_time_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->peak_time.gpsSeconds, "peak_time", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->peak_time.gpsNanoSeconds, "peak_time_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->duration, "duration", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->central_freq, "central_freq", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->bandwidth, "bandwidth", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->amplitude, "amplitude", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->snr, "snr", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->confidence, "confidence", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->chisq, "chisq", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->chisq_dof, "chisq_dof", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{NULL, NULL, -1, 0}
	};

	/* check for memory allocation failure.  remember to clean up row's
	 * memory. */
	if(!new) {
		XLALPrintError("memory allocation failure\n");
		free(row.cells);
		return -1;
	}

	/* unpack.  have to do the strings manually because they get copied
	 * by value rather than reference.  ligolw_unpacking_row_builder()
	 * cleans up row's memory for us. */
	strncpy(new->ifo, ligolw_row_get_cell(row, "ifo").as_string, LIGOMETA_IFO_MAX - 1);
	new->ifo[LIGOMETA_IFO_MAX - 1] = '\0';
	strncpy(new->search, ligolw_row_get_cell(row, "search").as_string, LIGOMETA_SEARCH_MAX - 1);
	new->search[LIGOMETA_SEARCH_MAX - 1] = '\0';
	strncpy(new->channel, ligolw_row_get_cell(row, "channel").as_string, LIGOMETA_CHANNEL_MAX - 1);
	new->channel[LIGOMETA_CHANNEL_MAX - 1] = '\0';

	result_code = ligolw_unpacking_row_builder(table, row, spec);
	if(result_code > 0) {
		/* missing required column */
		XLALPrintError("failure parsing row: missing column \"%s\"\n", spec[result_code - 1].name);
		LALFree(new);
		return -1;
	} else if(result_code < 0) {
		/* column type mismatch */
		XLALPrintError("failure parsing row: incorrect type for column \"%s\"\n", spec[-result_code - 1].name);
		LALFree(new);
		return -1;
	}

	/* add new object to head of linked list.  the linked list is
	 * reversed with respect to the file's contents.  it will be
	 * reversed again below */
	new->next = *head;
	*head = new;

	/* success */
	return 0;
}


int gstlal_snglburst_array_from_file(const char *filename, SnglBurst **bankarray)
{
	SnglBurst *head = NULL;
	SnglBurst *row;
	ezxml_t xmldoc;
	ezxml_t elem;
	struct ligolw_table *table;
	int num = 0;

	/*
	 * so there's no confusion in case of error
	 */

	*bankarray = NULL;

	/*
	 * parse the document
	 */

	g_assert(filename != NULL);
	xmldoc = ezxml_parse_file(filename);
	if(!xmldoc) {
		XLALPrintError("%s(): error parsing \"%s\"\n", __func__, filename);
		goto parsefailed;
	}

	/*
	 * load sngl_burst table.
	 */

	elem = ligolw_table_get(xmldoc, "sngl_burst");
	if(elem) {
		table = ligolw_table_parse(elem, sngl_burst_row_callback, &head);
		if(!table) {
			XLALPrintError("%s(): failure parsing sngl_burst table in \"%s\"\n", __func__, filename);
			goto tablefailed;
		}
		ligolw_table_free(table);
	} else {
		XLALPrintError("%s(): no sngl_burst table in \"%s\"\n", __func__, filename);
		goto tablefailed;
	}

	/*
	 * clean up
	 */

	ezxml_free(xmldoc);

	/*
	 * count rows.  can't use table->n_rows because the callback
	 * interecepted the rows, and the table object is empty
	 */

	for(num = 0, row = head; row; row = row->next, num++);

	/*
	 * copy the linked list of templates into the template array in
	 * reverse order.  the linked list is reversed with respect to the
	 * contents of the file, so this constructs an array of templates
	 * in the order in which they appear in the file.
	 */

	*bankarray = calloc(num, sizeof(**bankarray));
	for(row = &(*bankarray)[num - 1]; head; row--) {
		SnglBurst *next = head->next;
		*row = *head;
		LALFree(head);
		head = next;
	}

	/*
	 * success
	 */

	return num;

	/*
	 * error
	 */

tablefailed:
	ezxml_free(xmldoc);
parsefailed:
	return -1;
}


void gstlal_snglburst_array_free(SnglBurst *bankarray)
{
	free(bankarray);
}
