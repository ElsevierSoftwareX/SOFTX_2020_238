/*
 * $Id: ezligolw.h,v 1.4 2008/07/31 08:28:42 kipp Exp $
 *
 * Copyright (C) 2007  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include <stdio.h>
#include <stdint.h>
#include <gstlal/ezxml.h>


struct ligolw_table {
	const char *name;
	char delimiter;
	int n_columns;
	struct ligolw_table_column {
		const char *name;
		struct ligolw_table *table;
		enum ligolw_table_cell_type {
			ligolw_cell_type_char_s,
			ligolw_cell_type_char_v,
			ligolw_cell_type_ilwdchar,
			ligolw_cell_type_ilwdchar_u,
			ligolw_cell_type_blob,
			ligolw_cell_type_lstring,
			ligolw_cell_type_int_2s,
			ligolw_cell_type_int_2u,
			ligolw_cell_type_int_4s,
			ligolw_cell_type_int_4u,
			ligolw_cell_type_int_8s,
			ligolw_cell_type_int_8u,
			ligolw_cell_type_real_4,
			ligolw_cell_type_real_8
		} type;
	} *columns;
	int n_rows;
	struct ligolw_table_row {
		struct ligolw_table *table;
		union ligolw_table_cell {
			int64_t as_int;
			uint64_t as_uint;
			float as_float;
			double as_double;
			const char *as_string;
			const unsigned char *as_blob;
		} *cells;
	} *rows;
};


ezxml_t ligolw_table_get(ezxml_t, const char *);
enum ligolw_table_cell_type ligolw_table_type_name_to_enum(const char *);
const char *ligolw_table_type_enum_to_name(enum ligolw_table_cell_type);
int ligolw_table_default_row_callback(struct ligolw_table *, struct ligolw_table_row, void *);
struct ligolw_table *ligolw_table_parse(ezxml_t, int (*)(struct ligolw_table *, struct ligolw_table_row, void *), void *);
union ligolw_table_cell ligolw_row_get_cell(struct ligolw_table_row, const char *);
void ligolw_table_free(struct ligolw_table *);
int ligolw_table_get_column(struct ligolw_table *, const char *, enum ligolw_table_cell_type *);
int ligolw_table_print(FILE *, struct ligolw_table *);


#define LIGOLW_UNPACKING_REQUIRED 0x1


struct ligolw_unpacking_spec {
	void *dest;
	const char *name;
	enum ligolw_table_cell_type type;
	int flags;
};

int ligolw_unpacking_row_builder(struct ligolw_table *, struct ligolw_table_row, void *);
