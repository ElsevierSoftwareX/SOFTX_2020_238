/*
 * $Id: ezligolw.c,v 1.4 2008/07/31 08:28:42 kipp Exp $
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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ezligolw.h>


/*
 * Extract the meaningful portion of a table name.  Returns a pointer to
 * the last colon-delimited substring before an optional ":table" suffix.
 */


static const char *ligolw_strip_table_name(const char *Name)
{
	char buff[strlen(Name) + 1];
	char *pos = buff;
	char *start;

	strcpy(buff, Name);

	do
		start = strsep(&pos, ":");
	while(pos && strncmp(pos, "table", 5));

	return Name + (start - buff);
}


/*
 * Extract the meaningful portion of a column name.  Returns a pointer to
 * the last colon-delimited substring.
 */


static const char *ligolw_strip_column_name(const char *Name)
{
	char buff[strlen(Name) + 1];
	char *pos = buff;
	char *start;

	strcpy(buff, Name);

	do
		start = strsep(&pos, ":");
	while(pos);

	return Name + (start - buff);
}


/*
 * Convert a LIGO Light Weight type name string to/from a numeric type
 * index.
 */


static const struct name_to_enum {
	const char *name;
	enum ligolw_table_cell_type type;
} name_to_enum[] = {
	{"char_s", ligolw_cell_type_char_s},
	{"char_v", ligolw_cell_type_char_v},
	{"ilwd:char", ligolw_cell_type_ilwdchar},
	{"ilwd:char_u", ligolw_cell_type_ilwdchar_u},
	{"lstring", ligolw_cell_type_lstring},
	{"string", ligolw_cell_type_lstring},
	{"int_2s", ligolw_cell_type_int_2s},
	{"int_2u", ligolw_cell_type_int_2u},
	{"int_4s", ligolw_cell_type_int_4s},
	{"int", ligolw_cell_type_int_4s},
	{"int_4u", ligolw_cell_type_int_4u},
	{"int_8s", ligolw_cell_type_int_8s},
	{"int_8u", ligolw_cell_type_int_8u},
	{"real_4", ligolw_cell_type_real_4},
	{"float", ligolw_cell_type_real_4},
	{"real_8", ligolw_cell_type_real_8},
	{"double", ligolw_cell_type_real_8},
	{NULL, -1}
};


enum ligolw_table_cell_type ligolw_table_type_name_to_enum(const char *name)
{
	const struct name_to_enum *n_to_e;

	for(n_to_e = name_to_enum; n_to_e->name; n_to_e++)
		if(!strcmp(n_to_e->name, name))
			/* found it */
			return n_to_e->type;

	/* unrecognized type */
	return -1;
}


const char *ligolw_table_type_enum_to_name(enum ligolw_table_cell_type t)
{
	const struct name_to_enum *n_to_e;

	for(n_to_e = name_to_enum; n_to_e->name; n_to_e++)
		if(n_to_e->type == t)
			/* found it */
			return n_to_e->name;

	/* unrecognized type */
	return NULL;
}


/*
 * Default row builder call-back.
 */


int ligolw_table_default_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *ignored)
{
	table->rows = realloc(table->rows, (table->n_rows + 1) * sizeof(*table->rows));
	table->rows[table->n_rows] = row;
	table->n_rows++;
	return 0;
}


/*
 * Parse an ezxml_t Table element into a struct ligolw_table structure.  If
 * row_callback() is NULL, then the default row builder is used, which
 * inserts the rows into the ligolw_table structure.  Calling code can
 * provide it's own function, which will be called after each row is
 * constructed.  This allows the rows to be "intercepted", so that some
 * other thing can be done with them other than being inserted into the
 * ligolw_table.  The call-back function will be passed the pointer to the
 * current ligolw_table structure as its first argument, the pointer to the
 * new row as its second, and the callback_data pointer as its third
 * argument.  The row_callback() function must free the row's cells element
 * if it will not be saving it, or memory will be leaked.  The call-back
 * returns 0 to indicate success, non-zero to indicate failure.
 *
 * ligolw_table_parse() return the pointer to the new struct ligolw_table
 * structure on success, NULL on failure.
 */


static void next_token(char **start, char **end, char **next_start, char delimiter)
{
	char *c;

	/* find the token's start */
	for(c = *start; *c && isspace(*c) && *c != delimiter && *c != '"'; c++);

	/* quoted token */
	if(*c == '"') {
		/* start at first character next to quote charater '"' */
		*start = ++c;
		/* end at '\0' or '"' */
		for(; *c && *c != '"'; c++);
		*end = c;
		/* find the delimiter, this marks the end of current token */
		if(*c == '"')
			c++;
		for(; *c && isspace(*c) && *c != delimiter; c++);
	}
	/* token has zero length */
	else if(!*c || *c == delimiter) {
		/* at the delimiter, this marks the end of current token */
		*start = *end = c;
	}
	/* unquoted token */
	else {
		/* start at first non-white space and non-quote character */
		*start = c;
		/* end at space or delimiter or '\0' */
		for(++c; *c && !isspace(*c) && *c != delimiter; c++);
		*end = c;
		/* find the delimiter, this marks the end of current token */
		for(; *c && isspace(*c) && *c != delimiter; c++);
	}

	/* skip the delimiter and white spaces and go to next start */
	if(*c == delimiter)
		c++;
	for(; *c && isspace(*c) && *c != delimiter; c++);
	*next_start = c;
}


struct ligolw_table *ligolw_table_parse(ezxml_t elem, int (row_callback)(struct ligolw_table *, struct ligolw_table_row, void *), void *callback_data)
{
	struct ligolw_table *table;
	char *txt;
	ezxml_t column;
	ezxml_t stream;

	table = malloc(sizeof(*table));
	if(!table)
		return NULL;

	table->name = ligolw_strip_table_name(ezxml_attr(elem, "Name"));

	table->n_columns = 0;
	table->columns = NULL;
	table->n_rows = 0;
	table->rows = NULL;

	for(column = ezxml_child(elem, "Column"); column; column = column->next) {
		table->columns = realloc(table->columns, (table->n_columns + 1) * sizeof(*table->columns));

		table->columns[table->n_columns].name = ligolw_strip_column_name(ezxml_attr(column, "Name"));
		table->columns[table->n_columns].table = table;
		table->columns[table->n_columns].type = ligolw_table_type_name_to_enum(ezxml_attr(column, "Type"));

		table->n_columns++;
	}

	stream = ezxml_child(elem, "Stream");
	if(!stream) {
		/* DTD allows Table to have 0 Stream children */
		table->delimiter = '\0';
		return table;
	}

	table->delimiter = *ezxml_attr(stream, "Delimiter");

	if(!row_callback)
		row_callback = ligolw_table_default_row_callback;

	for(txt = stream->txt; txt && *txt; ) {
		struct ligolw_table_row row;
		int c;

		row.table = table;
		row.cells = malloc(table->n_columns * sizeof(*row.cells));

		for(c = 0; c < table->n_columns; c++) {
			char *end, *next;

			next_token(&txt, &end, &next, table->delimiter);

			switch(table->columns[c].type) {
			case ligolw_cell_type_char_s:
			case ligolw_cell_type_char_v:
			case ligolw_cell_type_ilwdchar:
			case ligolw_cell_type_ilwdchar_u:
			case ligolw_cell_type_blob:
			case ligolw_cell_type_lstring:
				/* FIXME: move into a separate buffer so
				 * that the original document is not
				 * modified (see null terminator below) */
				/* FIXME: binary types need to be sent
				 * through a decoder following this */
				row.cells[c].as_string = txt;
				break;

			case ligolw_cell_type_int_2s:
			case ligolw_cell_type_int_4s:
			case ligolw_cell_type_int_8s:
				row.cells[c].as_int = strtoll(txt, NULL, 0);
				break;

			case ligolw_cell_type_int_2u:
			case ligolw_cell_type_int_4u:
			case ligolw_cell_type_int_8u:
				row.cells[c].as_uint = strtoull(txt, NULL, 0);
				break;

			case ligolw_cell_type_real_4:
				row.cells[c].as_float = strtod(txt, NULL);
				break;

			case ligolw_cell_type_real_8:
				row.cells[c].as_double = strtod(txt, NULL);
				break;
			}

			/* null-terminate current token.  this does not
			 * interfer with the exit test for the loop over
			 * txt because end and next can only point to the
			 * same address if that address is the end of the
			 * text */
			*end = '\0';

			/* advance to next token */
			txt = next;
		}

		if(row_callback(table, row, callback_data)) {
			ligolw_table_free(table);
			return NULL;
		}
	}

	return table;
}


/*
 * Free a struct ligolw_table.
 */


void ligolw_table_free(struct ligolw_table *table)
{

	if(table) {
		int i;
		for(i = 0; i < table->n_rows; i++)
			free(table->rows[i].cells);
		free(table->rows);
		free(table->columns);
	}
	free(table);
}


/*
 * Get a column index by name from within a table.  Returns the index of
 * the column within table's columns array (and thus of the corresponding
 * cell within each row's cell array) or -1 on failure.  If type is not
 * NULL, the place it points to is set to the columns's table_cell_type.
 */


int ligolw_table_get_column(struct ligolw_table *table, const char *name, enum ligolw_table_cell_type *type)
{
	int i;

	for(i = 0; i < table->n_columns; i++)
		if(!strcmp(table->columns[i].name, name)) {
			/* found it */
			if(type)
				*type = table->columns[i].type;
			return i;
		}

	/* couldn't find that column name */
	if(type)
		*type = -1;
	return -1;
}


/*
 * Retrieve the value stored in a cell within a table row.  No error
 * checking is done, you should ensure the requested column is present
 * before calling this function.
 */


union ligolw_table_cell ligolw_row_get_cell(struct ligolw_table_row row, const char *name)
{
	return row.cells[ligolw_table_get_column(row.table, name, NULL)];
}


/*
 * Find an ezxml_t Table element in a document.
 */


ezxml_t ligolw_table_get(ezxml_t xmldoc, const char *table_name)
{
	int n = strlen(table_name);
	ezxml_t table;

	for(table = ezxml_child(xmldoc, "Table"); table; table = table->next)
		if(!strncmp(ligolw_strip_table_name(ezxml_attr(table, "Name")), table_name, n))
			break;

	return table;
}


/*
 * Generic unpacking row builder.
 */


int ligolw_unpacking_row_builder(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	struct ligolw_unpacking_spec *spec;

	for(spec = data; spec->name; spec++) {
		int c;
		enum ligolw_table_cell_type type;
		if((c = ligolw_table_get_column(table, spec->name, &type)) < 0) {
			/* no column by that name */
			if(!(spec->flags & LIGOLW_UNPACKING_REQUIRED))
				/* not required */
				continue;
			free(row.cells);
			return spec - (struct ligolw_unpacking_spec *) data + 1;
		}
		if(spec->type != type) {
			/* type mismatch */
			free(row.cells);
			return -(spec - (struct ligolw_unpacking_spec *) data + 1);
		}
		if(!spec->dest)
			/* column has a valid name and the correct type,
			 * but is ignored */
			continue;

		switch(spec->type) {
		case ligolw_cell_type_char_s:
		case ligolw_cell_type_char_v:
		case ligolw_cell_type_ilwdchar:
		case ligolw_cell_type_ilwdchar_u:
		case ligolw_cell_type_lstring:
			*(const char **) spec->dest = row.cells[c].as_string;
			break;

		case ligolw_cell_type_blob:
			*(const unsigned char **) spec->dest = row.cells[c].as_blob;
			break;

		case ligolw_cell_type_int_2s:
			*(int16_t *) spec->dest = row.cells[c].as_int;
			break;

		case ligolw_cell_type_int_2u:
			*(int16_t *) spec->dest = row.cells[c].as_uint;
			break;

		case ligolw_cell_type_int_4s:
			*(int32_t *) spec->dest = row.cells[c].as_int;
			break;

		case ligolw_cell_type_int_4u:
			*(uint32_t *) spec->dest = row.cells[c].as_uint;
			break;

		case ligolw_cell_type_int_8s:
			*(int64_t *) spec->dest = row.cells[c].as_int;
			break;

		case ligolw_cell_type_int_8u:
			*(uint64_t *) spec->dest = row.cells[c].as_uint;
			break;

		case ligolw_cell_type_real_4:
			*(float *) spec->dest = row.cells[c].as_float;
			break;

		case ligolw_cell_type_real_8:
			*(double *) spec->dest = row.cells[c].as_double;
			break;
		}
	}

	free(row.cells);

	if(spec - (struct ligolw_unpacking_spec *) data != table->n_columns) {
		/* table has more columns than allowed */
		/* FIXME:  if this is an error, return an error code */
		/* FIXME:  this test doesn't work if the same column gets
		 * unpacked into more than one location, which the design
		 * of the loop above would allow */
	}

	return 0;
}


/*
 * Print a struct ligolw_table structure
 */


int ligolw_table_print(FILE *f, struct ligolw_table *table)
{
	char short_name[strlen(table->name) + 1];
	int r, c;

	/* create a version of the table name with the optional :table
	 * suffix removed */
	strcpy(short_name, table->name);
	{
	char *x = strchr(short_name, ':');
	if(x)
		*x = '\0';
	}

	/* print the table metadata */
	fprintf(f, "<Table Name=\"%s\">\n", table->name);
	for(c = 0; c < table->n_columns; c++)
		fprintf(f, "\t<Column Name=\"%s:%s\" Type=\"%s\"/>\n", short_name, table->columns[c].name, ligolw_table_type_enum_to_name(table->columns[c].type));
	fprintf(f, "\t<Stream Name=\"%s\" Type=\"Local\" Delimiter=\"%c\">\n", table->name, table->delimiter);

	/* print the rows */
	for(r = 0; r < table->n_rows; r++) {
		if(r)
			fprintf(f, ",\n\t\t");
		else
			fprintf(f, "\t\t");
		for(c = 0; c < table->n_columns; c++) {
			if(c)
				fprintf(f, "%c", table->delimiter);

			switch(table->columns[c].type) {
			case ligolw_cell_type_char_s:
			case ligolw_cell_type_char_v:
			case ligolw_cell_type_ilwdchar:
			case ligolw_cell_type_ilwdchar_u:
			case ligolw_cell_type_blob:
			case ligolw_cell_type_lstring:
				/* FIXME: binary types need to pass through
				 * encoders first */
				/* FIXME: string types need to pass through
				 * encoders first */
				fprintf(f, "\"%s\"", table->rows[r].cells[c].as_string);
				break;

			case ligolw_cell_type_int_2s:
			case ligolw_cell_type_int_4s:
			case ligolw_cell_type_int_8s:
				fprintf(f, "%lld", (long long) table->rows[r].cells[c].as_int);
				break;

			case ligolw_cell_type_int_2u:
			case ligolw_cell_type_int_4u:
			case ligolw_cell_type_int_8u:
				fprintf(f, "%llu", (unsigned long long) table->rows[r].cells[c].as_uint);
				break;

			case ligolw_cell_type_real_4:
				fprintf(f, "%.7g", (double) table->rows[r].cells[c].as_float);
				break;

			case ligolw_cell_type_real_8:
				fprintf(f, "%.16g", table->rows[r].cells[c].as_double);
				break;
			}
		}
	}

	/* finish 'er off */
	fprintf(f, "\n\t</Stream>\n</Table>\n");

	return 0;
}
