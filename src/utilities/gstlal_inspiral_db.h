#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>

int int_inspiral_column_by_name(SnglInspiralTable *tab, const char *name);
float float_inspiral_column_by_name(SnglInspiralTable *tab, const char *name);
double double_inspiral_column_by_name(SnglInspiralTable *tab, const char *name);
char * string_inspiral_column_by_name(SnglInspiralTable *tab, const char *name);
int insert_from_sngl_inspiral_table(PGconn *connection, SnglInspiralTable *tab,
                                    int num_rows);
int insert_inspiral_xml(PGconn *connection, char *filename);
