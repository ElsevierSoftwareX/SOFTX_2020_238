#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>
#include "gstlal_connect_db.h"
#include "gstlal_inspiral_db.h"

int main()
  {
  int i;
  PGconn *connection = connect_to_postgres_database("port=3333 dbname=inspiral user=postgres host=localhost");
  SnglInspiralTable *tab;
  tab = calloc(1,sizeof(SnglInspiralTable));
  
  insert_from_sngl_inspiral_table(connection, tab, 1);

  free(tab);
  return 0;
  }

  
