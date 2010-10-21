#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>
#include "gstlal_connect_db.h"
#include "gstlal_inspiral_db.h"
#include <lal/LALError.h>
int main()
  {
  int i;
  PGconn *connection = connect_to_postgres_database("port=3333 dbname=inspiral user=postgres host=localhost");

  coinc_on_insert_from_inspiral_xml(connection, "H1-INSPIRAL_FIRST_PLAYGROUND-815751380-2048.xml");
  coinc_on_insert_from_inspiral_xml(connection, "H2-INSPIRAL_FIRST_PLAYGROUND-815751084-2048.xml");

  return 0;
  }

  
