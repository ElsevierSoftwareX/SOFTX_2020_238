#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>


PGconn * connect_to_postgres_database(char *connection_string);

