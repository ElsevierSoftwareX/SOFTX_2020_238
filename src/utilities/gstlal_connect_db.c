#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>
#include "gstlal_connect_db.h"

PGconn * connect_to_postgres_database(char *connection_string)
  {
  PGconn *connection = PQconnectdb(connection_string);
  int status = PQstatus(connection);
  PostgresPollingStatusType poll;
  int socket;
  fd_set fdset;
  int exit_flag = 1;

  fprintf(stderr,"\n\n##########################################\n");
  fprintf(stderr,"status = %d connnection = %p\n",status,connection);

  /* Set up structures used for the system select() command to see that the 
 *   *  connection is ready */
  FD_ZERO(&fdset);

  if (status == CONNECTION_BAD)
    {
    fprintf(stderr, "BAD Connection to inspiral database\n");
    fprintf(stderr,"##########################################\n\n");
    return NULL;
    }
  socket = PQsocket(connection);
  FD_SET(socket, &fdset);

  /* Wait for the connection to be made successfully */

  while(exit_flag)
    {
    switch(PQconnectPoll(connection))
      {
      case PGRES_POLLING_READING:
        fprintf(stderr, "Polling reading of inspiral database\n\n");
        select(socket,&fdset,NULL,NULL,NULL); /* blocks until a change in reading/writing */
        break;
      case PGRES_POLLING_WRITING:
        fprintf(stderr, "Polling wrting of inspiral database\n\n");
        select(socket,&fdset,NULL,NULL,NULL); /* blocks until a change in reading/writing */
        break;
      case PGRES_POLLING_FAILED:
        fprintf(stderr, "Connection failed to inspiral database\n");
	fprintf(stderr,"##########################################\n\n");
        status = 1;
        exit_flag = 0;
        break;
      case PGRES_POLLING_OK:
        fprintf(stderr, "Connection succeeded to inspiral database\n");
	fprintf(stderr,"##########################################\n\n");
        status = 0;
        exit_flag = 0;
        break;
      }
    }
  if (status) return NULL;
  else return connection;
  }

