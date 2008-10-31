#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include <sys/time.h>


/* A program to test the connection to the inspiral database */
int main()
  {
  /* Start a connection to the postgres database.  This is a local connection 
   * that connects through ssl to port 3333.  However port 3333 must be mapped
   * via an ssh tunnel to another machine that hosts an inspiral database
   */
  PGconn *connection = PQconnectdb("port=3333 dbname=inspiral user=postgres host=localhost");
  /* The combination of these two function calls makes a nonblocking connection
   * to the database.  See this excerpt from 
   * http://www.postgresql.org/docs/8.2/static/libpq-connect.html
   *
   * These two functions are used to open a connection to a database server 
   * such that your application's thread of execution is not blocked on remote 
   * I/O whilst doing so. The point of this approach is that the waits for I/O 
   * to complete can occur in the application's main loop, rather than down 
   * inside PQconnectdb, and so the application can manage this operation in 
   * parallel with other activities. 
   */
  int status = PQstatus(connection);
  fprintf(stderr,"status = %d connnection = %p\n",status,connection);

  /* Set up structures used for the system select() command to see that the 
   * connection is ready */
  int socket;
  fd_set fdset;
  int exit_flag = 1;
  FD_ZERO(&fdset);

  PostgresPollingStatusType poll;
  if (status == CONNECTION_BAD) 
    {
    fprintf(stderr, "BAD Connection to inspiral database\n");
    return status;
    }
  socket = PQsocket(connection);
  FD_SET(socket, &fdset);
  /* Wait for the connection to be made successfully */

  while(exit_flag)
    {
    switch(PQconnectPoll(connection))
      {
      case PGRES_POLLING_READING:
        fprintf(stderr, "Polling reading of inspiral database\n");
        select(socket,&fdset,NULL,NULL,NULL); /* blocks until a change in reading/writing */
	break;
      case PGRES_POLLING_WRITING:
        fprintf(stderr, "Polling wrting of inspiral database\n");
        select(socket,&fdset,NULL,NULL,NULL); /* blocks until a change in reading/writing */
	break;
      case PGRES_POLLING_FAILED:
        fprintf(stderr, "Connection failed to inspiral database\n");
        status = 1;
        exit_flag = 0;
        break;
      case PGRES_POLLING_OK:
        fprintf(stderr, "Connection succeeded to inspiral database\n");
        status = 0;
        exit_flag = 0;
        break;
      }
    }
  return status;
  }

