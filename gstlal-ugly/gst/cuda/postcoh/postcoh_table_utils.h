/*
# This should be the place we define our own table: postcoh_inspiral
# Compare it with the tables in lalmetaio/ src/ LIGOMetadataTables.h

# This file also includes some utilities to handle the table.
# Compare it with lalinspiral/ src/ LIGOLwXMLInspiralRead.c
*/
#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <lal/LALStdlib.h> // for the datatypes
#include <lal/Date.h> // for the LIGOTimeGPS

#define MAX_IFO_LEN 4 
#define MAX_ALLIFO_LEN 14 
#define MAX_SKYMAP_FNAME_LEN 50
typedef struct
tagPostcohTable
{
  struct tagPostcohTable *next;
  LIGOTimeGPS	end_time;
  CHAR		is_background;
  CHAR		ifos[MAX_ALLIFO_LEN];
  CHAR		pivotal_ifo[MAX_IFO_LEN];
  INT4		tmplt_idx;
  INT4		pix_idx;
  REAL4		maxsnglsnr;	
  REAL4         cohsnr;
  REAL4         nullsnr;
  REAL4         chisq;
  CHAR		skymap_fname[MAX_SKYMAP_FNAME_LEN];			// location of skymap
}
PostcohTable;

void postcoh_table_init(XmlTable *table);

