# This should be the place we define our own table: postcoh_inspiral
# Compare it with the tables in lalmetaio/ src/ LIGOMetadataTables.h

# This file also includes some utilities to handle the table.
# Compare it with lalinspiral/ src/ LIGOLwXMLInspiralRead.c

#define MAX_IFO_LEN 4 
#define MAX_ALLIFO_LEN 14 
#define MAX_SKYMAP_FNAME_LEN 20
typedef struct
tagPostCohSimpTable
{
  struct tagPostCohSimpTable *next;
  CHAR		is_background;
  CHAR		ifos[MAX_ALLIFO_LEN];
  CHAR		pivotal_ifo[MAX_IFO_LEN];
  INT4		tmplt_index;
  INT4		pixel_index;
  REAL4		max_snglsnr;	
  REAL4         cohsnr;
  REAL4         nullsnr;
  REAL4         aver_chisq;
  CHAR		skymap_fname[MAX_SKYMAP_FNAME_LEN];			// location of skymap
}
PostCohInspiralTable;



