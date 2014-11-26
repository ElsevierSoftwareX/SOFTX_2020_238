# This should be the place we define our own table: postcoh_inspiral
# Compare it with the tables in lalmetaio/ src/ LIGOMetadataTables.h

# This file also includes some utilities to handle the table.
# Compare it with lalinspiral/ src/ LIGOLwXMLInspiralRead.c

typedef struct
tagCoincInspiralTable
{
  struct tagCoincInspiralTable *next;
  CHAR                ifos[LIGOMETA_IFOS_MAX];
  INT4                numIfos;
  SnglInspiralTable  *snglInspiral[LAL_NUM_IFO];
  SimInspiralTable   *simInspiral;
  REAL4              stat;
}
CoincInspiralTable;


typedef struct
tagSnglInspiralTable
{
  struct tagSnglInspiralTable *next;
  CHAR          ifo[LIGOMETA_IFO_MAX];
  CHAR          search[LIGOMETA_SEARCH_MAX];
  CHAR          channel[LIGOMETA_CHANNEL_MAX];
  LIGOTimeGPS   end_time;
  REAL8         end_time_gmst;
  LIGOTimeGPS   impulse_time;
  REAL8         template_duration;
  REAL8         event_duration;
  REAL4         amplitude;
  REAL4         eff_distance;
  REAL4         coa_phase;
  REAL4         mass1;
  REAL4         mass2;
  REAL4         mchirp;
  REAL4         mtotal;
  REAL4         eta;
  REAL4         kappa;
  REAL4         chi;
  REAL4         tau0;
  REAL4         tau2;
  REAL4         tau3;
  REAL4         tau4;
  REAL4         tau5;
  REAL4         ttotal;
  REAL4         psi0;
  REAL4         psi3;
  REAL4         alpha;
  REAL4         alpha1;
  REAL4         alpha2;
  REAL4         alpha3;
  REAL4         alpha4;
  REAL4         alpha5;
  REAL4         alpha6;
  REAL4         beta;
  REAL4         f_final;
  REAL4         snr;
  REAL4         chisq;
  INT4          chisq_dof;
  REAL4         bank_chisq;
  INT4          bank_chisq_dof;
  REAL4         cont_chisq;
  INT4          cont_chisq_dof;
  REAL8         sigmasq;
  REAL4         rsqveto_duration;
  REAL4         Gamma[10];    /* metric co-efficients */
  REAL4         spin1x;
  REAL4         spin1y;
  REAL4         spin1z;
  REAL4         spin2x;
  REAL4         spin2y;
  REAL4         spin2z;
  EventIDColumn *event_id;
}
SnglInspiralTable;


