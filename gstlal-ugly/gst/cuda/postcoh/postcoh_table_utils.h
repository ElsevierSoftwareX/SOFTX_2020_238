# This should be the place we define our own table: postcoh_inspiral
# Compare it with the tables in lalmetaio/ src/ LIGOMetadataTables.h

# This file also includes some utilities to handle the table.
# Compare it with lalinspiral/ src/ LIGOLwXMLInspiralRead.c

typedef struct
tagPostCohInspiralTable
{
  struct tagPostCohInspiralTable *next;
  CHAR          ifos[LIGOMETA_IFOS_MAX];
  CHAR          search[LIGOMETA_SEARCH_MAX];
  LIGOTimeGPS   end_time;
  REAL8         end_time_gmst;
  LIGOTimeGPS   impulse_time;
  REAL4         amplitude;
  REAL4         distance;
  REAL4         eff_dist_h1;
  REAL4         eff_dist_h2;
  REAL4         eff_dist_l;
  REAL4         eff_dist_g;
  REAL4         eff_dist_t;
  REAL4         eff_dist_v;
  REAL4         eff_dist_h1h2;
  REAL4         coa_phase;
  REAL4         mass1;
  REAL4         mass2;
  REAL4         mchirp;
  REAL4         eta;
  REAL4         chi;
  REAL4         kappa;
  REAL4         tau0;
  REAL4         tau2;
  REAL4         tau3;
  REAL4         tau4;
  REAL4         tau5;
  REAL4         ttotal;
  REAL4         snr; 			// max coherent snr
  INT4          snr_dof; 		// coherent snr dof
  REAL4         chisq;			// coherent chisq 
  INT4          chisq_dof;		// coherent chisq dof
  REAL4         bank_chisq;
  INT4          bank_chisq_dof;
  REAL4         cont_chisq;
  INT4          cont_chisq_dof;
  REAL4         trace_snr;
  REAL4         snr_h1;
  REAL4         snr_h2;
  REAL4         snr_l;
  REAL4         snr_g;
  REAL4         snr_t;
  REAL4         snr_v;
  REAL4         amp_term_1;
  REAL4         amp_term_2;
  REAL4         amp_term_3;
  REAL4         amp_term_4;
  REAL4         amp_term_5;
  REAL4         amp_term_6;
  REAL4         amp_term_7;
  REAL4         amp_term_8;
  REAL4         amp_term_9;
  REAL4         amp_term_10;
  REAL8         sigmasq_h1;
  REAL8         sigmasq_h2;
  REAL8         sigmasq_l;
  REAL8         sigmasq_g;
  REAL8         sigmasq_t;
  REAL8         sigmasq_v;
  REAL4         chisq_h1;		// chisq for each detector
  REAL4         chisq_h2;
  REAL4         chisq_l;
  REAL4         chisq_g;
  REAL4         chisq_t;
  REAL4         chisq_v;
  INT4          sngl_chisq_dof;
  REAL4         bank_chisq_h1;
  REAL4         bank_chisq_h2;
  REAL4         bank_chisq_l;
  REAL4         bank_chisq_g;
  REAL4         bank_chisq_t;
  REAL4         bank_chisq_v;
  INT4          sngl_bank_chisq_dof;
  REAL4         cont_chisq_h1;
  REAL4         cont_chisq_h2;
  REAL4         cont_chisq_l;
  REAL4         cont_chisq_g;
  REAL4         cont_chisq_t;
  REAL4         cont_chisq_v;
  INT4          sngl_cont_chisq_dof;
  REAL4         ra;
  REAL4         dec;
  REAL4         ligo_angle;
  REAL4         ligo_angle_sig;
  REAL4         inclination;
  REAL4         polarization;
  REAL4         null_statistic;
  REAL4         null_stat_h1h2;
  REAL4         null_stat_degen;
  EventIDColumn *event_id;
  COMPLEX8      h1quad;
  COMPLEX8      h2quad;
  COMPLEX8      l1quad;
  COMPLEX8      g1quad;
  COMPLEX8      t1quad;
  COMPLEX8      v1quad;
  REAL4         coh_snr_h1h2;
  REAL4         cohSnrSqLocal;
  REAL4         autoCorrCohSq;
  REAL4         crossCorrCohSq;
  REAL4         autoCorrNullSq;
  REAL4         crossCorrNullSq;
  REAL8         ampMetricEigenVal1;
  REAL8         ampMetricEigenVal2;
  EventIDColumn *time_slide_id;
  Char		*skymap_name;			// location of skymap
}
PostCohInspiralTable;



