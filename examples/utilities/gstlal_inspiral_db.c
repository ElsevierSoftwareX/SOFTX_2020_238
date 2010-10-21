#include <stdio.h>
#include <math.h>
#include <postgresql/libpq-fe.h>
#include "gstlal_inspiral_db.h"
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/Date.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALStdio.h>
#include <lal/CoincInspiralEllipsoid.h>
#include <lal/TimeDelay.h>
#include <lal/LIGOLwXML.h>


const int num_inspiral_columns = 58;
const char empty_str = '\0';

const char sngl_inspiral_table_columns[58][2][32] =
   {
     {{"process_id"},{"string"}},
     {{"ifo"},{"string"}},
     {{"search"},{"string"}},
     {{"channel"},{"string"}},
     {{"end_time"},{"int"}},
     {{"end_time_ns"},{"int"}},
     {{"end_time_gmst"},{"double"}},
     {{"impulse_time"},{"int"}},
     {{"impulse_time_ns"},{"int"}},
     {{"template_duration"},{"float"}},
     {{"event_duration"},{"double"}},
     {{"amplitude"},{"float"}},
     {{"eff_distance"},{"float"}},
     {{"coa_phase"},{"float"}},
     {{"mass1"},{"float"}},
     {{"mass2"},{"float"}},
     {{"mchirp"},{"float"}},
     {{"mtotal"},{"float"}},
     {{"eta"},{"float"}},
     {{"kappa"},{"float"}},
     {{"chi"},{"float"}},
     {{"tau0"},{"float"}},
     {{"tau2"},{"float"}},
     {{"tau3"},{"float"}},
     {{"tau4"},{"float"}},
     {{"tau5"},{"float"}},
     {{"ttotal"},{"float"}},
     {{"psi0"},{"float"}},
     {{"psi3"},{"float"}},
     {{"alpha"},{"float"}},
     {{"alpha1"},{"float"}},
     {{"alpha2"},{"float"}},
     {{"alpha3"},{"float"}},
     {{"alpha4"},{"float"}},
     {{"alpha5"},{"float"}},
     {{"alpha6"},{"float"}},
     {{"beta"},{"float"}},
     {{"f_final"},{"float"}},
     {{"snr"},{"float"}},
     {{"chisq"},{"float"}},
     {{"chisq_dof"},{"float"}},
     {{"bank_chisq"},{"float"}},
     {{"bank_chisq_dof"},{"float"}},
     {{"cont_chisq"},{"float"}},
     {{"cont_chisq_dof"},{"float"}},
     {{"sigmasq"},{"double"}},
     {{"rsqveto_duration"},{"float"}},
     {{"gamma0"},{"float"}},
     {{"gamma1"},{"float"}},
     {{"gamma2"},{"float"}},
     {{"gamma3"},{"float"}},
     {{"gamma4"},{"float"}},
     {{"gamma5"},{"float"}},
     {{"gamma6"},{"float"}},
     {{"gamma7"},{"float"}},
     {{"gamma8"},{"float"}},
     {{"gamma9"},{"float"}},
     {{"event_id"},{"string"}}
   };

int sngl_inspiral_table_value_to_string(SnglInspiralTable *tab, char *str, int column)
  {
  switch ( column )
    {
    /* No 0 column which is the process id*/
    case 0:
         sprintf(str, "''");
	 break;
    case 1: 
 	sprintf(str, "%s", tab->ifo);
	break;
    case 2: 
	sprintf(str, "%s", tab->search);
	break;
    case 3: 
	sprintf(str, "%s", tab->channel);
	break;
    case 4: 
	sprintf(str, "%d", tab->end_time.gpsSeconds);
	break;
    case 5: 
	sprintf(str, "%d", tab->end_time.gpsNanoSeconds);
	break;
    case 6: 
	sprintf(str, "%f", tab->end_time_gmst);
	break;
    case 7: 
	sprintf(str, "%d", tab->impulse_time.gpsSeconds);
	break;
    case 8: 
	sprintf(str, "%d", tab->impulse_time.gpsNanoSeconds);
	break;
    case 9: 
	sprintf(str, "%f", tab->event_duration);
	break;
    case 10: 
	sprintf(str, "%f", tab->template_duration);
	break;
    case 11: 
	sprintf(str, "%f", tab->amplitude);
	break;
    case 12: 
	sprintf(str, "%f", tab->eff_distance);
	break;
    case 13: 
	sprintf(str, "%f", tab->coa_phase);
	break;
    case 14: 
	sprintf(str, "%f", tab->mass1);
	break;
    case 15: 
	sprintf(str, "%f", tab->mass2);
	break;
    case 16:
        sprintf(str, "%f", tab->mchirp);
        break;
    case 17: 
	sprintf(str, "%f", tab->mtotal);
	break;
    case 18: 
	sprintf(str, "%f", tab->eta);
	break;
    case 19: 
	sprintf(str, "%f", tab->kappa);
	break;
    case 20: 
	sprintf(str, "%f", tab->chi);
	break;
    case 21: 
	sprintf(str, "%f", tab->tau0);
	break;
    case 22:
         sprintf(str, "%f", tab->tau2);
	 break;
    case 23: 
	sprintf(str, "%f", tab->tau3);
	break;
    case 24: 
	sprintf(str, "%f", tab->tau4);
	break;
    case 25: 
	sprintf(str, "%f", tab->tau5);	
	break;
    case 26: 
	sprintf(str, "%f", tab->ttotal);
	break;
    case 27: 
	sprintf(str, "%f", tab->psi0);
	break;
    case 28: 
	sprintf(str, "%f", tab->psi3);
	break;
    case 29: 
	sprintf(str, "%f", tab->alpha);
	break;
    case 30: 
	sprintf(str, "%f", tab->alpha1);
	break;
    case 31: 
	sprintf(str, "%f", tab->alpha2);
	break;
    case 32: 
	sprintf(str, "%f", tab->alpha3);
	break;
    case 33: 
	sprintf(str, "%f", tab->alpha4);
	break;
    case 34: 
	sprintf(str, "%f", tab->alpha5);	
	break;
    case 35: 
	sprintf(str, "%f", tab->alpha6);
	break;
    case 36: 
	sprintf(str, "%f", tab->beta);
	break;
    case 37: 
	sprintf(str, "%f", tab->f_final);
	break;
    case 38: 
	sprintf(str, "%f", tab->snr);
	break;
    case 39: 
	sprintf(str, "%f", tab->chisq);
	break;
    case 40: 
	sprintf(str, "%d", tab->chisq_dof);
	break;
    case 41: 
	sprintf(str, "%f", tab->bank_chisq);
	break;
    case 42: 
	sprintf(str, "%d", tab->bank_chisq_dof);
	break;
    case 43: 
	sprintf(str, "%f", tab->cont_chisq);
	break;
    case 44: 
	sprintf(str, "%d", tab->cont_chisq_dof);
	break;
    case 45: 
	sprintf(str, "%f", tab->sigmasq);
	break;
    case 46: 
	sprintf(str, "%f", tab->rsqveto_duration);
	break;
    case 47: 
	sprintf(str, "%f", tab->Gamma[0]);
	break;
    case 48: 
	sprintf(str, "%f", tab->Gamma[1]);
	break;
    case 49: 
	sprintf(str, "%f", tab->Gamma[2]);
	break;
    case 50: 
	sprintf(str, "%f", tab->Gamma[3]);
	break;
    case 51: 
	sprintf(str, "%f", tab->Gamma[4]);
	break;
    case 52: 
	sprintf(str, "%f", tab->Gamma[5]);
	break;
    case 53: 
	sprintf(str, "%f", tab->Gamma[6]);
	break;
    case 54: 
	sprintf(str, "%f", tab->Gamma[7]);
	break;
    case 55: 
	sprintf(str, "%f", tab->Gamma[8]);
	break;
    case 56: 
	sprintf(str, "%f", tab->Gamma[9]);
	break;
    case 57: 
	sprintf(str, "%d", tab->event_id->id);
	break;
    }
  }

int int_inspiral_column_by_name(SnglInspiralTable *tab, const char *name)
  {
  if (!strcmp(name,"end_time")) return tab->end_time.gpsSeconds;
  if (!strcmp(name,"end_time_ns")) return tab->end_time.gpsNanoSeconds;
  if (!strcmp(name,"impulse_time")) return tab->impulse_time.gpsSeconds;
  if (!strcmp(name,"impulse_time_ns")) return tab->impulse_time.gpsNanoSeconds;
  return 0;
  }

int set_int_inspiral_column_by_name(SnglInspiralTable *tab, const char *name,
                                    int value)
  {
  if (!strcmp(name,"end_time")) tab->end_time.gpsSeconds = value;
  if (!strcmp(name,"end_time_ns")) tab->end_time.gpsNanoSeconds = value;
  if (!strcmp(name,"impulse_time")) tab->impulse_time.gpsSeconds = value;
  if (!strcmp(name,"impulse_time_ns")) tab->impulse_time.gpsNanoSeconds = value;
  return 0;
  }


float float_inspiral_column_by_name(SnglInspiralTable *tab, const char *name)
  {
  if (!strcmp(name,"template_duration")) return tab->template_duration;
  if (!strcmp(name,"amplitude")) return tab->amplitude;
  if (!strcmp(name,"eff_distance")) return tab->eff_distance;
  if (!strcmp(name,"coa_phase")) return tab->coa_phase;
  if (!strcmp(name,"mass1")) return tab->mass1;
  if (!strcmp(name,"mass2")) return tab->mass2;
  if (!strcmp(name,"mchirp")) return tab->mchirp;
  if (!strcmp(name,"mtotal")) return tab->mtotal;
  if (!strcmp(name,"eta")) return tab->eta;
  if (!strcmp(name,"kappa")) return tab->kappa;
  if (!strcmp(name,"chi")) return tab->chi;
  if (!strcmp(name,"tau0")) return tab->tau0;
  if (!strcmp(name,"tau2")) return tab->tau2;
  if (!strcmp(name,"tau3")) return tab->tau3;
  if (!strcmp(name,"tau4")) return tab->tau4;
  if (!strcmp(name,"tau5")) return tab->tau5;
  if (!strcmp(name,"ttotal")) return tab->ttotal;
  if (!strcmp(name,"psi0")) return tab->psi0;
  if (!strcmp(name,"psi3")) return tab->psi3;
  if (!strcmp(name,"alpha")) return tab->alpha;
  if (!strcmp(name,"alpha1")) return tab->alpha1;
  if (!strcmp(name,"alpha2")) return tab->alpha2;
  if (!strcmp(name,"alpha3")) return tab->alpha3;
  if (!strcmp(name,"alpha4")) return tab->alpha4;
  if (!strcmp(name,"alpha5")) return tab->alpha5;
  if (!strcmp(name,"alpha6")) return tab->alpha6;
  if (!strcmp(name,"beta")) return tab->beta;
  if (!strcmp(name,"f_final")) return tab->f_final;
  if (!strcmp(name,"snr")) return tab->snr;
  if (!strcmp(name,"chisq")) return tab->chisq;
  if (!strcmp(name,"chisq_dof")) return tab->chisq_dof;
  if (!strcmp(name,"bank_chisq")) return tab->bank_chisq;
  if (!strcmp(name,"bank_chisq_dof")) return tab->bank_chisq_dof;
  if (!strcmp(name,"cont_chisq")) return tab->cont_chisq_dof;
  if (!strcmp(name,"rsq_veto_duration")) return tab->rsqveto_duration;
  if (!strcmp(name,"gamma0")) return tab->Gamma[0];
  if (!strcmp(name,"gamma1")) return tab->Gamma[1];
  if (!strcmp(name,"gamma2")) return tab->Gamma[2];
  if (!strcmp(name,"gamma3")) return tab->Gamma[3];
  if (!strcmp(name,"gamma4")) return tab->Gamma[4];
  if (!strcmp(name,"gamma5")) return tab->Gamma[5];
  if (!strcmp(name,"gamma6")) return tab->Gamma[6];
  if (!strcmp(name,"gamma7")) return tab->Gamma[7];
  if (!strcmp(name,"gamma8")) return tab->Gamma[8];
  if (!strcmp(name,"gamma9")) return tab->Gamma[9];
  return 0;
  }

int set_float_inspiral_column_by_name(SnglInspiralTable *tab, 
                                        const char *name,
                                        float value)
  {
  if (!strcmp(name,"template_duration")) tab->template_duration = value;
  if (!strcmp(name,"amplitude")) tab->amplitude = value;
  if (!strcmp(name,"eff_distance")) tab->eff_distance = value;
  if (!strcmp(name,"coa_phase")) tab->coa_phase = value;
  if (!strcmp(name,"mass1")) tab->mass1 = value;
  if (!strcmp(name,"mass2")) tab->mass2 = value;
  if (!strcmp(name,"mchirp")) tab->mchirp = value;
  if (!strcmp(name,"mtotal")) tab->mtotal = value;
  if (!strcmp(name,"eta")) tab->eta = value;
  if (!strcmp(name,"kappa")) tab->kappa = value;
  if (!strcmp(name,"chi")) tab->chi = value;
  if (!strcmp(name,"tau0")) tab->tau0 = value;
  if (!strcmp(name,"tau2")) tab->tau2 = value;
  if (!strcmp(name,"tau3")) tab->tau3 = value;
  if (!strcmp(name,"tau4")) tab->tau4 = value;
  if (!strcmp(name,"tau5")) tab->tau5 = value;
  if (!strcmp(name,"ttotal")) tab->ttotal = value;
  if (!strcmp(name,"psi0")) tab->psi0 = value;
  if (!strcmp(name,"psi3")) tab->psi3 = value;
  if (!strcmp(name,"alpha")) tab->alpha = value;
  if (!strcmp(name,"alpha1")) tab->alpha1 = value;
  if (!strcmp(name,"alpha2")) tab->alpha2 = value;
  if (!strcmp(name,"alpha3")) tab->alpha3 = value;
  if (!strcmp(name,"alpha4")) tab->alpha4 = value;
  if (!strcmp(name,"alpha5")) tab->alpha5 = value;
  if (!strcmp(name,"alpha6")) tab->alpha6 = value;
  if (!strcmp(name,"beta")) tab->beta = value;
  if (!strcmp(name,"f_final")) tab->f_final = value;
  if (!strcmp(name,"snr")) tab->snr = value;
  if (!strcmp(name,"chisq")) tab->chisq = value;
  if (!strcmp(name,"chisq_dof")) tab->chisq_dof = value;
  if (!strcmp(name,"bank_chisq")) tab->bank_chisq = value;
  if (!strcmp(name,"bank_chisq_dof")) tab->bank_chisq_dof = value;
  if (!strcmp(name,"cont_chisq")) tab->cont_chisq_dof = value;
  if (!strcmp(name,"rsq_veto_duration")) tab->rsqveto_duration = value;
  if (!strcmp(name,"gamma0")) tab->Gamma[0] = value;
  if (!strcmp(name,"gamma1")) tab->Gamma[1] = value;
  if (!strcmp(name,"gamma2")) tab->Gamma[2] = value;
  if (!strcmp(name,"gamma3")) tab->Gamma[3] = value;
  if (!strcmp(name,"gamma4")) tab->Gamma[4] = value;
  if (!strcmp(name,"gamma5")) tab->Gamma[5] = value;
  if (!strcmp(name,"gamma6")) tab->Gamma[6] = value;
  if (!strcmp(name,"gamma7")) tab->Gamma[7] = value;
  if (!strcmp(name,"gamma8")) tab->Gamma[8] = value;
  if (!strcmp(name,"gamma9")) tab->Gamma[9] = value;
  return 0;
  }

double double_inspiral_column_by_name(SnglInspiralTable *tab, const char *name)
  {
  if (!strcmp(name,"end_time_gmst")) return tab->end_time_gmst;
  if (!strcmp(name,"event_duration")) return tab->event_duration;
  if (!strcmp(name,"sigmasq")) return tab->impulse_time.gpsSeconds;
  if (!strcmp(name,"impulse_time_ns")) return tab->impulse_time.gpsNanoSeconds;
  return 0;
  }

int set_double_inspiral_column_by_name(SnglInspiralTable *tab, 
                                          const char *name,
                                          double value)
  {
  if (!strcmp(name,"end_time_gmst")) tab->end_time_gmst = value;
  if (!strcmp(name,"event_duration")) tab->event_duration = value;
  if (!strcmp(name,"sigmasq")) tab->impulse_time.gpsSeconds = value;
  if (!strcmp(name,"impulse_time_ns")) tab->impulse_time.gpsNanoSeconds = value;
  return 0;
  }


int set_string_inspiral_column_by_name(SnglInspiralTable *tab, 
                                       const char *name,
                                       char *value)
  {
  char *tmp;
  if (!strcmp(name,"process_id")) return 0; /* FIXME THIS ISNT IN THE LAL DEF */
  if (!strcmp(name,"ifo")) strcpy(tab->ifo,value);
  if (!strcmp(name,"search")) strcpy(tab->search,value);
  if (!strcmp(name,"channel")) strcpy(tab->channel,value);
  if (!strcmp(name,"event_id"))
    {
    if (tab->event_id) strcpy(tab->event_id->textId,value);
    if (tab->event_id) tab->event_id->id = (long long int) strtod(value,&tmp);
    }
  return 0;
  }

char * string_inspiral_column_by_name(SnglInspiralTable *tab, const char *name)
  {
  if (!strcmp(name,"process_id")) return (char *) &empty_str; /* THIS ISNT IN THE LAL DEF */
  if (!strcmp(name,"ifo")) return tab->ifo;
  if (!strcmp(name,"search")) return tab->search;
  if (!strcmp(name,"channel")) return tab->channel;
  if (!strcmp(name,"event_id"))
    {
    if (tab->event_id) return tab->event_id->textId;
    else return (char *) &empty_str;
    }
  return (char *) &empty_str;
  }

int get_sngl_inspiral_table_column_string(char *str, int insert)
  {
  int i;
  char buf[255];
  int num_col = num_inspiral_columns;
  if (insert) 
    {
    num_col -= 1;
    strcat(str, " (");
    }
  else strcat(str, " ");
  for (i=0; i < (num_col); i++)
    {
    if (i < (num_col-1))
      {
      sprintf(buf, "%s, ", sngl_inspiral_table_columns[i][0]);
      strcat(str,buf);
      }
    else
      {
      if (insert) sprintf(buf, "%s) ", sngl_inspiral_table_columns[i][0]);
      else sprintf(buf, "%s ", sngl_inspiral_table_columns[i][0]);
      strcat(str,buf);
      }
    }
  return 0;
  }

int get_sngl_inspiral_table_values_string(char *str, SnglInspiralTable *tab, int skip_event_id)
  {
  int i;
  char buf[64 * num_inspiral_columns];
  int num_col = num_inspiral_columns;
  int cnt = 0;
  if (skip_event_id) num_col -= 1;
  if (!tab) return 1; /* There is nothing to do */
  while(tab)
    {
    /*strcat(str, " (");*/
    cnt++;
    /*for (i=0; i < (num_col); i++)
      {
      sngl_inspiral_table_value_to_string(tab, buf, i);
      strcat(str, buf);*/
      
      sprintf(buf,"('%s', '%s', '%s', '%s', %d, %d, %f, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d, %f, %d, %f, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
         &empty_str,
         tab->ifo, 
	 tab->search, 
	 tab->channel, 
	 tab->end_time.gpsSeconds,
         tab->end_time.gpsNanoSeconds, 
	 tab->end_time_gmst,
         tab->impulse_time.gpsSeconds, 
	 tab->impulse_time.gpsNanoSeconds, 
         tab->template_duration, 
	 tab->event_duration, 
         tab->amplitude, 
	 tab->eff_distance, 
	 tab->coa_phase, 
	 tab->mass1,
         tab->mass2, 
	 tab->mchirp, 
	 tab->mtotal, 
	 tab->eta, 
	 tab->kappa, 
         tab->chi, 
	 tab->tau0, 
	 tab->tau2, 
	 tab->tau3, 
	 tab->tau4, 
	 tab->tau5,
         tab->ttotal, 
	 tab->psi0, 
	 tab->psi3, 
	 tab->alpha, 
	 tab->alpha1,
         tab->alpha2, 
	 tab->alpha3, 
	 tab->alpha4, 
	 tab->alpha5, 
	 tab->alpha6,
         tab->beta, 
	 tab->f_final, 
	 tab->snr, 
	 tab->chisq, 
	 tab->chisq_dof, 
         tab->bank_chisq, 
	 tab->bank_chisq_dof, 
	 tab->cont_chisq, 
         tab->cont_chisq_dof, 
	 tab->sigmasq, 
	 tab->rsqveto_duration,
	 tab->Gamma[0], 
	 tab->Gamma[1],
         tab->Gamma[2], 
	 tab->Gamma[3], 
	 tab->Gamma[4], 
	 tab->Gamma[5], 
         tab->Gamma[6], 
	 tab->Gamma[7], 
	 tab->Gamma[8], 
	 tab->Gamma[9]);
      strcat(str, buf);
      if (skip_event_id) strcat(str,") ");
      else
        {
        sprintf(buf, ", %d)",tab->event_id->id);
        strcat(str,buf);
        }
    if (tab->next) strcat(str, ", ");
    /*else strcat(str,")");*/
    tab = tab->next;
    }
  return 0;
  }

PGresult * execute_and_check_query(PGconn *con, char *command)
  {
  PGresult *result = NULL;
  ExecStatusType stat;
  result = PQexec(con, command);
  stat = PQresultStatus(result);
  if (stat == PGRES_COMMAND_OK)
    {
    /*fprintf(stderr, "QUERY succeeded - no results\n");*/
    return result;
    }
  if (stat == PGRES_TUPLES_OK)
    {
    /*fprintf(stderr, "QUERY succeeded - returning %d X %d results\n", PQnfields(result),PQntuples(result));*/
    return result;
    }
  if (stat == PGRES_FATAL_ERROR)
    {
    fprintf(stderr, "## QUERY FAILED WITH FATAL ERROR ##- is the database connection set up?  Have you done a proper tunnel? This is likely to be somthing silly like that and not a syntax error in the request.\n\n");
    fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));
    return result;
    }
  /* Otherwise ...*/
  fprintf(stderr, "\n############################################\n");
  fprintf(stderr, "### QUERY FAILED ###########################\n");
  fprintf(stderr, "############################################\n\n\t\%s\n\n", command);
  fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));
    
  return NULL;
  }

PGresult * execute_and_check_prepared_query(PGconn *con, char *command, 
               const char *name, int npar, char **parval)
  {
  PGresult *result = NULL;
  ExecStatusType stat;
  result = PQexecPrepared(con, name, npar, (const char * const *) parval, NULL, NULL, 0);


  stat = PQresultStatus(result);
  if (stat == PGRES_COMMAND_OK)
    {
    /*fprintf(stderr, "QUERY succeeded - no results\n");*/
    return result;
    }
  if (stat == PGRES_TUPLES_OK)
    {
    /*fprintf(stderr, "QUERY succeeded - returning %d X %d results\n", PQnfields(result),PQntuples(result));*/
    return result;
    }
  if (stat == PGRES_FATAL_ERROR)
    {
    fprintf(stderr, "## QUERY FAILED WITH FATAL ERROR ##- is the database connection set up?  Have you done a proper tunnel? This is likely to be somthing silly like that and not a syntax error in the request.\n\n");
    fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));
    return result;
    }
  /* Otherwise ...*/
  fprintf(stderr, "\n############################################\n");
  fprintf(stderr, "### QUERY FAILED ###########################\n");
  fprintf(stderr, "############################################\n\n\t\%s\n\n", command);
  fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));

  return NULL;
  }

PGresult *prepare_and_check_query(PGconn *con, char *name, char *query, int n)
  {
  PGresult *result = PQprepare(con, name, query, 56, NULL);
  ExecStatusType stat;
  stat = PQresultStatus(result);
  if (stat == PGRES_COMMAND_OK)
    {
    fprintf(stderr, "QUERY PREPARE succeeded - no results\n");
    return result;
    }
  if (stat == PGRES_TUPLES_OK)
    {
    fprintf(stderr, "QUERY PREPARE succeeded - returning %d X %d results\n", PQnfields(result),PQntuples(result));
    return result;
    }
  if (stat == PGRES_FATAL_ERROR)
    {
    fprintf(stderr, "## QUERY PREPARE FAILED WITH FATAL ERROR ##- is the database connection set up?  Have you done a proper tunnel? This is likely to be somthing silly like that and not a syntax error in the request.\n\n");
    fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));
    return result;
    }
  /* Otherwise ...*/
  fprintf(stderr, "\n############################################\n");
  fprintf(stderr, "### QUERY PREPARE FAILED ###################\n");
  fprintf(stderr, "############################################\n\n\t\%s\n\n", query);
  fprintf(stderr, "stat %d ERROR MESSAGE IS: \n\t%s\n\n", stat, PQresultErrorMessage(result));

  return NULL;
  }

int insert_sngl_inspiral_table(PGconn *con, SnglInspiralTable *tab)
  {
  /* 1 less column because the event id and proc id is not used */
  /* Careful strings longer than 128 will be a problem */
  char p[num_inspiral_columns -2][128];
  char * pval[num_inspiral_columns -2];
  char *query = "INSERT INTO sngl_inspiral_table VALUES ('', $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56);";
  int i;
  PGresult *result = prepare_and_check_query(con, "", query, 56);

  PQclear(result);
  while (tab)
    {
    /* start after process id */
    for (i = 1; i < num_inspiral_columns - 1; i++)
      {
      /* The i-1 is because the process_id is skipped */
      sngl_inspiral_table_value_to_string(tab, p[i-1], i);
      pval[i-1] =  p[i-1];
      }
    /*fprintf(stderr, "executing query \n");*/
    result = execute_and_check_prepared_query(con, query, "", 56, pval);
    PQclear(result);
    tab = tab->next;
    }
  }

#if 0
int insert_from_sngl_inspiral_table(PGconn *connection, SnglInspiralTable *tab,
                                    int num_rows)
  {
  /* try to make a buffer big enough to hold the insert statment*/
  char buf[255];
  char str[(num_rows+3) * num_inspiral_columns * 16];
  int i;
  PGresult *result = NULL;
  if (!tab)
    {
    fprintf(stderr,"No triggers in sngl_inspiral_table to insert :(\n\n");
    fprintf(stderr,"insert_from_sngl_inspiral_table() Failed\n\n");
    return 1;
    }

  /* BEGIN THE INSERT STATEMENT */  
  sprintf(str, "INSERT INTO sngl_inspiral_table ");
  /* The last argument skips the event_id - we want the database to assign 
   * this */
  get_sngl_inspiral_table_column_string(str, 1);
  strcat(str, " VALUES ");
  get_sngl_inspiral_table_values_string(str, tab, 1);
  /*fprintf(stderr,"\n\n%s\n\n",str);*/
  result = execute_and_check_query(connection, str);
  if (PQresultStatus(result) != PGRES_COMMAND_OK) return 1;

/* All DONE */
  return 0;
  }
#endif

int set_pgvalue_to_int(char *value)
  {
  char *endptr;
  return (int) floor(strtod(value,&endptr));
  }

float set_pgvalue_to_float(char *value)
  {
  char *endptr;
  return (float) strtod(value,&endptr);
  }

double set_pgvalue_to_double(char *value)
  {
  char *endptr;
  return (int) strtod(value,&endptr);
  }

char *set_pgvalue_to_string(char *value)
  {
  return value;
  }

int set_sngl_inspiral_table_member(SnglInspiralTable *tab, char *value,
                                   char column_def[2][32])
  {
  /* column_def is a 2x32 character array of column name and type */
  if (!strcmp(column_def[1],"int"))
    set_int_inspiral_column_by_name(tab, column_def[0],
                                    set_pgvalue_to_int(value));
  if (!strcmp(column_def[1],"float"))   
    set_float_inspiral_column_by_name(tab, column_def[0],
                                    set_pgvalue_to_float(value));
  if (!strcmp(column_def[1],"double"))   
    set_double_inspiral_column_by_name(tab, column_def[0],
                                    set_pgvalue_to_double(value));
  if (!strcmp(column_def[1],"string"))   
    set_string_inspiral_column_by_name(tab, column_def[0],
                                    set_pgvalue_to_string(value));
  return 0;
  }

SnglInspiralTable * PGresult_to_sngl_inspiral_table(PGresult *result)
  {
  int i,j;  
  int nFields = PQnfields(result);
  int nRows =  PQntuples(result);
  char columns[nFields][2][32];
  SnglInspiralTable *tab = (SnglInspiralTable *) calloc(1,sizeof(SnglInspiralTable));
  tab->event_id = (EventIDColumn *) calloc(1,sizeof(EventIDColumn));
  SnglInspiralTable *first = tab;

  /* Work out what columns are in the result to get the c types defined here */
  for (i = 0; i < nFields; i++)
    {
    for (j = 0; j < num_inspiral_columns; j++)
      {
      if ( !(strcmp(PQfname(result,i), sngl_inspiral_table_columns[j][0])) )
         {
         strcpy(columns[i][0],sngl_inspiral_table_columns[j][0]);
         strcpy(columns[i][1],sngl_inspiral_table_columns[j][1]);
	 }
      }
    }
  /* Now extract the data */

  for (i = 0; i < nRows; i++)
    {
    for (j = 0; j < nFields; j++)
      {
      set_sngl_inspiral_table_member(tab, PQgetvalue(result, i, j), 
                                     columns[j]);
      }
    /*fprintf(stderr,"Gamma0 %f\n",tab->Gamma[0]);*/
    if (i < (nRows-1)) 
      {
      tab = tab->next = (SnglInspiralTable *) calloc(1,sizeof(SnglInspiralTable));  
      tab->event_id = (EventIDColumn *) calloc(1,sizeof(EventIDColumn));
      }
    }
  fprintf(stderr, "Found %d inspiral triggers in requested range\n",nRows);
  return first;
  }

int free_sngl_inspiral_table(SnglInspiralTable* tab)
  {
  SnglInspiralTable *tmp;
  while (tab)
    {
    tmp = tab;
    tab = tab->next;
    if (tmp->event_id) free(tmp->event_id);
    free(tmp);
    }
  return 0;
  }

int free_coinc_inspiral_table(CoincInspiralTable *tab)
  {
  /* THIS FUNCTION ASSUMES THAT THE ASSOCIATED TABLES HAVE BEEN FREED*/
  CoincInspiralTable *tmp;
  int i; 
  while (tab)
    {
    tmp = tab;
    tab = tab->next;
    free(tmp);
    }
  return 0;
  }



SnglInspiralTable * select_to_sngl_inspiral_table(PGconn *connection, 
                                         char *WHERE_str)
  {
  char *str = NULL;
  int i;
  PGresult *result = NULL;
  SnglInspiralTable *tab = NULL;
  /* Try to allocate enough of a buffer for the query */
  str = calloc((10) * num_inspiral_columns * 16, sizeof(char));
  sprintf(str, "SELECT ");
  /* The false argument doesn't skip the event_id - we want that from the
   * database - it was generated automatically on insert */
  get_sngl_inspiral_table_column_string(str, 0);
  strcat(str, " FROM sngl_inspiral_table ");
  strcat(str,WHERE_str);
  strcat(str,";");
  result = execute_and_check_query(connection, str);
  tab = PGresult_to_sngl_inspiral_table(result);
  free(str);
  PQclear(result);
  return tab;
  }

SnglInspiralTable * select_to_sngl_inspiral_table_from_GPS(PGconn *connection, 
                                           double GPS_start, 
                                           double GPS_stop)
  {
  char buf[255];
  int i;
  SnglInspiralTable *tab = NULL;
  PGresult *result = NULL;
  /* Try to allocate enough of a buffer for the query */
  sprintf(buf," WHERE sngl_inspiral_table.end_time BETWEEN %d AND %d ", (int) floor(GPS_start),(int) ceil(GPS_stop));
  tab = select_to_sngl_inspiral_table(connection, buf);
  return tab;
  }


int get_time_boundaries_from_sngl_inspiral(SnglInspiralTable *tab, 
					   double boundaries[2])
  {
  double time = 0;
  if (tab == NULL)
    {
    fprintf(stderr, "Got null pointer for inspiral table get_time_boundaries_from_sngl_inspiral()\n");
    return 1;
    }
  boundaries[0] = boundaries[1] = XLALGPSGetREAL8( &(tab->end_time) );

  while (tab)
    {
    time = XLALGPSGetREAL8( &(tab->end_time) );
    if (time < boundaries[0]) boundaries[0] = time;
    if (time > boundaries[1]) boundaries[1] = time;
    tab = tab->next;
    }
  return 0;
  }

int get_number_of_ifos_from_sngl_inspiral_table(SnglInspiralTable *tab)
  {
  int ifo[6] = {0,0,0,0,0,0};
  int i, out;

  while (tab)
    {
    if (!strcmp(tab->ifo,"G1")) ifo[LAL_IFO_G1] = 1;
    if (!strcmp(tab->ifo,"H1")) ifo[LAL_IFO_H1] = 1;
    if (!strcmp(tab->ifo,"H2")) ifo[LAL_IFO_H2] = 1;
    if (!strcmp(tab->ifo,"L1")) ifo[LAL_IFO_L1] = 1;
    if (!strcmp(tab->ifo,"V1")) ifo[LAL_IFO_V1] = 1;
    if (!strcmp(tab->ifo,"T1")) ifo[LAL_IFO_T1] = 1;
    tab = tab->next;
    }
  out = 0;
  for (i=0; i<6; i++)
    {
    out += ifo[i];
    }
  return out;
  }

CoincInspiralTable * compute_coincs_from_sngl_inspiral_table(SnglInspiralTable *tab)
  {
  InspiralAccuracyList  *accuracy;
  LALStatus *stat;
  CoincInspiralTable  *coincTable = NULL;
  int N, numIFO;
  stat = (LALStatus *) calloc(1, sizeof(LALStatus));
  accuracy = (InspiralAccuracyList *) calloc(1, sizeof(InspiralAccuracyList));
  XLALPopulateAccuracyParams(accuracy);
  accuracy->exttrig=0;
  
  accuracy->iotaCutH1H2=-1.0;
  accuracy->test = ellipsoid;
  /* FIXME this is hardcoded to be very loose.  The user can cut */
  accuracy->eMatch = 2000.0;
  /* FIXME These time accuracies are hardcoded for now.  We want them to be
   * loose - the user can always cut on this later */
  accuracy->ifoAccuracy[LAL_IFO_G1].dt = 1000000000.0;
  accuracy->ifoAccuracy[LAL_IFO_H1].dt = 1000000000.0;
  accuracy->ifoAccuracy[LAL_IFO_H2].dt = 1000000000.0;
  accuracy->ifoAccuracy[LAL_IFO_L1].dt = 1000000000.0;
  accuracy->ifoAccuracy[LAL_IFO_V1].dt = 1000000000.0;
  accuracy->ifoAccuracy[LAL_IFO_T1].dt = 1000000000.0;

  /* Now actually do the coincidence test */
  LALCreateTwoIFOCoincListEllipsoid(stat, &coincTable, tab, accuracy);
  numIFO = get_number_of_ifos_from_sngl_inspiral_table(tab);
  fprintf(stderr, "computing multi ifo coincs...\n");
  for( N = 3; N <= numIFO; N++)
    LALCreateNIFOCoincList( stat, &coincTable, accuracy, N);  
  /*fprintf(stderr, "removing repeated coincs\n");
  LALRemoveRepeatedCoincs( stat, &coincTable );*/
  return coincTable;
  free(accuracy);
  free(stat);
  }

int insert_CoincInspiralRow(PGconn *connection, CoincInspiralTable *row)
  {
  char insert[256+LAL_NUM_IFO*64];
  char buf[256];
  PGresult *result = NULL;
  int i;
  int ID;
  int cnt;

  sprintf(insert, "INSERT INTO coinc_event (coinc_def_id, nevents) VALUES ('sngl_inspiral_to_sngl_inspiral', %d) RETURNING coinc_event_id", row->numIfos);
  result = execute_and_check_query(connection, insert);
  ID = set_pgvalue_to_int( PQgetvalue(result, 0, 0) );
  sprintf(insert, "INSERT INTO coinc_event_map (coinc_event_id, table_name, event_id) VALUES ");
  cnt = 0;
  for (i = 0; i < LAL_NUM_IFO; i++)
    {
    if (row->snglInspiral[i])
      {
      sprintf(buf," (%d, 'sngl_inspiral_table', %d)", ID,
               row->snglInspiral[i]->event_id->id);
      strcat(insert,buf);
      cnt++;
      }
    if (cnt == row->numIfos)
      {
      strcat(insert,";");
      break;
      }
    if (cnt >= 1 && cnt < row->numIfos) strcat(insert,",");
    }
  fprintf(stderr, "%s\n", insert);
  result = execute_and_check_query(connection, insert);
  PQclear(result);
  }

int insert_CoincInspiralTable(PGconn *connection, 
                             CoincInspiralTable *tab)
  {
  fprintf(stderr,"Inserting coinc inspiral triggers\n");
  while (tab)
    {
    insert_CoincInspiralRow(connection, tab);
    /* walk the list */
    tab = tab->next;
    }
  }

static int * set_ids_from_coinc_table(CoincInspiralTable *tab)
  {
  int *ids1 = calloc(LAL_NUM_IFO, sizeof(int));
  if (tab->snglInspiral[LAL_IFO_G1])
    ids1[LAL_IFO_G1] = tab->snglInspiral[LAL_IFO_G1]->event_id->id;
  if (tab->snglInspiral[LAL_IFO_H1])
    ids1[LAL_IFO_H1] = tab->snglInspiral[LAL_IFO_H1]->event_id->id;
  if (tab->snglInspiral[LAL_IFO_H2])
    ids1[LAL_IFO_H2] = tab->snglInspiral[LAL_IFO_H2]->event_id->id;
  if (tab->snglInspiral[LAL_IFO_L1])
    ids1[LAL_IFO_L1] = tab->snglInspiral[LAL_IFO_L1]->event_id->id;
  if (tab->snglInspiral[LAL_IFO_T1])
    ids1[LAL_IFO_T1] = tab->snglInspiral[LAL_IFO_T1]->event_id->id;
  if (tab->snglInspiral[LAL_IFO_V1])
    ids1[LAL_IFO_V1] = tab->snglInspiral[LAL_IFO_V1]->event_id->id;
  return ids1;
  }

static int ifo_to_num(char *ifo)
  {
  if (!(strcmp(ifo, "G1"))) return LAL_IFO_G1;
  if (!(strcmp(ifo, "H1"))) return LAL_IFO_H1;
  if (!(strcmp(ifo, "H2"))) return LAL_IFO_H2;
  if (!(strcmp(ifo, "L1"))) return LAL_IFO_L1;
  if (!(strcmp(ifo, "T1"))) return LAL_IFO_T1;
  if (!(strcmp(ifo, "V1"))) return LAL_IFO_V1;
  }



static int * set_ids_from_coinc_result(PGresult *res, int *nrows, int *coinc_ids, int *num_ifos)
  {
  int i,cnt;
  int nFields = PQnfields(res);
  int nRows =  PQntuples(res);
  int cid;
  int * ids2 = NULL;
  coinc_ids = NULL;
  num_ifos = NULL;
  (*nrows) = 0;
  /* if there weren't any coincidences then we are done */
  if (nRows == 0) return NULL;

  cid = set_pgvalue_to_int(PQgetvalue(res,0,3));

  for (i=0; i < nRows; i++)
    {
    /* if this is a new coincidence reset */
    if (cid != set_pgvalue_to_int(PQgetvalue(res,0,3)))
      (*nrows)++;    
    cid = set_pgvalue_to_int(PQgetvalue(res,i,3));
    }
  
  ids2 = (int *) calloc((*nrows)*LAL_NUM_IFO, sizeof(int));
  coinc_ids = (int *) calloc(*nrows, sizeof(int));
  num_ifos = (int *) calloc(*nrows, sizeof(int));
  /* reset the coinc id to check */
  cid = set_pgvalue_to_int(PQgetvalue(res,0,3));
  cnt = 0;
  for (i=0; i < nRows; i++)
    {
    coinc_ids[cnt] = set_pgvalue_to_int(PQgetvalue(res,i,3));
    num_ifos[cnt] = set_pgvalue_to_int(PQgetvalue(res,i,4));
    /* if this is a new coincidence move on*/
    if (cid != coinc_ids[cnt]) cnt++;   
    cid = set_pgvalue_to_int(PQgetvalue(res,i,3));
    ids2[cnt * (*nrows) + ifo_to_num(PQgetvalue(res,cnt,1))] =
      set_pgvalue_to_int(PQgetvalue(res,i,0)); 
    }
  return ids2;  
  }

static int check_coincs(int *ids1, int *ids2, int ncoinc)
  {
  int sum1 = 0;
  int sum2 = 0;
  int i = 0;
  for (i = 0; i < LAL_NUM_IFO; i++)
    {
    sum1 += ids1[i];
    sum2 += ids2[LAL_NUM_IFO*ncoinc +i];
    }
  if ( (sum1) && (sum2)   ) return 1; /* The new coinc doesn't look like any of the coincs - this must be a new coinc - add it*/
  if ( (sum1) && (!sum2)  ) return 0; /* The new coinc is a subset of one that already exists - do nothing*/
  if ( (!sum1) && (sum2)  ) return -1; /* The new coinc is a superset of one that already exists - update the coinc */
  if ( (!sum1) && (!sum2) ) return 0; /* The coincs are identical - nothing to be done */
  }

static int update_coinc_map(PGconn *con, int cid, int *ids, int nevents)
  {
  char update_coinc_event[512];
  int i;
  PGresult *res = NULL;
  sprintf(update_coinc_event, "UPDATE coinc_event SET nevents = %d WHERE coinc_event_id = %d;", nevents+1, cid);
  res = execute_and_check_query(con, update_coinc_event);
  PQclear(res);
  for (i = 0; i < LAL_NUM_IFO; i++)
    {
    if (ids[i])
      {
      sprintf(update_coinc_event, "INSERT INTO coinc_event_map (coinc_event_id, table_name, event_id) VALUES (%d, 'sngl_inspiral_table', %d);", cid, ids[i]);
      res = execute_and_check_query(con, update_coinc_event);
      PQclear(res);
      }
    }
  }

static int update_coinc_row(PGconn *con, PGresult *res, CoincInspiralTable *tab)
  {
  /* There are only three options.  Either we update the coinc that exists or
   * we make a new one or we do nothing.  
   * So this logic has to determine whether or not the
   * coinc in question exists (even as a subset) in the database already */
  int *ids1 = set_ids_from_coinc_table(tab);
  int ncoincs, i, j, test;
  int *coinc_ids = NULL;
  int *num_ifos = NULL;
  int *ids2 = set_ids_from_coinc_result(res,&ncoincs,coinc_ids,num_ifos);
  int *tests = NULL;
  int mintest = 0;
  /* if there weren't any coincidences in the database then just add the new 
   * coinc */
  if (ncoincs == 0) 
    {
    /*fprintf(stderr, "no coincs found\n");*/
    insert_CoincInspiralRow(con, tab);
    return 1;
    }
  /*else fprintf(stderr, "found %d coincs\n",ncoincs);*/
  tests = (int *) calloc(ncoincs, sizeof(int));
  for (i = 0; i < ncoincs; i++)
    {
    for (j = 0; j < LAL_NUM_IFO; j++)
      { 
      if (ids1[j] = ids2[i*LAL_NUM_IFO + j])
        ids1[j] = 0;
	ids2[i*LAL_NUM_IFO + j] = 0;
      }
    tests[i] = check_coincs(ids1, ids2, i);
    /*
    if (!test) continue;
    if (test == 1) 
      {
      insert_CoincInspiralRow(con, tab);
      break;
      }
    if (test == -1)
      {
      update_coinc_map(con, coinc_ids[i],ids1, num_ifos[i]);
      break;
      }
    */
    }
  /* check what the appropriate thing to do is */
  /* The one with the minimum test value is what should be done for the coinc*/
  mintest = tests[0];
  for (i = 1; i < ncoincs; i++) if (tests[i] < mintest) mintest = tests[i];

  if (mintest == 1) insert_CoincInspiralRow(con, tab);
  if (test == -1) update_coinc_map(con, coinc_ids[i],ids1, num_ifos[i]);

  if (ids1) free(ids1);
  if (ids2) free(ids2);
  if (coinc_ids) free(coinc_ids);
  if (num_ifos) free(num_ifos);
  return mintest;
  }

int update_coincs_from_coinc_inspiral_table(PGconn *con,
                                            CoincInspiralTable *tab)
  {
  CoincInspiralTable *toInsert = NULL;
  char select[2048];
  char buf[512];
  int i, cnt, id, upflag;
  PGresult *res;
  int newcoinc = 0;
  int donothing = 0;
  int updatecoinc = 0;
  int ncoincs = 0;
  while (tab)
    {
    cnt = 0;
    for (i = 0; i < LAL_NUM_IFO; i++)
      {
      if (tab->snglInspiral[i])
        {
	id = tab->snglInspiral[i]->event_id->id;
        if (cnt == 0) sprintf(select,"SELECT sngl_inspiral_table.event_id, sngl_inspiral_table.ifo, coinc_event_map.table_name, coinc_event.coinc_event_id, coinc_event.nevents, coinc_event.process_id, coinc_event.coinc_def_id, coinc_event.time_slide_id, coinc_event.nevents FROM coinc_event INNER JOIN coinc_event_map ON coinc_event_map.coinc_event_id = coinc_event.coinc_event_id JOIN sngl_inspiral_table ON CAST (sngl_inspiral_table.event_id AS text) = coinc_event_map.event_id WHERE coinc_def_id = 'sngl_inspiral_to_sngl_inspiral' AND (sngl_inspiral_table.event_id = %d ",id);
        else 
	  {
	  sprintf(buf," OR sngl_inspiral_table.event_id = %d ",id);
	  strcat(select, buf);
	  }
	cnt++;
	}
      }
    strcat(select,") ORDER BY coinc_event.coinc_event_id;");
    res = execute_and_check_query(con, select);
    upflag = update_coinc_row(con, res, tab);
    if (upflag == 1) newcoinc++;
    if (upflag == 0) donothing++;
    if (upflag == -1) updatecoinc++;
    PQclear(res);
    tab = tab->next;
    ncoincs++;
    }
  fprintf(stderr, "updated %d coincs\n",ncoincs);
  fprintf(stderr, "%d did not change %d were updated %d were new\n",donothing, updatecoinc, newcoinc);
  return 0;  
  }

int coinc_on_insert_from_sngl_inspiral_table(PGconn *connection, 
                                           SnglInspiralTable *tab, int num_rows)
  {
  double times[2];
  SnglInspiralTable *coinc_query_sngls = NULL;
  CoincInspiralTable *coincs = NULL;
  insert_sngl_inspiral_table(connection, tab);
  /* Figure out the time boundaries of what we just inserted */
  get_time_boundaries_from_sngl_inspiral(tab, times);
  fprintf(stderr,"SELECTING sngl_inspirals on time interval %f-%f \n",times[0],times[1]);
  /* Now request all of the sngl inspiral triggers in this time */
  /* Add an unphysically large pad to overlap and remove boundary effects */
  coinc_query_sngls = 
    select_to_sngl_inspiral_table_from_GPS(connection,times[0]-1.,times[1]+1.);
  /* TBD */
  /*delete_coincs_from_sngl_inspiral_table();*/
  if (coinc_query_sngls) 
    coincs = compute_coincs_from_sngl_inspiral_table(coinc_query_sngls);  
  else fprintf(stderr, "no sngl inspiral triggers found\n");

  if (coincs)
    {
    update_coincs_from_coinc_inspiral_table(connection, coincs);
    /*insert_CoincInspiralTable(connection, coincs);*/
    }
  else fprintf(stderr, "no coincident inspiral triggers found\n");

  /* CAREFUL! This function must only delete coincs in which all of the
   * sngl_inspiral_triggers are in this table.  It is possible that only one
   * of the inspiral triggers will be from this table, and we don't want to
   * delete that!! */
  /* FIXME also free the coincs !!! */
  fprintf(stderr, "freeing sngl inspiral triggers \n");
  free_sngl_inspiral_table(coinc_query_sngls);
  /* always free the sngls first, cause this will attempt to free them if they
   * exist and maybe break the list */
  fprintf(stderr, "freeing coinc inspiral triggers \n");
  free_coinc_inspiral_table(coincs);
  /* let the calling function worry about cleaning up tab */
  return 0;
  }

int coinc_on_insert_from_inspiral_xml(PGconn *connection, char *filename)
  {
  SnglInspiralTable *tab = NULL;
  fprintf(stderr,"reading triggers from file %s\n", filename);
  int num_rows = LALSnglInspiralTableFromLIGOLw( &tab, filename,-1,-1);
  fprintf(stderr,"Trying to insert %d rows to the sngl_inspiral_table\n", num_rows);
  coinc_on_insert_from_sngl_inspiral_table(connection, tab, num_rows);
  fprintf(stderr, "Freeing sngl_inspiral table\n"); 
  free_sngl_inspiral_table(tab);
  return 0;
  }

int insert_inspiral_xml(PGconn *connection, char *filename)
  {
  SnglInspiralTable *tab = NULL;
  SnglInspiralTable *tmp = NULL;
  SearchSummaryTable *summTab = NULL;
  SearchSummvarsTable *sumvars = NULL;
  fprintf(stderr,"reading triggers from file %s\n", filename);
  int num_rows = LALSnglInspiralTableFromLIGOLw( &tab, filename,-1,-1);
  
  /*FIXME I have to insert and then free the other tables */
  /*int num_rows = 
      XLALReadInspiralTriggerFile (&tab, &tmp, &summTab, &sumvars, filename);*/

  if (insert_sngl_inspiral_table(connection, tab))
    fprintf(stderr,"Inserting from %s failed\n\n",filename);
  else fprintf(stderr,"Inserted %d inspiral rows\n\n",num_rows);
  /* clean up */
  free_sngl_inspiral_table(tab);
  return 0;
  }

