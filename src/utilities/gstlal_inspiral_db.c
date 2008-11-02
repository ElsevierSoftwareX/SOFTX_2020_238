#include <stdio.h>
#include <postgresql/libpq-fe.h>
#include "gstlal_inspiral_db.h"
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>

const int num_inspiral_columns = 56;
const char empty_str = '\0';

const char sngl_inspiral_table_columns[56][2][32] =
   {
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
     {{"Gamma0"},{"float"}},
     {{"Gamma1"},{"float"}},
     {{"Gamma2"},{"float"}},
     {{"Gamma3"},{"float"}},
     {{"Gamma4"},{"float"}},
     {{"Gamma5"},{"float"}},
     {{"Gamma6"},{"float"}},
     {{"Gamma7"},{"float"}},
     {{"Gamma8"},{"float"}},
     {{"Gamma9"},{"float"}},
     {{"event_id"},{"string"}}
   };


int int_inspiral_column_by_name(SnglInspiralTable *tab, const char *name)
  {
  if (!strcmp(name,"end_time")) return tab->end_time.gpsSeconds;
  if (!strcmp(name,"end_time_ns")) return tab->end_time.gpsNanoSeconds;
  if (!strcmp(name,"impulse_time")) return tab->impulse_time.gpsSeconds;
  if (!strcmp(name,"impulse_time_ns")) return tab->impulse_time.gpsNanoSeconds;
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
  if (!strcmp(name,"Gamma0")) return tab->Gamma[0];
  if (!strcmp(name,"Gamma1")) return tab->Gamma[1];
  if (!strcmp(name,"Gamma2")) return tab->Gamma[2];
  if (!strcmp(name,"Gamma3")) return tab->Gamma[3];
  if (!strcmp(name,"Gamma4")) return tab->Gamma[4];
  if (!strcmp(name,"Gamma5")) return tab->Gamma[5];
  if (!strcmp(name,"Gamma6")) return tab->Gamma[6];
  if (!strcmp(name,"Gamma7")) return tab->Gamma[7];
  if (!strcmp(name,"Gamma8")) return tab->Gamma[8];
  if (!strcmp(name,"Gamma9")) return tab->Gamma[9];
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

int get_sngl_inspiral_table_column_string(char *str)
  {
  int i;
  char buf[255];
  strcat(str, " (");
  for (i=0; i < (num_inspiral_columns); i++)
    {
    if (i != (num_inspiral_columns-1))
      {
      sprintf(buf, "%s, ", sngl_inspiral_table_columns[i][0]);
      strcat(str,buf);
      }
    else
      {
      sprintf(buf, "%s) ", sngl_inspiral_table_columns[i][0]);
      strcat(str,buf);
      }
    }
  return 0;
  }

int get_sngl_inspiral_table_values_string(char *str, SnglInspiralTable *tab)
  {
  int i;
  char buf[255];
  strcat(str, " (");
  while(tab)
    {
    for (i=0; i < (num_inspiral_columns); i++)
      {
      if (!(strcmp(sngl_inspiral_table_columns[i][1],"int")))
        {
        sprintf(buf,"%d",
          int_inspiral_column_by_name(tab,sngl_inspiral_table_columns[i][0]));
        strcat(str,buf);
        }
      if (!(strcmp(sngl_inspiral_table_columns[i][1],"float")))
        {
        sprintf(buf,"%f",
         float_inspiral_column_by_name(tab,sngl_inspiral_table_columns[i][0]));
        strcat(str,buf);
        }
      if (!(strcmp(sngl_inspiral_table_columns[i][1],"double")))
        {
        sprintf(buf,"%f",
         double_inspiral_column_by_name(tab,sngl_inspiral_table_columns[i][0]));
        strcat(str,buf);
        }
      if (!(strcmp(sngl_inspiral_table_columns[i][1],"string")))
        {
        sprintf(buf,"'%s'",
         string_inspiral_column_by_name(tab,sngl_inspiral_table_columns[i][0]));
        strcat(str,buf);
        }
      if (i != (num_inspiral_columns-1)) strcat(str,", ");
      else strcat(str, " ");
      }
    if (tab->next) strcat(str, "), ");
    else strcat(str,")");
    tab = tab->next;
    }
  return 0;
  }


int insert_from_sngl_inspiral_table(PGconn *connection, SnglInspiralTable *tab,
                                    int num_rows)
  {
  /* try to make a buffer big enough to hold the insert statment*/
  char buf[255];
  char *str = NULL;
  int i;
  PGresult *result = NULL;
  str = calloc((num_rows+3) * num_inspiral_columns * 16, sizeof(char));
  /* BEGIN THE INSERT STATEMENT */
  sprintf(str, "INSERT INTO sngl_inspiral_table ");
  get_sngl_inspiral_table_column_string(str);
  strcat(str, " VALUES ");
  get_sngl_inspiral_table_values_string(str, tab);

  fprintf(stderr,"\n\n%s\n\n",str);
  result = PQexec(connection, str);
  /* All DONE */
  free(str);
  return 0;
  }


int insert_inspiral_xml(PGconn *connection, char *filename)
  {
  SnglInspiralTable *tab = NULL;
  SnglInspiralTable *tmp = NULL;
  int num_rows = LALSnglInspiralTableFromLIGOLw( &tab, filename,-1,-1);
  insert_from_sngl_inspiral_table(connection, tab, num_rows);
  /* clean up */
  while (tab)
    {
    tmp = tab;
    tab = tab->next;
    free(tmp);
    }
  return 0;
  }

