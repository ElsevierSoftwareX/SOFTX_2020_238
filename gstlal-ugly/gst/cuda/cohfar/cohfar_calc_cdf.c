
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <glib.h>
#include "background_stats_utils.h"

#define MAX_FNAMES 100
#define __DEBUG__ 1

static void parse_opts(int argc, char *argv[], gchar **pin, gchar **pfmt, gchar **pout, gchar **pifos)
{
	int option_index = 0;
	struct option long_opts[] =
	{
		{"input-filename",	required_argument,	0,	'i'},
		{"input-format",	required_argument,	0,	'f'},
		{"output-filename",	required_argument,	0,	'o'},
		{"ifos",		required_argument,	0,	'd'},
		{0, 0, 0, 0}
	};
	int opt;
	while ((opt = getopt_long(argc, argv, "i:f:o:d:", long_opts, &option_index)) != -1) {
		switch (opt) {
			case 'i':
				*pin = g_strdup((gchar *)optarg);
				break;
			case 'f':
				*pfmt = g_strdup((gchar *)optarg);
				break;
			case 'o':
				*pout = g_strdup((gchar *)optarg);
				break;
			case 'd':
				*pifos = g_strdup((gchar *)optarg);
				break;
			default:
				exit(0);
		}
	}
}

void cohfar_get_data_from_file(gchar **in_fnames, gsl_vector **pdata_dim1, gsl_vector **pdata_dim2)
{

	gchar **ifname;
	int nevent = 0;
	char buf[100];
	FILE *fp;
	/* count number of events */
	for (ifname = in_fnames; *ifname; ifname++) {
		if ((fp = fopen(*ifname, "r")) == NULL) { /* open file of data in dimension2*/
			printf("read file error dimen1\n");
		}
		while (fgets(buf, 100, fp) != NULL) { // count number of lines in file.
			nevent++;								//each file's number of lines must be the same
		}
		fclose(fp);
	}
	//data_dim1 and data_dim2 contain the data of two dimensions
	*pdata_dim1 = gsl_vector_alloc(nevent);
	*pdata_dim2 = gsl_vector_alloc(nevent);
	gsl_vector *data_dim1 = *pdata_dim1;
	gsl_vector *data_dim2 = *pdata_dim2;
	int ievent = 0;
	char *token, *savePtr;
	double idata;
	for (ifname = in_fnames; *ifname; ifname++) {
		if ((fp = fopen(*ifname, "r")) == NULL) { /* open file of data in dimension2*/
			printf("read file error dimen1\n");
		}
		while (fgets(buf, 100, fp) != NULL) {
			token = strtok_r(buf, " ", &savePtr);
			#ifdef __DEBUG__
			if (ievent == 0)
				printf("data snr%d, %s\n", ievent, token);
			#endif
			sscanf(token, "%lg", &idata);
			gsl_vector_set(data_dim1, ievent, log10(idata));
			token = strtok_r(NULL, " ", &savePtr);
			#ifdef __DEBUG__
			if (ievent == 0)
				printf("data chisq%d, %s\n", ievent, token);
			#endif
			sscanf(token, "%lg", &idata);
			gsl_vector_set(data_dim2, ievent, log10(idata));
			ievent++;
		}
		fclose(fp);
	}
}	

void cohfar_get_stats_from_file(gchar **in_fnames, BackgroundStats **stats_in, BackgroundStats **stats_out, int ncombo)
{
	gchar **ifname;
	int icombo;
	for (ifname = in_fnames; *ifname; ifname++) {
		printf("%s\n", *ifname);
		background_stats_from_xml(stats_in, ncombo, *ifname);
		printf("%s done read\n", *ifname);
		for (icombo=0; icombo<ncombo; icombo++)
			background_stats_rates_add(stats_out[icombo]->rates, stats_in[icombo]->rates);
	}
}


int main(int argc, char *argv[])
{
	gchar **pin = (gchar **)malloc(sizeof(gchar *));
	gchar **pfmt = (gchar **)malloc(sizeof(gchar *));
	gchar **pout = (gchar **)malloc(sizeof(gchar *));
	gchar **pifos = (gchar **)malloc(sizeof(gchar *));

	parse_opts(argc, argv, pin, pfmt, pout, pifos);
	int nifo = strlen(*pifos) / IFO_LEN;
	int ncombo = pow(2, nifo) - 1 - nifo, icombo;
	
	BackgroundStats **stats_in = background_stats_create(*pifos);
	BackgroundStats **stats_out = background_stats_create(*pifos);
	gchar **in_fnames = g_strsplit(*pin, ",", MAX_FNAMES);

	gsl_vector *data_dim1, *data_dim2;
	if (g_strcmp0(*pfmt, "data") == 0) {
		cohfar_get_data_from_file(in_fnames, &data_dim1, &data_dim2);
		// FIXME: hardcoded to only update the last stats
		background_stats_pdf_from_data(data_dim1, data_dim2, stats_out[ncombo-1]->rates->logsnr_bins, stats_out[ncombo-1]->rates->logchisq_bins, stats_out[ncombo-1]->pdf);
	} else if(g_strcmp0(*pfmt, "stats") == 0) {
		cohfar_get_stats_from_file(in_fnames, stats_in, stats_out, ncombo);
		for (icombo=0; icombo<ncombo; icombo++) {
			background_stats_rates_to_pdf(stats_out[icombo]->rates, stats_out[icombo]->pdf);
			background_stats_pdf_to_cdf(stats_out[icombo]->pdf, stats_out[icombo]->cdf);
		}
	}
	background_stats_to_xml(stats_out, ncombo, *pout);
	//FIXME: free stats
	g_strfreev(in_fnames);

	g_free(*pin);
	g_free(*pfmt);
	g_free(*pout);
	g_free(*pifos);

	free(pin);
	free(pfmt);
	free(pout);
	free(pifos);
	return 0;
}
