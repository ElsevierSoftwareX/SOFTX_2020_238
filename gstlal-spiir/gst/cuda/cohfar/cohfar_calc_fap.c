/*
 * Copyright (C) 2015 Qi Chu <qi.chu@uwa.edu.au>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#include <getopt.h>
#include <math.h>
#include <string.h>
#include <glib.h>
#include <cohfar/background_stats_utils.h>

#define __DEBUG__ 1

static void parse_opts(int argc, char *argv[], gchar **pin, gchar **pfmt, gchar **pout, gchar **pifos, gchar **pwalltime)
{
	int option_index = 0;
	struct option long_opts[] =
	{
		{"input",		required_argument,	0,	'i'},
		{"input-format",	required_argument,	0,	'f'},
		{"output",		required_argument,	0,	'o'},
		{"walltime",		optional_argument,	0,	'u'},
		{"ifos",		required_argument,	0,	'd'},
		{0, 0, 0, 0}
	};
	int opt;
	while ((opt = getopt_long(argc, argv, "i:f:o:u:d:", long_opts, &option_index)) != -1) {
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
			case 'u':
				*pwalltime = g_strdup((gchar *)optarg);
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
			nevent++;
		}
		fclose(fp);
	}
	#ifdef __DEBUG__
	printf("read %d events from file\n", nevent);
	#endif
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
			gsl_vector_set(data_dim1, ievent, idata);
			token = strtok_r(NULL, " ", &savePtr);
			#ifdef __DEBUG__
			if (ievent == 0)
				printf("data chisq%d, %s\n", ievent, token);
			#endif
			sscanf(token, "%lg", &idata);
			gsl_vector_set(data_dim2, ievent, idata);
			ievent++;
		}
		fclose(fp);
	}
	gsl_vector_free(*pdata_dim1);
	gsl_vector_free(*pdata_dim2);
}	

void cohfar_get_stats_from_file(gchar **in_fnames, TriggerStatsXML *stats_in, TriggerStatsXML *stats_out, int *hist_trials)
{
	gchar **ifname;
	int icombo;
	for (ifname = in_fnames; *ifname; ifname++) {
		printf("%s\n", *ifname);
		trigger_stats_xml_from_xml(stats_in, hist_trials, *ifname);
		for (icombo=0; icombo<stats_in->ncombo; icombo++){
			trigger_stats_feature_rates_add(stats_out->multistats[icombo]->feature, stats_in->multistats[icombo]->feature, stats_out->multistats[icombo]);
			trigger_stats_livetime_add(stats_out->multistats, stats_in->multistats, icombo);
		}
	}
}


int main(int argc, char *argv[])
{
	gchar **pin = (gchar **)malloc(sizeof(gchar *));
	gchar **pfmt = (gchar **)malloc(sizeof(gchar *));
	gchar **pout = (gchar **)malloc(sizeof(gchar *));
	gchar **pifos = (gchar **)malloc(sizeof(gchar *));
	gchar **pwalltime = (gchar **)malloc(sizeof(gchar *));

	parse_opts(argc, argv, pin, pfmt, pout, pifos, pwalltime);
	int nifo = strlen(*pifos) / IFO_LEN;
	int icombo, ncombo = get_ncombo(nifo), hist_trials;
		
	TriggerStatsXML *zlstats_in = trigger_stats_xml_create(*pifos, STATS_XML_TYPE_ZEROLAG);
	TriggerStatsXML *zlstats_out = trigger_stats_xml_create(*pifos, STATS_XML_TYPE_ZEROLAG);

	TriggerStatsXML *bgstats_in = trigger_stats_xml_create(*pifos, STATS_XML_TYPE_BACKGROUND);
	TriggerStatsXML *bgstats_out = trigger_stats_xml_create(*pifos, STATS_XML_TYPE_BACKGROUND);
	gchar **in_fnames = g_strsplit(*pin, ",", -1); // split the string completely

	if (g_strcmp0(*pfmt, "data") == 0) {
		gsl_vector *data_dim1 = NULL, *data_dim2 = NULL;
		cohfar_get_data_from_file(in_fnames, &data_dim1, &data_dim2);
		// FIXME: hardcoded to only update the last stats
		trigger_stats_feature_rates_update_all(data_dim1, data_dim2, bgstats_out->multistats[ncombo-1]->feature, bgstats_out->multistats[ncombo-1]);
		trigger_stats_feature_rates_to_pdf(bgstats_out->multistats[ncombo-1]->feature);
		trigger_stats_feature_to_rank(bgstats_out->multistats[ncombo-1]->feature, bgstats_out->multistats[ncombo-1]->rank);
		if (data_dim1) {
			free(data_dim1);
			free(data_dim2);
		}
		
		// trigger_stats_pdf_from_data(data_dim1, data_dim2, stats_out[ncombo-1]->rates->lgsnr_bins, stats_out[ncombo-1]->rates->lgchisq_bins, stats_out[ncombo-1]->pdf);
	} else if(g_strcmp0(*pfmt, "stats") == 0) {
		cohfar_get_stats_from_file(in_fnames, bgstats_in, bgstats_out, &hist_trials);
		cohfar_get_stats_from_file(in_fnames, zlstats_in, zlstats_out, &hist_trials);
		for (icombo=0; icombo<ncombo; icombo++) {
			trigger_stats_feature_rates_to_pdf(zlstats_out->multistats[icombo]->feature);
			trigger_stats_feature_to_rank(zlstats_out->multistats[icombo]->feature, zlstats_out->multistats[icombo]->rank);
			/* livetime calculated from all walltimes in the input files, will deprecate the following */
			//stats_out[icombo]->livetime = atol(*pwalltime);
			printf("zlstats_out livetime %d\n", zlstats_out->multistats[icombo]->livetime );
			trigger_stats_feature_rates_to_pdf(bgstats_out->multistats[icombo]->feature);
			trigger_stats_feature_to_rank(bgstats_out->multistats[icombo]->feature, bgstats_out->multistats[icombo]->rank);
			/* livetime calculated from all walltimes in the input files, will deprecate the following */
			//stats_out[icombo]->livetime = atol(*pwalltime);
			printf("bgstats_out livetime %d\n", bgstats_out->multistats[icombo]->livetime );
		}
	}
    xmlTextWriterPtr stats_writer = NULL;
	trigger_stats_xml_dump(bgstats_out, hist_trials, *pout, STATS_XML_WRITE_START, &stats_writer);
	trigger_stats_xml_dump(zlstats_out, hist_trials, *pout, STATS_XML_WRITE_END, &stats_writer);
	
	trigger_stats_xml_destroy(bgstats_in);
	trigger_stats_xml_destroy(bgstats_out);
	trigger_stats_xml_destroy(zlstats_in);
	trigger_stats_xml_destroy(zlstats_out);

	g_strfreev(in_fnames);

	g_free(*pin);
	g_free(*pfmt);
	g_free(*pout);
	g_free(*pifos);

	free(pin);
	free(pfmt);
	free(pout);
	free(pifos);
	pin = NULL;
	pfmt = NULL;
	pout = NULL;
	pifos = NULL;
	return 0;
}
