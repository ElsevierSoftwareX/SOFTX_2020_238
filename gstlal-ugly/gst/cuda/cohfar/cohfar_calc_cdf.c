
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <glib.h>
#include "background_stats_utils.h"

#define MAX_FNAMES 100

static int parse_opts(int argc, char *argv[], gchar **input, gchar **output, gchar **ifos)
{
	int option_index = 0;
	struct option long_opts[] =
	{
		{"input-filename",	required_argument,	0,	'i'},
		{"output-filename",	required_argument,	0,	'o'},
		{"ifos",		required_argument,	0,	'd'},
		{0, 0, 0, 0}
	};
	int opt;
	while ((opt = getopt_long(argc, argv, "i:o:d:", long_opts, &option_index)) != -1) {
		switch (opt) {
			case 'i':
				*input = g_strdup((gchar *)optarg);
				break;
			case 'o':
				*output = g_strdup((gchar *)optarg);
				break;
			case 'd':
				*ifos = g_strdup((gchar *)optarg);
				break;
			default:
				exit(0);
		}
	}
}

int main(int argc, char *argv[])
{
	gchar **in = (gchar **)malloc(sizeof(gchar *));
	gchar **out = (gchar **)malloc(sizeof(gchar *));
	gchar **ifos = (gchar **)malloc(sizeof(gchar *));

	parse_opts(argc, argv, in, out, ifos);
	int nifo = strlen(*ifos) / IFO_LEN;
	int ncombo = pow(2, nifo) - 1 - nifo, icombo;
	
	BackgroundStats **stats_in = background_stats_create(*ifos);
	BackgroundStats **stats_out = background_stats_create(*ifos);
	gchar **in_fnames = g_strsplit(*in, ",", MAX_FNAMES);
	gchar **ifname;

	for (ifname = in_fnames; *ifname; ifname++) {
		printf("%s\n", *ifname);
		background_stats_from_xml(stats_in, ncombo, *ifname);
		printf("%s done read\n", *ifname);
		for (icombo=0; icombo<ncombo; icombo++)
			background_stats_rates_add(stats_out[icombo]->rates, stats_in[icombo]->rates);
	}

	for (icombo=0; icombo<ncombo; icombo++)
		background_stats_rates_to_pdf(stats_out[icombo]->rates, stats_out[icombo]->pdf);
	background_stats_to_xml(stats_out, ncombo, *out);
	//FIXME: free stats
	g_strfreev(in_fnames);

	g_free(*in);
	g_free(*out);
	g_free(*ifos);

	free(in);
	free(out);
	free(ifos);
}
