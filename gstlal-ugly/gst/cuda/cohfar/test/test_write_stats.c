
#include "../background_stats_utils.h"
#include "../../LIGOLw_xmllib/LIGOLwHeader.h"

int main(int argc, char *argv[])
{

	char *ifos = "H1L1";
	char *output_fname = "test_stats.xml.gz";
	BackgroundStats ** stats = background_stats_create(ifos);

	gsl_matrix_set_all(stats[0]->cdf->data, 0.1);
	background_stats_to_xml(stats, 1, output_fname);
	return 0;	
}
