#include "../detresponse_skymap.c"
#include <math.h>
//#include <stdlib.h>
//
int main()
{
	char ** ifos, str1[]="L1", str2[]="H1", str3[]="V1";
	int nifo = 3;


	ifos = (char**)malloc(nifo*sizeof(char*));
	double *horizons = (double*)malloc(nifo*sizeof(double));
	int i;
	for(i=0;i<nifo;i++)
		ifos[i] = (char*)malloc(sizeof(char)*4);
	strcpy(ifos[0], str1);
	strcpy(ifos[1], str2);
	strcpy(ifos[2], str3);

	horizons[0] = 200;
	horizons[1] = 200;
	horizons[2] = 180;

	for(i=0;i<nifo;i++)
		printf("ifo%d %s\n", i, ifos[i]);
	DetSkymap *det_map = create_detresponse_skymap(ifos, nifo, horizons, 1800 ,4);

	to_xml(det_map, "L1H1V1_skymap.xml", "L1H1V1_postcoh_skymap", 0);
	return 0;
}
