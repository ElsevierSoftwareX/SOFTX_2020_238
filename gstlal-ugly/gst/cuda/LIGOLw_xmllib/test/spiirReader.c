#include "../LIGOLwHeader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "multiratespiir.h"

extern void readArray(xmlTextReaderPtr reader, void *data);
extern void readParam(xmlTextReaderPtr reader, void *data);

extern void freeArray(XmlArray *array);
extern void freeParam(XmlParam *param);

extern void
streamFile(const char *filename, XmlNodeTag *xnt, int len); 

void
spiir_state_load_bank(SpiirState **spstate, const char *filename, cudaStream_t stream)
{
    LIBXML_TEST_VERSION

	XmlNodeTag	prexnt;
	XmlParam	xparam = {0, NULL};
	strncpy((char *)prexnt.tag, "Param-sample_rate:param", XMLSTRMAXLEN);
	prexnt.processPtr = readParam;
	prexnt.data = &xparam;

	// start parsing, get the information of depth
    streamFile(filename, &prexnt, 1);

	int depth = 0;
	int maxrate = 1;
	int temp;
	int i;
	gchar **rates = g_strsplit((const gchar*)xparam.data, (const gchar*)",", 16);
	while (rates[depth])
	{
		temp = atoi((const char*)rates[depth]);
		maxrate = maxrate > temp ? maxrate : temp;
		++depth;	
	}
	g_strfreev(rates);
	freeParam(&xparam);

	XmlNodeTag	*inxnt		= (XmlNodeTag*)malloc(sizeof(XmlNodeTag)*depth*3);
	XmlArray	*d_array	= (XmlArray*)malloc(sizeof(XmlArray)*depth);
	XmlArray	*a_array	= (XmlArray*)malloc(sizeof(XmlArray)*depth);
	XmlArray	*b_array	= (XmlArray*)malloc(sizeof(XmlArray)*depth);
	for (i = 0; i < depth; ++i)
	{
		// configure d_array 
		d_array[i].ndim = 0;
		sprintf((char *)inxnt[i + 0 * depth].tag, "Array-d_%d:array", maxrate >> i);
		inxnt[i + 0 * depth].processPtr = readArray;
		inxnt[i + 0 * depth].data = d_array + i;

		// configure a_array
		a_array[i].ndim = 0;
		sprintf((char *)inxnt[i + 1 * depth].tag, "Array-a_%d:array", maxrate >> i);
		inxnt[i + 1 * depth].processPtr = readArray;
		inxnt[i + 1 * depth].data = a_array + i;	

		// configure b_array
		b_array[i].ndim = 0;
		sprintf((char *)inxnt[i + 2 * depth].tag, "Array-b_%d:array", maxrate >> i);
		inxnt[i + 2 * depth].processPtr = readArray;
		inxnt[i + 2 * depth].data = b_array + i;	
	}

	// start parsing xml file, get the requested array
    streamFile(filename, inxnt, depth * 3);

	// free array memory 
	int num_filters, num_templates;
	int j, k;
	for (i = 0; i < depth; ++i)
	{
		num_filters		= (gint)d_array[i].dim[0];
		num_templates	= (gint)d_array[i].dim[1];
		// spstate[i]->d_d = (long*)inxnt[i].data;
		spstate[i]->d_d = (int*)malloc(sizeof(int)*num_filters*num_templates);
		printf("%d - d_dim: (%d, %d) a_dim: (%d, %d) b_dim: (%d, %d)\n", i, d_array[i].dim[0], d_array[i].dim[1],
				a_array[i].dim[0], a_array[i].dim[1], b_array[i].dim[0], b_array[i].dim[1]);

		spstate[i]->num_filters		= num_filters;
		spstate[i]->num_templates	= num_templates;
		spstate[i]->d_a1 = (COMPLEX_F*)malloc(sizeof(COMPLEX_F)*num_filters*num_templates);
		spstate[i]->d_b0 = (COMPLEX_F*)malloc(sizeof(COMPLEX_F)*num_filters*num_templates);
		for (j = 0; j < num_filters; ++j)
		{
			for (k = 0; k < num_templates; ++k)
			{
				spstate[i]->d_d[j * num_templates + k] = ((long*)(d_array[i].data))[j * num_templates + k];
				spstate[i]->d_a1[j * num_templates + k].re = ((double*)(a_array[i].data))[j * 2 * num_templates + k];
				spstate[i]->d_a1[j * num_templates + k].im = ((double*)(a_array[i].data))[j * 2 * num_templates + num_templates + k];
				spstate[i]->d_b0[j * num_templates + k].re = ((double*)(b_array[i].data))[j * 2 * num_templates + k];
				spstate[i]->d_b0[j * num_templates + k].im = ((double*)(b_array[i].data))[j * 2 * num_templates + num_templates + k];
			}
		}

		freeArray(d_array + i);
		freeArray(a_array + i);
		freeArray(b_array + i);

		printf("1st a: (%.3f + %.3fi) 1st b: (%.3f + %.3fi) 1st d: %d\n", spstate[i]->d_a1[0].re, spstate[i]->d_a1[0].im,
				spstate[i]->d_b0[0].re, spstate[i]->d_b0[0].im, spstate[i]->d_d[0]);
	}
	
	free(inxnt);

    xmlCleanupParser();
    xmlMemoryDump();
}

int main(int argc, char **argv) {
    if (argc != 2)
        return(1);


/*
	XmlNodeTag *xnt = (XmlNodeTag *)malloc(sizeof(XmlNodeTag) * PROCESSLEN);
	XmlArray xarray = {0, {1, 1, 1}, NULL};
	XmlParam xparam = {0, NULL};
	XmlTable xtable = {NULL, NULL};

    strncpy((char *)xnt[0].tag, "Array-b:array", XMLSTRMAXLEN);
    xnt[0].processPtr = readArray;
    xnt[0].data = &xarray;
    strncpy((char *)xnt[1].tag, "Param-helmet:param", XMLSTRMAXLEN);
    xnt[1].processPtr = readParam;
    xnt[1].data = &xparam;
*/

	SpiirState *spstates[7];
	int i;
	for (i = 0; i < 7; ++i)
		spstates[i] = (SpiirState*)malloc(sizeof(SpiirState));
	spiir_state_load_bank(spstates, argv[1], NULL);

/*
    float *data = (float*)xarray.data;
    int i, j;
    for (i = 0; i < xarray.dim[0]; ++i)
    {
        for (j = 0; j < xarray.dim[1]; ++j)
        {
            printf("data[%d][%d] = %.1f ", i, j, data[i * xarray.dim[1] + j]);
        }
        printf("\n");
    }

    printf("%s --> %d\n", xnt[1].tag, *((int*)xparam.data));

    printf("\nTable Name: %s; Columns: %u;\n", (xtable.tableName)->str, (xtable.names)->len);
    GHashTable *hash = xtable.hashContent;
    for (i = 0; i < (xtable.names)->len; ++i)
    {
        GString *colName = &g_array_index(xtable.names, GString, i);
        printf("colName: %s\n", colName->str);
        XmlHashVal *val = g_hash_table_lookup(hash, colName);
        printf("type: %s name: %s\n", val->type->str, val->name->str);
        for (j = 0; j < val->data->len; ++j)
        {
            if (strcmp(val->type->str, "real_4") == 0) {
                printf("%.2f\t", g_array_index(val->data, float, j));
            } else if (strcmp(val->type->str, "real_8") == 0) {
                printf("%.2lf\t", g_array_index(val->data, double, j));
            } else if (strcmp(val->type->str, "int_4s") == 0) {
                printf("%d\t", g_array_index(val->data, int, j));
            } else {
                printf("%s\t", (g_array_index(val->data, GString, j)).str);
            }
        }

        printf("\n");
    }

    free(xarray.data);
*/
    return(0);
}
