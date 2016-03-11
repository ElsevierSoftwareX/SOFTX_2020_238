#include "../LIGOLwHeader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define PROCESSLEN  3

int main(int argc, char **argv) {
    if (argc != 2)
        return(1);

    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    /*
     * this initialize the process functions
     */
	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * PROCESSLEN);
	XmlArray xarray = {0, {1, 1, 1}, NULL};
	XmlParam xparam = {0, NULL};
	XmlTable xtable = {NULL, NULL};

    strncpy((char *)xns[0].tag, "Array-b:array", XMLSTRMAXLEN);
    xns[0].processPtr = readArray;
    xns[0].data = &xarray;
    strncpy((char *)xns[1].tag, "Param-helmet:param", XMLSTRMAXLEN);
    xns[1].processPtr = readParam;
    xns[1].data = &xparam;
    strncpy((char *)xns[2].tag, "Table-sngl_inspiral:table", XMLSTRMAXLEN);
    xns[2].processPtr = readTable;
    xns[2].data = &xtable;

    parseFile(argv[1], xns, PROCESSLEN);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();

    /*
     * Validate correctness
     */
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

    printf("%s --> %d\n", xns[1].tag, *((int*)xparam.data));

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

    /*
     * deallocate resources
     */
    free(xarray.data);
    return(0);
}
