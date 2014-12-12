#include "../LIGOLwHeader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern void processArray(xmlTextReaderPtr reader, void *data);
extern void processParam(xmlTextReaderPtr reader, void *data);
extern void processTable(xmlTextReaderPtr reader, void *data);

extern void
streamFile(const char *filename, XmlNodeTag *xnt, int len); 

#define PROCESSLEN  3
XmlNodeTag xnt[PROCESSLEN]; 

XmlArray xarray = {0, {1, 1, 1}, NULL};
XmlParam xparam = {0, NULL};
XmlTable xtable = {NULL, NULL};

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
    strncpy(xnt[0].tag, "Array-b:array", XMLSTRMAXLEN);
    xnt[0].processPtr = processArray;
    xnt[0].data = &xarray;
    strncpy(xnt[1].tag, "Param-helmet:param", XMLSTRMAXLEN);
    xnt[1].processPtr = processParam;
    xnt[1].data = &xparam;
    strncpy(xnt[2].tag, "Table-sngl_inspiral:table", XMLSTRMAXLEN);
    xnt[2].processPtr = processTable;
    xnt[2].data = &xtable;

    streamFile(argv[1], xnt, PROCESSLEN);

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

    /*
     * deallocate resources
     */
    free(xarray.data);
    return(0);
}
