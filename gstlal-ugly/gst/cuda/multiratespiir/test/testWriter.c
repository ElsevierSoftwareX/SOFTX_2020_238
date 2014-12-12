
int
main(int argc, char *argv[])
{
    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    /* initialize array data */
    xarray.data = malloc(sizeof(double)*20);
    int i;
    for (i = 0; i < 20; ++i)
        ((double*)xarray.data)[i] = i % 7;

    /* initialize params data */
    xparams[0].data = malloc(sizeof(float));
    xparams[1].data = malloc(sizeof(double));
    xparams[2].data = malloc(sizeof(int));
    xparams[3].data = malloc(sizeof(char) * 6);
    sscanf("1.2", "%f", xparams[0].data);
    sscanf("2.3", "%lf", xparams[1].data);
    sscanf("7", "%d", xparams[2].data);
    sscanf("Hello", "%s", xparams[3].data);

    /* initialize table data */
    xy_table_init(&xtable);

    /* first, the file version */
    testXmlwriterFilename(argv[1]);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();

    /* free memory */
    free(xarray.data);
    for (i = 0; i < 4; ++i)
        free(xparams[i].data);
    return 0;
}
