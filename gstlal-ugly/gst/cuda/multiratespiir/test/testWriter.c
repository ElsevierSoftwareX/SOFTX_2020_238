#include "../LIGOLwHeader.h"
#include <libxml/xmlreader.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>

#define MY_ENCODING "utf-8"

xmlChar *ConvertInput(const char *in, const char *encoding);

XmlArray xarray = 
{
    2, {2, 10, 0}, NULL
};

XmlParam xparams[4]; 
XmlTable xtable;

/**
 * testXmlwriterFilename:
 * @uri: the output URI
 *
 * test the xmlWriter interface when writing to a new file
 */
void
testXmlwriterFilename(const char *uri)
{
    int rc;
    xmlTextWriterPtr writer;
    xmlChar *tmp;

    /* Create a new XmlWriter for uri, with no compression. */
    writer = xmlNewTextWriterFilename(uri, 0);
    if (writer == NULL) {
        printf("testXmlwriterFilename: Error creating the xml writer\n");
        return;
    }

    rc = xmlTextWriterSetIndent(writer, 1);
    rc = xmlTextWriterSetIndentString(writer, "\t");

    /* Start the document with the xml default for the version,
     * encoding utf-8 and the default for the standalone
     * declaration. */
    rc = xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartDocument\n");
        return;
    }

    rc = xmlTextWriterWriteDTD(writer, "LIGO_LW", NULL, "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
        return;
    }

    /* Start an element named "LIGO_LW". Since thist is the first
     * element, this will be the root element of the document. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Start an element named "LIGO_LW" as child of EXAMPLE. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Add an attribute with name "Name" and value "gstlal_iir_bank_Bank" to LIGO_LW. */
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
                                     BAD_CAST "gstlal_iir_bank_Bank");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
        return;
    }

    ligoxml_write_Array(writer, &xarray, "real_8", " ", "e:array");

    ligoxml_write_Param(writer, xparams + 0, "real_4", "FLOAT");
    ligoxml_write_Param(writer, xparams + 1, "real_8", "DOUBLE");
    ligoxml_write_Param(writer, xparams + 2, "int_4s", "INT");
    ligoxml_write_Param(writer, xparams + 3, "lstring", "STRING");

    ligoxml_write_Table(writer, &xtable);

    /* Start an element named "Param" as child of LIGO_LW. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "HEADER");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Write an element named "X_ORDER_ID" as child of HEADER. */
    rc = xmlTextWriterWriteFormatElement(writer, BAD_CAST "X_ORDER_ID",
                                         "%010d", 53535);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteFormatElement\n");
        return;
    }

    /* Write an element named "CUSTOMER_ID" as child of HEADER. */
    rc = xmlTextWriterWriteFormatElement(writer, BAD_CAST "CUSTOMER_ID",
                                         "%d", 1010);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteFormatElement\n");
        return;
    }

    /* Write an element named "NAME_1" as child of HEADER. */
    tmp = ConvertInput("Müller", MY_ENCODING);
    rc = xmlTextWriterWriteElement(writer, BAD_CAST "NAME_1", tmp);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteElement\n");
        return;
    }
    if (tmp != NULL) xmlFree(tmp);

    /* Write an element named "NAME_2" as child of HEADER. */
    tmp = ConvertInput("Jörg", MY_ENCODING);
    rc = xmlTextWriterWriteElement(writer, BAD_CAST "NAME_2", tmp);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteElement\n");
        return;
    }
    if (tmp != NULL) xmlFree(tmp);

    /* Close the element named HEADER. */
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndElement\n");
        return;
    }

    /* Start an element named "ENTRIES" as child of ORDER. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "ENTRIES");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Start an element named "ENTRY" as child of ENTRIES. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "ENTRY");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Write an element named "ARTICLE" as child of ENTRY. */
    rc = xmlTextWriterWriteElement(writer, BAD_CAST "ARTICLE",
                                   BAD_CAST "<Test>");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteElement\n");
        return;
    }

    /* Write an element named "ENTRY_NO" as child of ENTRY. */
    rc = xmlTextWriterWriteFormatElement(writer, BAD_CAST "ENTRY_NO", "%d",
                                         10);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteFormatElement\n");
        return;
    }

    /* Close the element named ENTRY. */
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndElement\n");
        return;
    }

    /* Start an element named "ENTRY" as child of ENTRIES. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "ENTRY");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Write an element named "ARTICLE" as child of ENTRY. */
    rc = xmlTextWriterWriteElement(writer, BAD_CAST "ARTICLE",
                                   BAD_CAST "<Test 2>");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteElement\n");
        return;
    }

    /* Write an element named "ENTRY_NO" as child of ENTRY. */
    rc = xmlTextWriterWriteFormatElement(writer, BAD_CAST "ENTRY_NO", "%d",
                                         20);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteFormatElement\n");
        return;
    }

    /* Close the element named ENTRY. */
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndElement\n");
        return;
    }

    /* Close the element named ENTRIES. */
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndElement\n");
        return;
    }

    /* Start an element named "FOOTER" as child of ORDER. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "FOOTER");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Write an element named "TEXT" as child of FOOTER. */
    rc = xmlTextWriterWriteElement(writer, BAD_CAST "TEXT",
                                   BAD_CAST "This is a text.");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteElement\n");
        return;
    }

    /* Close the element named FOOTER. */
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndElement\n");
        return;
    }

    /* Here we could close the elements ORDER and EXAMPLE using the
     * function xmlTextWriterEndElement, but since we do not want to
     * write any other elements, we simply call xmlTextWriterEndDocument,
     * which will do all the work. */
    rc = xmlTextWriterEndDocument(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
        return;
    }

    xmlFreeTextWriter(writer);
}

/**
 * ConvertInput:
 * @in: string in a given encoding
 * @encoding: the encoding used
 *
 * Converts @in into UTF-8 for processing with libxml2 APIs
 *
 * Returns the converted UTF-8 string, or NULL in case of error.
 */
xmlChar *
ConvertInput(const char *in, const char *encoding)
{
    xmlChar *out;
    int ret;
    int size;
    int out_size;
    int temp;
    xmlCharEncodingHandlerPtr handler;

    if (in == 0)
        return 0;

    handler = xmlFindCharEncodingHandler(encoding);

    if (!handler) {
        printf("ConvertInput: no encoding handler found for '%s'\n",
               encoding ? encoding : "");
        return 0;
    }

    size = (int) strlen(in) + 1;
    out_size = size * 2 - 1;
    out = (unsigned char *) xmlMalloc((size_t) out_size);

    if (out != 0) {
        temp = size - 1;
        ret = handler->input(out, &out_size, (const xmlChar *) in, &temp);
        if ((ret < 0) || (temp - size + 1)) {
            if (ret < 0) {
                printf("ConvertInput: conversion wasn't successful.\n");
            } else {
                printf
                    ("ConvertInput: conversion wasn't successful. converted: %i octets.\n",
                     temp);
            }

            xmlFree(out);
            out = 0;
        } else {
            out = (unsigned char *) xmlRealloc(out, out_size + 1);
            out[out_size] = 0;  /*null terminating out */
        }
    } else {
        printf("ConvertInput: no mem\n");
    }

    return out;
}

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
