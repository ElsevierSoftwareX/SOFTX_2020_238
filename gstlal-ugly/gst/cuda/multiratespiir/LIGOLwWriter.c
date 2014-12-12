/**
 * section: xmlWriter
 * synopsis: use various APIs for the xmlWriter
 * purpose: tests a number of APIs for the xmlWriter, especially
 *          the various methods to write to a filename, to a memory
 *          buffer, to a new document, or to a subtree. It shows how to
 *          do encoding string conversions too. The resulting
 *          documents are then serialized.
 * usage: testWriter
 * test: testWriter && for i in 1 2 3 4 ; do diff $(srcdir)/writer.xml writer$$i.tmp || break ; done
 * author: Alfred Mickautsch
 * copy: see Copyright for the status of this software.
 */
#include <stdio.h>
#include <string.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
// #include "xyHeader.h"
#include "LIGOLwHeader.h"

#if defined(LIBXML_WRITER_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)

#define MY_ENCODING "utf-8"

#define CHARBUFSIZE 32
#define PARAMBUFSIZE 1024
#define MAXLINESIZE (1 << 13)

XmlArray xarray = 
{
    2, {2, 10, 0}, NULL
};

XmlParam xparams[4]; 
XmlTable xtable;

void testXmlwriterFilename(const char *uri);
xmlChar *ConvertInput(const char *in, const char *encoding);

int ligoxml_write_Param(xmlTextWriterPtr writer, XmlParam *xparamPtr, const xmlChar* xml_type,
                        const xmlChar* Name)
{
    int rc;

    // Add the Param Node
    rc = xmlTextWriterStartElement(writer, "Param");

    // Add attributes to it
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST xml_type);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST Name);

    xmlChar *param = malloc(PARAMBUFSIZE);
    memset(param, 0, PARAMBUFSIZE);
    if (xmlStrcmp(xml_type, "real_4") == 0) {
        sprintf(param, "%.3f", *((float*)xparamPtr->data));
        rc = xmlTextWriterWriteString(writer, param);
    } else if (xmlStrcmp(xml_type, "real_8") == 0) {
        sprintf(param, "%.3lf", *((double*)xparamPtr->data));
        rc = xmlTextWriterWriteString(writer, param);
    } else if (xmlStrcmp(xml_type, "int_4s") == 0) {
        sprintf(param, "%d", *((int*)xparamPtr->data));
        rc = xmlTextWriterWriteString(writer, param);
    } else if (xmlStrcmp(xml_type, "lstring") == 0) {
        sprintf(param, "%s", (xmlChar*)xparamPtr->data);
        rc = xmlTextWriterWriteString(writer, param);
    } else {
        fprintf(stderr, "ERROR: UNKNOWN WRITING TYPE\n");
        return -1;
    }
    free(param);

    rc = xmlTextWriterEndElement(writer);
}

int ligoxml_write_Array(xmlTextWriterPtr writer, XmlArray *xarrayPtr, const xmlChar* xml_type, 
                        const xmlChar* delimiter, const xmlChar* Name)
{
    int rc;

    // Add the Array Node
    rc = xmlTextWriterStartElement(writer, "Array");

    // Add attributes to it
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST xml_type);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST Name);

    // Write DIM node
    int i, j;
    for (i = 0; i < xarrayPtr->ndim; ++i)
    {
        rc = xmlTextWriterWriteFormatElement(writer, BAD_CAST "DIM", "%d", xarrayPtr->dim[i]);
    }

    int rows = 1;
    for (i = 0; i < xarrayPtr->ndim - 1; ++i)
        rows *= xarrayPtr->dim[i];

    xmlChar *line; 
    line = malloc(MAXLINESIZE);
    memset(line, 0, MAXLINESIZE);
    xmlChar token[CHARBUFSIZE];

    // Write Stream Node
    rc = xmlTextWriterStartElement(writer, "Stream");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST "Local");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Delimiter", BAD_CAST delimiter);
    int numInLine = xarrayPtr->dim[xarrayPtr->ndim - 1];
    for (j = 0; j < rows; ++j)
    {
        printf("rows = %d\n", rows);
        for (i = 0; i < numInLine; ++i)
        {
            if (xmlStrcmp(xml_type, "real_4") == 0) {
                sprintf(token, "%.3f", ((float*)xarrayPtr->data)[j * numInLine + i]);
            } else if (xmlStrcmp(xml_type, "real_8") == 0) {
                sprintf(token, "%.3lf", ((double*)xarrayPtr->data)[j * numInLine + i]);
            } else if (xmlStrcmp(xml_type, "int_4s") == 0) {
                sprintf(token, "%d", ((int*)xarrayPtr->data)[j * numInLine + i]);
            } else {
                fprintf(stderr, "ERROR: UNKNOWN WRITING TYPE\n");
                return -1;
            }
            printf("token %s\n", token);
            strcat(token, delimiter);
            strcat(line, token);
        }
        printf("line = %s\n", line);
        rc = xmlTextWriterWriteString(writer, "\n\t\t\t\t");
        rc = xmlTextWriterWriteString(writer, line);
        line[0] = '\0';
    }
    free(line);
    // End the Stream Node
    rc = xmlTextWriterWriteString(writer, "\n\t\t\t");
    rc = xmlTextWriterEndElement(writer);

    // End the Array Node
    rc = xmlTextWriterEndElement(writer);

    return rc;
}

int ligoxml_write_Table(xmlTextWriterPtr writer, const XmlTable *xtablePtr)
{
    int rc;

    // Add the Table Node
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Table");

    // Add Attribute "Name"
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST xtablePtr->tableName->str);

    // Write Column Nodes
    int i, rows, j;
    rows = 0;
    for (i = 0; i < xtablePtr->names->len; ++i)
    {
        rc = xmlTextWriterStartElement(writer, BAD_CAST "Column");

        GString *colName = &g_array_index(xtablePtr->names, GString, i);
        XmlHashVal *val = g_hash_table_lookup(xtablePtr->hashContent, colName);
        GString *type = val->type;

        if (rows == 0)
            rows = val->data->len;

        rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST type->str);
        rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST colName->str);
        
        rc = xmlTextWriterEndElement(writer);
    }

    // Write Stream Nodes
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Stream"); 
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Delimiter", xtablePtr->delimiter->str);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST "Local");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST xtablePtr->tableName->str);
    
    rc = xmlTextWriterWriteString(writer, "\n");
    for (i = 0; i < rows; ++i)
    {
        // read each line
        GString *line = g_string_new("\t\t\t\t");
        GString *type, *content;
        XmlHashVal *hashVal;

        for (j = 0; j < xtablePtr->names->len; ++j)
        {
            hashVal = g_hash_table_lookup(xtablePtr->hashContent, &g_array_index(xtablePtr->names, GString, j));
            type = hashVal->type;
            if (strcmp(type->str, "real_4") == 0) {
                float num = g_array_index(hashVal->data, float, i);
                g_string_append_printf(line, "%f%s", num, xtablePtr->delimiter->str);
                #ifdef __DEBUG__
                printf("float %.3f\n", num);
                #endif
            } else if (strcmp(type->str, "real_8") == 0) {
                double num = g_array_index(hashVal->data, double, i);
                g_string_append_printf(line, "%lf%s", num, xtablePtr->delimiter->str);
            } else if (strcmp(type->str, "int_4s") == 0) {
                int num = g_array_index(hashVal->data, int, i);
                g_string_append_printf(line, "%d%s", num, xtablePtr->delimiter->str);
            } else {
                GString content = g_array_index(hashVal->data, GString, i);
                g_string_append_printf(line, "%s%s", content.str, xtablePtr->delimiter->str);
            }
        }
        g_string_append(line, "\n");
        // rc = xmlTextWriterWriteString(writer, line->str);
        rc = xmlTextWriterWriteFormatRaw(writer, line->str);
        g_string_free(line, TRUE);
    }
    rc = xmlTextWriterWriteString(writer, "\t\t\t");

    rc = xmlTextWriterEndElement(writer);

    // End Table Node
    xmlTextWriterEndElement(writer);
}

void xy_table_init(XmlTable *table)
{
    table->tableName = g_string_new("sngl_inspiral:table");

    table->delimiter = g_string_new(",");
    
    table->names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:cont_chisq"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:bank_chisq"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:chisq_dof"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:end_time_gmst"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:event_duration"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:event_id"));
    g_array_append_val(table->names, *g_string_new("sngl_inspiral:channel"));

    table->hashContent = g_hash_table_new((GHashFunc)g_string_hash, (GEqualFunc)g_string_equal);

    XmlHashVal *vals = (XmlHashVal*)malloc(sizeof(XmlHashVal)*7);

    float cont_chisq[3] = {0.1f, 0.3f, 0.2f};
    vals[0].name = g_string_new("sngl_inspiral:cont_chisq");
    vals[0].type = g_string_new("real_4");
    vals[0].data = g_array_new(FALSE, FALSE, sizeof(float));
    g_array_append_val(vals[0].data, cont_chisq[0]);
    g_array_append_val(vals[0].data, cont_chisq[1]);
    g_array_append_val(vals[0].data, cont_chisq[2]);
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:cont_chisq"), vals + 0);

    float bank_chisq[3] = {0.2f, 0.4f, 0.7f};
    vals[1].name = g_string_new("sngl_inspiral:bank_chisq");
    vals[1].type = g_string_new("real_4");
    vals[1].data = g_array_new(FALSE, FALSE, sizeof(float));
    g_array_append_val(vals[1].data, bank_chisq[0]);
    g_array_append_val(vals[1].data, bank_chisq[1]);
    g_array_append_val(vals[1].data, bank_chisq[2]);
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:bank_chisq"), vals + 1);

    int chisq_dof[3] = {3, 6, 9};
    vals[2].name = g_string_new("sngl_inspiral:chisq_dof");
    vals[2].type = g_string_new("int_4s");
    vals[2].data = g_array_new(FALSE, FALSE, sizeof(int));
    g_array_append_val(vals[2].data, chisq_dof[0]);
    g_array_append_val(vals[2].data, chisq_dof[1]);
    g_array_append_val(vals[2].data, chisq_dof[2]);
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:chisq_dof"), vals + 2);

    double end_time_gmst[3] = {0.4, 0.8, 0.5};
    vals[3].name = g_string_new("sngl_inspiral:end_time_gmst");
    vals[3].type = g_string_new("real_8");
    vals[3].data = g_array_new(FALSE, FALSE, sizeof(double));
    g_array_append_val(vals[3].data, end_time_gmst[0]);
    g_array_append_val(vals[3].data, end_time_gmst[1]);
    g_array_append_val(vals[3].data, end_time_gmst[2]);
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:end_time_gmst"), vals + 3);

    double event_duration[3] = {0.5, 0.9, 0.6};
    vals[4].name = g_string_new("sngl_inspiral:event_duration");
    vals[4].type = g_string_new("real_8");
    vals[4].data = g_array_new(FALSE, FALSE, sizeof(double));
    g_array_append_val(vals[4].data, event_duration[0]);
    g_array_append_val(vals[4].data, event_duration[1]);
    g_array_append_val(vals[4].data, event_duration[2]);
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:event_duration"), vals + 4);

    vals[5].name = g_string_new("sngl_inspiral:event_id");
    vals[5].type = g_string_new("ilwd:char");
    vals[5].data = g_array_new(FALSE, FALSE, sizeof(GString));
    g_array_append_val(vals[5].data, *g_string_new("\"sngl_inspiral:event_id:0\""));
    g_array_append_val(vals[5].data, *g_string_new("\"sngl_inspiral:event_id:1\""));
    g_array_append_val(vals[5].data, *g_string_new("\"sngl_inspiral:event_id:0\""));
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:event_id"), vals + 5);

    vals[6].name = g_string_new("sngl_inspiral:channel");
    vals[6].type = g_string_new("lstring");
    vals[6].data = g_array_new(FALSE, FALSE, sizeof(GString));
    g_array_append_val(vals[6].data, *g_string_new("\"FAKE-STRAIN\""));
    g_array_append_val(vals[6].data, *g_string_new("\"FAKE-STRAIN\""));
    g_array_append_val(vals[6].data, *g_string_new("\"FAKE-STRAIN\""));
    g_hash_table_insert(table->hashContent, g_string_new("sngl_inspiral:channel"), vals + 6);

    #ifdef __DEBUG__
    printf("hash table size: %u\n", g_hash_table_size(table->hashContent));
    #endif
}


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

#else
int main(void) {
    fprintf(stderr, "Writer or output support not compiled in\n");
    exit(1);
}
#endif
