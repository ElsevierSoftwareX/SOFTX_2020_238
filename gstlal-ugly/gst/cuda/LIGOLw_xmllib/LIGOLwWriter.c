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
#define MAXLINESIZE (1 << 16)

void testXmlwriterFilename(const char *uri);
xmlChar *ConvertInput(const char *in, const char *encoding);

int ligoxml_write_Param(xmlTextWriterPtr writer, XmlParam *xparamPtr, const xmlChar* xml_type,
                        const xmlChar* Name)
{
    int rc;

    // Add the Param Node
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Param");

    // Add attributes to it
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST xml_type);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST Name);

    xmlChar *param = malloc(PARAMBUFSIZE);
    memset(param, 0, PARAMBUFSIZE);
    int index;
    index = ligoxml_get_type_index(xml_type);

    typeMap[index].dts_func(param, xml_type, xparamPtr->data, 0);

    rc = xmlTextWriterWriteString(writer, param);
    if (rc < 0) {
	    fprintf(stderr, "Error writing param");
	    return -1;
    }
    free(param);

    rc = xmlTextWriterEndElement(writer);

	return rc;

}

int ligoxml_write_Array(xmlTextWriterPtr writer, XmlArray *xarrayPtr, const xmlChar* xml_type, 
                        const xmlChar* delimiter, const xmlChar* Name)
{
    printf("write array\n");
    int rc;

    // Add the Array Node
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Array");

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

    int numInLine = xarrayPtr->dim[xarrayPtr->ndim - 1];
    xmlChar *line; 
    int line_size = CHARBUFSIZE * rows * numInLine;
    line = malloc(line_size);
    memset(line, 0, line_size);
    xmlChar token[CHARBUFSIZE];
    int index;
    index = ligoxml_get_type_index(xml_type);


    // Write Stream Node
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Stream");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST "Local");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Delimiter", BAD_CAST delimiter);
    for (j = 0; j < rows; ++j)
    {
        //printf("row = %d\n", j);
        for (i = 0; i < numInLine; ++i)
        {
            typeMap[index].dts_func((char *)token, xml_type, xarrayPtr->data, j * numInLine + i);
            //printf("i = %d, token %s\n", i, (const char *)token);
            strcat((char *)token, (const char *)delimiter);
            strcat((char *)line, (const char *)token);
        }
        //printf("line = %s\n", (char *)line);
        rc = xmlTextWriterWriteString(writer, BAD_CAST "\n\t\t\t\t");
        rc = xmlTextWriterWriteString(writer, line);
	if (rc < 0)
		printf("write string error %d \n", rc);
        line[0] = '\0';
    }
    free(line);
    // End the Stream Node
    rc = xmlTextWriterWriteString(writer, BAD_CAST "\n\t\t\t");
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
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Delimiter", BAD_CAST xtablePtr->delimiter->str);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST "Local");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST xtablePtr->tableName->str);
    
    rc = xmlTextWriterWriteString(writer, BAD_CAST "\n");
    for (i = 0; i < rows; ++i)
    {
        // read each line
        GString *line = g_string_new("\t\t\t\t");
        GString *type; //, *content;
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
    rc = xmlTextWriterWriteString(writer, BAD_CAST "\t\t\t");

    rc = xmlTextWriterEndElement(writer);

    // End Table Node
    xmlTextWriterEndElement(writer);

	return rc;
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


#else
#endif
