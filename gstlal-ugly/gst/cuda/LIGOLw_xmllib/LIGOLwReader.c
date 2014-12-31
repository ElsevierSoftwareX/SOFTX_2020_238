/**
 * section: xmlReader
 * synopsis: Parse an XML file with an xmlReader
 * purpose: Demonstrate the use of xmlReaderForFile() to parse an XML file
 *          and dump the informations about the nodes found in the process.
 *          (Note that the XMLReader functions require libxml2 version later
 *          than 2.6.)
 * usage: reader1 <filename>
 * test: reader1 test2.xml > reader1.tmp && diff reader1.tmp $(srcdir)/reader1.res
 * author: Daniel Veillard
 * copy: see Copyright for the status of this software.
 */

#include <stdio.h>
#include <libxml/xmlreader.h>
#include <string.h>
#include "LIGOLwHeader.h"

// In Array Node, No Dim sub node should appear after Stream sub node
void readArray(xmlTextReaderPtr reader, void *data)
{
    printf("I'm Array\n");
    int i, rows, ntoken, ret, nodeType;
    size_t bytes;
    xmlChar *delimiter, *type;
    xmlChar *line, *token;
    const xmlChar *name;
    char *saveLinePtr, *saveTokenPtr;
    XmlArray *xArrayPtr = (XmlArray*)data;

    type = xmlTextReaderGetAttribute(reader, BAD_CAST "Type");
    #ifdef __DEBUG__
    printf("type = %s\n", (char *) type);
    #endif

    while (1)
    {
        ret = xmlTextReaderRead(reader); 
        if (ret != 1) 
        {
            fprintf(stderr, "Wrong\n");
            break;
        }
        
        name = xmlTextReaderConstName(reader);
        nodeType = xmlTextReaderNodeType(reader);
        #ifdef __DEBUG__
        printf("name = %s\n", (char *) name);
        #endif

        // make sure this is not the end node
        if (xmlStrcmp(name, BAD_CAST "Dim") == 0 && nodeType != 15)
        {
            // get the #text node of <Dim>
            ret = xmlTextReaderRead(reader);
            if (ret != 1)
            {
                fprintf(stderr, "Dim Wrong\n");
                break;
            }
            #ifdef __DEBUG__
            printf("Dim %d\n", atoi((const char *)xmlTextReaderConstValue(reader)));
            #endif
            // get and set the dim value
            xArrayPtr->dim[xArrayPtr->ndim++] = atoi((const char*)xmlTextReaderConstValue(reader));
        }

        if (xmlStrcmp(name, BAD_CAST "Stream") == 0 && nodeType != 15)
        {
            delimiter = xmlTextReaderGetAttribute(reader, BAD_CAST "Delimiter");
            // get the #text node of <Stream>
            ret = xmlTextReaderRead(reader);
            if (ret != 1)
            {
                fprintf(stderr, "Dim Wrong\n");
                break;
            }
            #ifdef __DEBUG__
            printf("Stream %s\n", (char *) xmlTextReaderConstValue(reader));
            #endif

            xmlChar* copy = xmlStrdup(xmlTextReaderConstValue(reader));
            // cacluate the number of rows
            rows = 1;
            for (i = 0; i < xArrayPtr->ndim - 1; ++i)
                rows *= xArrayPtr->dim[i];

            // allocate memories for data of type node.attribute(Type)
            // and of length rows * xArrayPtr->dim[ndim - 1]
            bytes = rows * xArrayPtr->dim[xArrayPtr->ndim - 1] * ligoxml_get_type_size(type);
            xArrayPtr->data = malloc(bytes);  

            // read each line
            line = (xmlChar *)strtok_r((char *)copy, "\n", &saveLinePtr);   
            for (i = 0; i < rows; ++i)
            {
                #ifdef __DEBUG__
                printf("line = %s\n", line);
                #endif

                xmlChar* copyline = xmlStrdup(line);
                // token = strtok_r(copyline, " ", &saveTokenPtr);
                token = (xmlChar*)strtok_r((char *)copyline, (char *)delimiter, &saveTokenPtr);
                ntoken = 0;
                while (token != NULL)
                {
                    #ifdef __DEBUG__
                    printf("token: %s (%d * %d + %d = %d)\n", (char *)token, i, 
                            xArrayPtr->dim[xArrayPtr->ndim-1], ntoken, i * xArrayPtr->dim[xArrayPtr->ndim - 1] + ntoken);
                    #endif
                    sscanf((char *)token, (char *)ligoxml_get_type_format(type), xArrayPtr->data + (i * xArrayPtr->dim[xArrayPtr->ndim - 1] + ntoken) * ligoxml_get_type_size(type));
                    // token = strtok_r(NULL, " ", &saveTokenPtr);
                    token = (xmlChar*)strtok_r(NULL, (char *)delimiter, &saveTokenPtr);
                    ++ntoken;
                }

                line = (xmlChar*)strtok_r(NULL, "\n", &saveLinePtr);
                free(copyline);
            }
            free(copy);
        }

        // 15 stands for end node
        if (xmlStrcmp(name, BAD_CAST "Array") == 0 && nodeType == 15)
        {
            // Work Done. Break out of the while loop
            break;
        }
    }
}

void freeArray(XmlArray *array)
{
	free(array->data);
}

void readParam(xmlTextReaderPtr reader, void *data)
{
    printf("I'm Param\n");

    xmlChar *type;
    const xmlChar *name, *content; 
    int nodeType;
    int ret;
    XmlParam *xmlParamPtr = (XmlParam*)data;

    type = xmlTextReaderGetAttribute(reader, BAD_CAST "Type");

    while (1)
    {
        ret = xmlTextReaderRead(reader);
        if (ret != 1)
        {
            fprintf(stderr, "Wrong\n");
            break;
        }

        name = xmlTextReaderConstName(reader);
        nodeType = xmlTextReaderNodeType(reader);
        content = xmlTextReaderConstValue(reader); 

        if (xmlStrcmp(name, BAD_CAST "#text") == 0)
        {
            #ifdef __DEBUG__
            printf("content = %s\n", content);
            #endif

            // allocate memory 
            if (xmlStrcmp(type, BAD_CAST "lstring") == 0) {

                // string like type needs special care
                xmlParamPtr->bytes = xmlStrlen(content) + 1;
                xmlParamPtr->data = malloc(xmlParamPtr->bytes);
                ((xmlChar*)xmlParamPtr->data)[xmlParamPtr->bytes - 1] = 0;

            } else {

                // other numeric type
                xmlParamPtr->bytes = ligoxml_get_type_size(type);
                xmlParamPtr->data = malloc(xmlParamPtr->bytes);
            }

            // save content
            sscanf((char *)content, ligoxml_get_type_format(type), xmlParamPtr->data);
        }
         
        // Meet node </Param>, just break out the loop
        if (xmlStrcmp(name, BAD_CAST "Param") == 0 && nodeType == 15)
        {
            break;
        }
    }

}

void freeParam(XmlParam *param)
{
	free(param->data);	
}

void readTable(xmlTextReaderPtr reader, void *data)
{
    printf("I'm Table\n");

    XmlTable *xmlTable = (XmlTable*)data;

    // init the table structure
    ligoxml_init_XmlTable(xmlTable);

    int ret, nodeType, numCol;
    const xmlChar *nodeName, *tableName;
    const xmlChar *columnName, *columnType;
    const xmlChar *delimiter;

    int i, j, numLine;
    GString *colName;

    tableName = xmlTextReaderGetAttribute(reader, BAD_CAST "Name");
    xmlTable->tableName = g_string_new((const gchar*)tableName);

    numCol = 0;
    while (1)
    {
        ret = xmlTextReaderRead(reader);
        if (ret != 1)
        {
            fprintf(stderr, "Wrong\n");
            break;
        }

        nodeName = xmlTextReaderConstName(reader);
        nodeType = xmlTextReaderNodeType(reader);

        if (xmlStrcmp(nodeName, BAD_CAST "Column") == 0)
        {
            // number of column increase
            ++numCol;

            columnName = xmlTextReaderGetAttribute(reader, BAD_CAST "Name");
            columnType = xmlTextReaderGetAttribute(reader, BAD_CAST "Type");
            #ifdef __DEBUG__
            printf("Column: %s Type: %s\n", columnName, columnType);
            #endif

            g_array_append_val(xmlTable->names, *g_string_new((const gchar*)columnName));

            // a wierd bug, doesn't work if I use "XmlHashVal val"
            XmlHashVal *val = (XmlHashVal*)malloc(sizeof(XmlHashVal));
            val->type = g_string_new((const gchar*)columnType); 
            val->name = g_string_new((const gchar*)columnName);
            if (xmlStrcmp(columnType, BAD_CAST "real_4") == 0) {
                val->data = g_array_new(FALSE, FALSE, sizeof(float)); 
            } else if (xmlStrcmp(columnType, BAD_CAST "real_8") == 0) {
                val->data = g_array_new(FALSE, FALSE, sizeof(double));
            } else if (xmlStrcmp(columnType, BAD_CAST "int_4s") == 0) {
                val->data = g_array_new(FALSE, FALSE, sizeof(int));
            } else {
                // arry of string
                val->data = g_array_new(FALSE, FALSE, sizeof(GString));
            }
            g_hash_table_insert(xmlTable->hashContent, g_string_new((const gchar*)columnName), (gpointer)(val));
        }

        if (xmlStrcmp(nodeName, BAD_CAST "Stream") == 0 && nodeType != 15)
        {
            delimiter = xmlTextReaderGetAttribute(reader, BAD_CAST "Delimiter");
            
            // save delimiter
            xmlTable->delimiter = g_string_new((const gchar*)delimiter);

            // get the #text node of <Stream>
            ret = xmlTextReaderRead(reader);

            GString *content = g_string_new((const gchar*)xmlTextReaderConstValue(reader));
            gchar **lines = g_strsplit(content->str, "\n", 0);
            gchar **tokens;
            
            i = 0;
            numLine = g_strv_length(lines) - 2;
            #ifdef __DEBUG__
            printf("numLine: %d\n", numLine);
            #endif
            // by pass the first line and the last line
            for (i = 1; i <= numLine; ++i)
            {
                #ifdef __DEBUG__
                printf("line len: %zu line %d: %s\n", strlen(lines[i]), i, (const char*)lines[i]);
                #endif

                // split the line by delimiter
                tokens = g_strsplit(lines[i], (const gchar*)delimiter, 0);

                for (j = 0; j < numCol; ++j)
                {
                    #ifdef __DEBUG__
                    printf("len: %zu, token %d: %s\n", strlen(tokens[j]), j, (const char*)tokens[j]);
                    #endif

                    colName = &g_array_index(xmlTable->names, GString, j);

                    #ifdef __DEBUG__
                    printf("colName = %s\n", colName->str);
                    #endif

                    XmlHashVal *valPtr = g_hash_table_lookup(xmlTable->hashContent, (gpointer)colName);
                    if (strcmp(valPtr->type->str, "real_4") == 0) {
                        float num;
                        sscanf(tokens[j], "%f", &num);
                        g_array_append_val(valPtr->data, num);
                    } else if (strcmp(valPtr->type->str, "real_8") == 0) {
                        double num;
                        sscanf(tokens[j], "%lf", &num);
                        g_array_append_val(valPtr->data, num);
                    } else if (strcmp(valPtr->type->str, "int_4s") == 0) {
                        int num;
                        sscanf(tokens[j], "%d", &num);
                        g_array_append_val(valPtr->data, num);
                    } else {
                        // string
                        GString *val = g_string_new(tokens[j]);
                        g_array_append_val(valPtr->data, *val);
                    }

                    #ifdef __DEBUG__
                    printf("name: %s\n", valPtr->name->str);
                    #endif
                }

                g_strfreev(tokens);
            }

            g_strfreev(lines);
            g_string_free(content, TRUE);
        }

        // Meet node </Table>, just break out of the loop
        if (xmlStrcmp(nodeName, BAD_CAST "Table") == 0 && nodeType == 15)
        {
            #ifdef __DEBUG__
            g_print("hash table size: %d\n", g_hash_table_size(xmlTable->hashContent));
            #endif
            break;
        }
    }
}

void freeTable(XmlTable *table)
{
	// to be updated
}

#ifdef LIBXML_READER_ENABLED

/**
 * processNode:
 * @reader: the xmlReader
 *
 * Dump information about the current node
 */
void
processNode(xmlTextReaderPtr reader, XmlNodeStruct *xns, int len) {
    const xmlChar *name, *value;

    name = xmlTextReaderConstName(reader);
    if (name == NULL)
	name = BAD_CAST "--";

    value = xmlTextReaderGetAttribute(reader, BAD_CAST "Name");
    
    xmlChar *tag = xmlStrncatNew(name, BAD_CAST "-", -1);
    xmlStrcat(tag, value);

    if (xmlTextReaderNodeType(reader) == 15)
        return;

    // Could be optimized by using HashMap
    int i, ret;
    for (i = 0; i < len; ++i)
    {
        ret = xmlStrcmp(tag, xns[i].tag);
        if (ret == 0)
        {
            xns[i].processPtr(reader, xns[i].data);
            return;
        }
    }
}

/**
 * parseFile:
 * @filename: the file name to parse
 *
 * Parse and print information about an XML file.
 */
void
parseFile(const char *filename, XmlNodeStruct *xns, int len) {
    xmlTextReaderPtr reader;
    int ret;

    reader = xmlReaderForFile(filename, NULL, 0);
    if (reader != NULL) {
        ret = xmlTextReaderRead(reader);
        while (ret == 1) {
            processNode(reader, xns, len);
            ret = xmlTextReaderRead(reader);
        }
        xmlFreeTextReader(reader);
        if (ret != 0) {
            fprintf(stderr, "%s : failed to parse\n", filename);
        }
    } else {
        fprintf(stderr, "Unable to open %s\n", filename);
    }
}


#else
int main(void) {
    fprintf(stderr, "XInclude support not compiled in\n");
    exit(1);
}
#endif
