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

#define XMLSTRMAXLEN    128 

typedef struct _XmlNodeTag
{
    // tag = Node-Name
    // for example: Array-b:array denotes an Array node
    // with Name attribute of value b:array
    xmlChar tag[XMLSTRMAXLEN];

    // process function pointer for this node
    // data is of void* type so as to be easily
    // converted to needed type
    void (*processPtr)(xmlTextReaderPtr reader, void *data);

    // the data to be processed
    void *data;

} XmlNodeTag;

typedef struct _XmlArray
{
    // number of dimensions
    int ndim;

    // 3 dimensional
    int dim[3];

    // raw data
    void *data; 

} XmlArray;

typedef struct _XmlParam
{
    // number of bytes
    int bytes;

    // raw data
    void *data;

} XmlParam;

// get the number of bytes this type requires
size_t get_type_size(xmlChar *type)
{
    if (!xmlStrcmp(type, "real_4")) {
        // of type 'float'
        return 4;
    } else if (!xmlStrcmp(type, "real_8")) {
        // of type 'double'
        return 8;
    } else if (!xmlStrcmp(type, "int_4s")) {
        // of type 'signed int'
        return 4;
    }

    // default value
    return 1;
}

// get the format string of this type for printing
xmlChar* get_type_char(xmlChar *type)
{
    if (!xmlStrcmp(type, "real_4")) {
        // of type 'float'
        return "%f";
    } else if (!xmlStrcmp(type, "real_8")) {
        // of type 'double'
        return "%lf";
    } else if (!xmlStrcmp(type, "int_4s")) {
        // of type 'signed int'
        return "%d";
    } 

    // of string type
    return "%s";
}

#define PROCESSLEN  2
XmlNodeTag xnt[PROCESSLEN]; 

XmlArray xarray = {0, {1, 1, 1}, NULL};
XmlParam xparam = {0, NULL};

// In Array Node, No Dim sub node should appear after Stream sub node
void processArray(xmlTextReaderPtr reader, void *data)
{
    printf("I'm Array\n");
    int i, rows, ntoken, ret, nodeType;
    size_t bytes;
    xmlChar *delimiter, *type;
    xmlChar *line, *token;
    const xmlChar *name;
    char *saveLinePtr, *saveTokenPtr;
    XmlArray *xArrayPtr = (XmlArray*)data;

    type = xmlTextReaderGetAttribute(reader, "Type");
    #ifdef __DEBUG__
    printf("type = %s\n", type);
    // xmlDocDump(stdout, node->doc);
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
        printf("name = %s\n", name);
        #endif

        // make sure this is not the end node
        if (xmlStrcmp(name, "Dim") == 0 && nodeType != 15)
        {
            // get the #text node of <Dim>
            ret = xmlTextReaderRead(reader);
            if (ret != 1)
            {
                fprintf(stderr, "Dim Wrong\n");
                break;
            }
            #ifdef __DEBUG__
            printf("Dim %d\n", atoi(xmlTextReaderConstValue(reader)));
            #endif
            // get and set the dim value
            xArrayPtr->dim[xArrayPtr->ndim++] = atoi(xmlTextReaderConstValue(reader));
        }

        if (xmlStrcmp(name, "Stream") == 0 && nodeType != 15)
        {
            delimiter = xmlTextReaderGetAttribute(reader, "Delimiter");
            // get the #text node of <Stream>
            ret = xmlTextReaderRead(reader);
            if (ret != 1)
            {
                fprintf(stderr, "Dim Wrong\n");
                break;
            }
            #ifdef __DEBUG__
            printf("Stream %s\n", xmlTextReaderConstValue(reader));
            #endif

            xmlChar* copy = xmlStrdup(xmlTextReaderConstValue(reader));
            // cacluate the number of rows
            rows = 1;
            for (i = 0; i < xArrayPtr->ndim - 1; ++i)
                rows *= xArrayPtr->dim[i];

            // allocate memories for data of type node.attribute(Type)
            // and of length rows * xArrayPtr->dim[ndim - 1]
            bytes = rows * xArrayPtr->dim[xArrayPtr->ndim - 1] * get_type_size(type);
            xArrayPtr->data = malloc(bytes);  

            // read each line
            line = strtok_r(copy, "\n", &saveLinePtr);   
            for (i = 0; i < rows; ++i)
            {
                #ifdef __DEBUG__
                printf("line = %s\n", line);
                #endif

                xmlChar* copyline = xmlStrdup(line);
                // token = strtok_r(copyline, " ", &saveTokenPtr);
                token = strtok_r(copyline, delimiter, &saveTokenPtr);
                ntoken = 0;
                while (token != NULL)
                {
                    #ifdef __DEBUG__
                    printf("token: %s (%d * %d + %d = %d)\n", token, i, 
                            xArrayPtr->dim[xArrayPtr->ndim-1], ntoken, i * xArrayPtr->dim[xArrayPtr->ndim - 1] + ntoken);
                    #endif
                    sscanf(token, get_type_char(type), xArrayPtr->data + (i * xArrayPtr->dim[xArrayPtr->ndim - 1] + ntoken) * get_type_size(type));
                    // token = strtok_r(NULL, " ", &saveTokenPtr);
                    token = strtok_r(NULL, delimiter, &saveTokenPtr);
                    ++ntoken;
                }

                line = strtok_r(NULL, "\n", &saveLinePtr);
                free(copyline);
            }
            free(copy);
        }

        // 15 stands for end node
        if (xmlStrcmp(name, "Array") == 0 && nodeType == 15)
        {
            // Work Done. Break out of the while loop
            break;
        }
    }
}

void processParam(xmlTextReaderPtr reader, void *data)
{
    printf("I'm Param\n");

    xmlChar *type;
    const xmlChar *name, *content; 
    int nodeType;
    int ret;
    XmlParam *xmlParamPtr = (XmlParam*)data;

    type = xmlTextReaderGetAttribute(reader, "Type");

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

        if (xmlStrcmp(name, "#text") == 0)
        {
            #ifdef __DEBUG__
            printf("content = %s\n", content);
            #endif

            // allocate memory 
            if (xmlStrcmp(type, "lstring") == 0) {

                // string like type needs special care
                xmlParamPtr->bytes = xmlStrlen(content) + 1;
                xmlParamPtr->data = malloc(xmlParamPtr->bytes);
                ((xmlChar*)xmlParamPtr->data)[xmlParamPtr->bytes - 1] = 0;

            } else {

                // other numeric type
                xmlParamPtr->bytes = get_type_size(type);
                xmlParamPtr->data = malloc(xmlParamPtr->bytes);
            }

            // save content
            sscanf(content, get_type_char(type), xmlParamPtr->data);
        }
         
        // Meet node </Param>, just break out the loop
        if (xmlStrcmp(name, "Param") == 0 && nodeType == 15)
        {
            break;
        }
    }
}

#ifdef LIBXML_READER_ENABLED

/**
 * processNode:
 * @reader: the xmlReader
 *
 * Dump information about the current node
 */
static void
processNode(xmlTextReaderPtr reader) {
    const xmlChar *name, *value;

    name = xmlTextReaderConstName(reader);
    if (name == NULL)
	name = BAD_CAST "--";

    value = xmlTextReaderGetAttribute(reader, "Name");
    
    xmlChar *tag = xmlStrncatNew(name, "-", -1);
    xmlStrcat(tag, value);

    if (xmlTextReaderNodeType(reader) == 15)
        return;

    // Could be optimized by using HashMap
    int i, ret;
    for (i = 0; i < PROCESSLEN; ++i)
    {
        ret = xmlStrcmp(tag, xnt[i].tag);
        if (ret == 0)
        {
            xnt[i].processPtr(reader, xnt[i].data);
            return;
        }
    }
}

/**
 * streamFile:
 * @filename: the file name to parse
 *
 * Parse and print information about an XML file.
 */
static void
streamFile(const char *filename) {
    xmlTextReaderPtr reader;
    int ret;

    reader = xmlReaderForFile(filename, NULL, 0);
    if (reader != NULL) {
        ret = xmlTextReaderRead(reader);
        while (ret == 1) {
            processNode(reader);
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

    streamFile(argv[1]);

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

    /*
     * deallocate resources
     */
    free(xarray.data);
    return(0);
}

#else
int main(void) {
    fprintf(stderr, "XInclude support not compiled in\n");
    exit(1);
}
#endif
