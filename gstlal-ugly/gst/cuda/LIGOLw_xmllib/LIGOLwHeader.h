#ifndef __LIGOLW_XMLHEADER__
#define __LIGOLW_XMLHEADER__

#define XMLSTRMAXLEN    1024

#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <libxml/encoding.h>
#include <libxml/xmlreader.h>
#include <libxml/xmlwriter.h>

typedef struct _XmlNodeStruct
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

} XmlNodeStruct;

typedef enum { UNKNOWN, LSTRING, REAL_4, REAL_8, INT_4S } xmlType;

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

typedef void (*DataToString) (void* des, const xmlChar *xml_type, void* data, int pos);

void data_to_string_char (void* des, const xmlChar *xml_type, void* data, int pos);
void data_to_string_double (void* des, const xmlChar *xml_type, void* data, int pos);
void data_to_string_float (void* des, const xmlChar *xml_type, void* data, int pos);
void data_to_string_int (void* des, const xmlChar *xml_type, void* data, int pos);
void data_to_string_long (void* des, const xmlChar *xml_type, void* data, int pos);

typedef struct _XmlTypeMap
{
    // Type in XML
    const xmlChar* xml_type;

    // Type in C
    //const xmlChar* c_type;
    DataToString dts_func;

    // Format String
    const char* format;

    // Size in bytes
    size_t bytes;

    // Type Number
    int index;

} XmlTypeMap;

typedef struct _DComplex
{
    double real;
    double imag;
} DComplex;

#define XMLTABLEMASKLEN 105

typedef struct _XmlTable
{
    // name of each column, of type GString
    GArray  *names;

    // table
    GHashTable *hashContent;;

    // name of the table
    GString *tableName;

    // delimiter
    GString *delimiter;

} XmlTable;

typedef struct _XmlHashVal
{
    // pointer to data, of type "$type"
    GArray*     data;

    // type
    GString*    type;

    // name
    GString*    name;

} XmlHashVal;

// must be called before manipulating XmlTable
void ligoxml_init_XmlTable(XmlTable *table);

#define MAPSIZE 5
static const XmlTypeMap typeMap[MAPSIZE] =
{
    {BAD_CAST "lstring",	data_to_string_char,	"%s",   sizeof(char),   0},
    {BAD_CAST "real_8",		data_to_string_double,    "%lf",  sizeof(double), 1},
    {BAD_CAST "real_4",		data_to_string_float,     "%f",   sizeof(float),  2},
    {BAD_CAST "int_4s",		data_to_string_int,	"%d",   sizeof(int),    3},
    {BAD_CAST "int_8s",		data_to_string_long,	"%ld",	sizeof(long),	4}
};
int ligoxml_get_type_index(const xmlChar *type);

// get the number of bytes this type requires
size_t ligoxml_get_type_size(const xmlChar *type);

// get the format string of this type for printing
const char* ligoxml_get_type_format(const xmlChar *type);

void readArray(xmlTextReaderPtr reader, void *data);

void freeArray(XmlArray *array);

void readParam(xmlTextReaderPtr reader, void *data);

void freeParam(XmlParam *param);

void readTable(xmlTextReaderPtr reader, void *data);

void freeTable(XmlTable *table);

void
processNode(xmlTextReaderPtr reader, XmlNodeStruct *xns, int len);

void
parseFile(const char *filename, XmlNodeStruct *xns, int len);

void testXmlwriterFilename(const char *uri);
xmlChar *ConvertInput(const char *in, const char *encoding);

int ligoxml_write_Param(xmlTextWriterPtr writer, XmlParam *xparamPtr, const xmlChar* xml_type,
                        const xmlChar* Name);

int ligoxml_write_Array(xmlTextWriterPtr writer, XmlArray *xarrayPtr, const xmlChar* xml_type, 
                        const xmlChar* delimiter, const xmlChar* Name);

int ligoxml_write_Table(xmlTextWriterPtr writer, const XmlTable *xtablePtr);

#endif
