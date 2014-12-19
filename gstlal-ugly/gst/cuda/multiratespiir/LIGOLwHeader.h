#ifndef __XY_XMLHEADER__
#define __XY_XMLHEADER__

#define XMLSTRMAXLEN    1024

#include <stdio.h>
#include <libxml/xmlreader.h>
#include <glib.h>

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

typedef struct _XmlTypeMap
{
    // Type in XML
    const xmlChar* xml_type;

    // Type in C
    const xmlChar* c_type;

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
extern void ligoxml_init_XmlTable(XmlTable *table);

#define MAPSIZE 5
static const XmlTypeMap typeMap[MAPSIZE] =
{
    {BAD_CAST "lstring",	BAD_CAST "char*",	"%s",   sizeof(char),   0},
    {BAD_CAST "real_8",		BAD_CAST "double",  "%lf",  sizeof(double), 1},
    {BAD_CAST "real_4",		BAD_CAST "float",   "%f",   sizeof(float),  2},
    {BAD_CAST "int_4s",		BAD_CAST "int",		"%d",   sizeof(int),    3},
	{BAD_CAST "int_8s",		BAD_CAST "long",	"%ld",	sizeof(long),	4}
};

// get the number of bytes this type requires
extern size_t ligoxml_get_type_size(const xmlChar *type);

// get the format string of this type for printing
extern const char* ligoxml_get_type_format(const xmlChar *type);

#endif
