#include "LIGOLwHeader.h"

void data_to_string_char (void* des, const xmlChar *xml_type, void* data, int pos)
{
	/* pos is ignored for this function */
	int index;
	index = ligoxml_get_type_index(xml_type);
        sprintf((char*)des, typeMap[index].format, (char*)(data));
}

void data_to_string_double (void* des, const xmlChar *xml_type, void* data, int pos)
{
	int index;
	index = ligoxml_get_type_index(xml_type);
        sprintf((char*)des, typeMap[index].format, *((double*)(data) + pos));
}

void data_to_string_float (void* des, const xmlChar *xml_type, void* data, int pos)
{
	int index;
	index = ligoxml_get_type_index(xml_type);
        sprintf((char*)des, typeMap[index].format, *((float*)(data) + pos));
}


void data_to_string_int (void* des, const xmlChar *xml_type, void* data, int pos)
{
	int index;
	index = ligoxml_get_type_index(xml_type);
        sprintf((char*)des, typeMap[index].format, *((int*)(data) + pos));
}

void data_to_string_long (void* des, const xmlChar *xml_type, void* data, int pos)
{
	int index;
	index = ligoxml_get_type_index(xml_type);
        sprintf((char*)des, typeMap[index].format, *((long*)(data) + pos));

}

int ligoxml_get_type_index(const xmlChar *type)
{
    int i;
    for (i = 0; i < MAPSIZE; ++i)
    {
        if (xmlStrcmp(type, typeMap[i].xml_type) == 0)
            return typeMap[i].index;
    }

    // Wrong Type
    return -1;
}


size_t ligoxml_get_type_size(const xmlChar *type)
{
    int i;
    for (i = 0; i < MAPSIZE; ++i)
    {
        if (xmlStrcmp(type, typeMap[i].xml_type) == 0)
            return typeMap[i].bytes;
    }

    // Wrong Type
    return -1;
}

const char* ligoxml_get_type_format(const xmlChar *type)
{
    int i;
    for (i = 0; i < MAPSIZE; ++i)
    {
        if (xmlStrcmp(type, typeMap[i].xml_type) == 0)
            return typeMap[i].format;
    }

    // Wrong Type
    return "";
}

void ligoxml_init_XmlTable(XmlTable *table)
{
    // GArray of GStrings
    table->names = g_array_sized_new(FALSE, FALSE, sizeof(GString), 128);
    // HashTable (key: GString, val: XmlHashVal)
    table->hashContent = g_hash_table_new((GHashFunc)g_string_hash, (GEqualFunc)g_string_equal);
}
