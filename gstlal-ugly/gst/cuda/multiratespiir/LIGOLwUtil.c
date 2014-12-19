#include "LIGOLwHeader.h"

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
