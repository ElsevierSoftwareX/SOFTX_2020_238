#include "postcoh_table_utils.h"

void postcoh_table_init(XmlTable *table)
{
    table->tableName = g_string_new("postcoh:table");

    table->delimiter = g_string_new(",");
    
    table->names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->names, *g_string_new("postcoh:end_time"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_ns"));
    g_array_append_val(table->names, *g_string_new("postcoh:is_background"));
    g_array_append_val(table->names, *g_string_new("postcoh:ifos"));
    g_array_append_val(table->names, *g_string_new("postcoh:pivotal_ifo"));
    g_array_append_val(table->names, *g_string_new("postcoh:tmplt_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:pix_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:maxsnglsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:cohsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:nullsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:chisq"));
    g_array_append_val(table->names, *g_string_new("postcoh:skymap_fname"));

    table->type_names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("lstring"));
    g_array_append_val(table->type_names, *g_string_new("lstring"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("lstring"));
}


