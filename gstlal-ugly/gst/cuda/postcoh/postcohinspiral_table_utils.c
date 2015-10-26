/* 
 * Copyright (C) 2014 Qi Chu <qi.chu@ligo.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "postcohinspiral_table_utils.h"

void postcohinspiral_table_init(XmlTable *table)
{
    table->tableName = g_string_new("postcoh:table");

    table->delimiter = g_string_new(",");
    
    table->names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->names, *g_string_new("postcoh:end_time"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_ns"));
    g_array_append_val(table->names, *g_string_new("postcoh:is_background"));
    g_array_append_val(table->names, *g_string_new("postcoh:livetime"));
    g_array_append_val(table->names, *g_string_new("postcoh:ifos"));
    g_array_append_val(table->names, *g_string_new("postcoh:pivotal_ifo"));
    g_array_append_val(table->names, *g_string_new("postcoh:tmplt_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:pix_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:maxsnglsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:cohsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:nullsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:chisq"));
    g_array_append_val(table->names, *g_string_new("postcoh:spearman_pval"));
    g_array_append_val(table->names, *g_string_new("postcoh:fap"));
    g_array_append_val(table->names, *g_string_new("postcoh:far"));
    g_array_append_val(table->names, *g_string_new("postcoh:skymap_fname"));
    g_array_append_val(table->names, *g_string_new("postcoh:template_duration"));
    g_array_append_val(table->names, *g_string_new("postcoh:mchirp"));
    g_array_append_val(table->names, *g_string_new("postcoh:mtotal"));
    g_array_append_val(table->names, *g_string_new("postcoh:mass1"));
    g_array_append_val(table->names, *g_string_new("postcoh:mass2"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin1x"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin1y"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin1z"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin2x"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin2y"));
    g_array_append_val(table->names, *g_string_new("postcoh:spin2z"));
    g_array_append_val(table->names, *g_string_new("postcoh:ra"));
    g_array_append_val(table->names, *g_string_new("postcoh:dec"));

    table->type_names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
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
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("lstring"));
    g_array_append_val(table->type_names, *g_string_new("real_8"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_4"));
    g_array_append_val(table->type_names, *g_string_new("real_8"));
    g_array_append_val(table->type_names, *g_string_new("real_8"));
}


