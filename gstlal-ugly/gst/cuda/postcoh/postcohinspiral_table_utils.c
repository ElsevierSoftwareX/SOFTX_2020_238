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
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_ns_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_ns_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_V"));
    g_array_append_val(table->names, *g_string_new("postcoh:end_time_ns_V"));
    g_array_append_val(table->names, *g_string_new("postcoh:is_background"));
    g_array_append_val(table->names, *g_string_new("postcoh:livetime"));
    g_array_append_val(table->names, *g_string_new("postcoh:ifos"));
    g_array_append_val(table->names, *g_string_new("postcoh:pivotal_ifo"));
    g_array_append_val(table->names, *g_string_new("postcoh:tmplt_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:pix_idx"));
    g_array_append_val(table->names, *g_string_new("postcoh:snglsnr_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:snglsnr_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:snglsnr_V"));
    g_array_append_val(table->names, *g_string_new("postcoh:coaphase_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:coaphase_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:coaphase_V"));
    g_array_append_val(table->names, *g_string_new("postcoh:chisq_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:chisq_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:chisq_V"));
    g_array_append_val(table->names, *g_string_new("postcoh:cohsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:nullsnr"));
    g_array_append_val(table->names, *g_string_new("postcoh:cmbchisq"));
    g_array_append_val(table->names, *g_string_new("postcoh:spearman_pval"));
    g_array_append_val(table->names, *g_string_new("postcoh:fap"));
    g_array_append_val(table->names, *g_string_new("postcoh:far"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_l"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_h"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_v"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_2h"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_1d"));
    g_array_append_val(table->names, *g_string_new("postcoh:far_1w"));
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
    g_array_append_val(table->names, *g_string_new("postcoh:deff_L"));
    g_array_append_val(table->names, *g_string_new("postcoh:deff_H"));
    g_array_append_val(table->names, *g_string_new("postcoh:deff_V"));

    table->type_names = g_array_new(FALSE, FALSE, sizeof(GString)); 
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
    g_array_append_val(table->type_names, *g_string_new("int_4s"));
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
    g_array_append_val(table->type_names, *g_string_new("real_8"));
    g_array_append_val(table->type_names, *g_string_new("real_8"));
    g_array_append_val(table->type_names, *g_string_new("real_8"));
}

void postcohinspiral_table_set_line(GString *line, PostcohInspiralTable *table, XmlTable *xtable)
{
	g_string_append_printf(line, "%d%s", table->end_time_L.gpsSeconds, xtable->delimiter->str); // for end_time_ns
	g_string_append_printf(line, "%d%s", table->end_time_L.gpsNanoSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_L.gpsSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_L.gpsNanoSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_H.gpsSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_H.gpsNanoSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_V.gpsSeconds, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->end_time_V.gpsNanoSeconds, xtable->delimiter->str);
	
	g_string_append_printf(line, "%d%s", table->is_background, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->livetime, xtable->delimiter->str);
	g_string_append_printf(line, "%s%s", table->ifos, xtable->delimiter->str);
	g_string_append_printf(line, "%s%s", table->pivotal_ifo, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->tmplt_idx, xtable->delimiter->str);
	g_string_append_printf(line, "%d%s", table->pix_idx, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->snglsnr_L, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->snglsnr_H, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->snglsnr_V, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->coaphase_L, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->coaphase_H, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->coaphase_V, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->chisq_L, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->chisq_H, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->chisq_V, xtable->delimiter->str);

	g_string_append_printf(line, "%g%s", table->cohsnr, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->nullsnr, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->cmbchisq, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spearman_pval, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->fap, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->far, xtable->delimiter->str);
	g_string_append_printf(line, "%s%s", table->skymap_fname, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->template_duration, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->mchirp, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->mtotal, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->mass1, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->mass2, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin1x, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin1y, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin1z, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin2x, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin2y, xtable->delimiter->str);
	g_string_append_printf(line, "%g%s", table->spin2z, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->ra, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->dec, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->deff_L, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->deff_H, xtable->delimiter->str);
	g_string_append_printf(line, "%lg%s", table->deff_V, xtable->delimiter->str);
	
	g_string_append(line, "\n");
	//printf("%s", line->str);

}



