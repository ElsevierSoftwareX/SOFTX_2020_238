/*
 * Copyright (C) 2010 Leo Singer <leo.singer@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#include <Python.h>
#include <pygobject.h>


static GType
state_flags_get_type (void)
{
    static GType tp = 0;
    static const GFlagsValue values[] = {
		{1 << 0, "STATE_SCI", "operator set to go to science mode"},
		{1 << 1, "STATE_CON", "conlog unsets this bit for non-harmless epics changes"},
		{1 << 2, "STATE_UP" , "set by locking scripts"},
		{1 << 3, "STATE_INJ", "injections unset this bit"},
		{1 << 4, "STATE_EXC", "unauthorized excitations cause this bit to be unset"},
        {0, NULL, NULL},
    };

    if (G_UNLIKELY (tp == 0)) {
        tp = g_flags_register_static ("GSTLALOnlineHoftSrcStateFlags", values);
    }
    return tp;
}


static GType
dq_flags_get_type (void)
{
    static GType tp = 0;
    static const GFlagsValue values[] = {
		{1 << 0, "DQ_SCIENCE", "SV_SCIENCE & LIGHT"},
		{1 << 1, "DQ_INJECTION", "Injection: same as statevector"},
		{1 << 2, "DQ_UP", "SV_UP & LIGHT"},
		{1 << 3, "DQ_CALIBRATED", "SV_UP & LIGHT & (not TRANSIENT)"},
		{1 << 4, "DQ_BADGAMMA", "calibration is bad (outside 0.8 < gamma < 1.2)"},
		{1 << 5, "DQ_LIGHT", "Light in the arms ok"},
		{1 << 6, "DQ_MISSING", "Indication that data was dropped in DMT (currently not implemented)"},
        {0, NULL, NULL},
    };

    if (G_UNLIKELY (tp == 0)) {
        tp = g_flags_register_static ("GSTLALOnlineHoftSrcDataQualityFlags", values);
    }
    return tp;
}


void init_onlinehoftsrc(void)
{
	pygobject_init(-1, -1, -1);
	PyObject* mod = Py_InitModule("_onlinehoftsrc", NULL);
	pyg_flags_add(mod, "StateFlags", "", state_flags_get_type());
	pyg_flags_add(mod, "DQFlags", "", dq_flags_get_type());
}
