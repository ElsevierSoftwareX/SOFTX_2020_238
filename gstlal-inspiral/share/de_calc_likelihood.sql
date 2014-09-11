-- Copyright (C) 2014  Kipp Cannon
--
-- This program is free software; you can redistribute it and/or modify it
-- under the terms of the GNU General Public License as published by the
-- Free Software Foundation; either version 2 of the License, or (at your
-- option) any later version.
--
-- This program is distributed in the hope that it will be useful, but
-- WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
-- Public License for more details.
--
-- You should have received a copy of the GNU General Public License along
-- with this program; if not, write to the Free Software Foundation, Inc.,
-- 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


-- gstlal_inspiral_calc_likelihood, by default, will not re-process a file it
-- has already processed.  This behaviour allows a sort of "check-pointing"
-- feature where by if a job is evicted from the cluster and started up again,
-- it will skip through files it has already processed and continue with the
-- ones that it hadn't yet gotten to.  The --force command line option can be
-- used to force it to reprocess files, but this disables the check-pointing
-- effect and is not suitable when doing a bulk re-analysis of
-- previously-processed files.  In that case, use this script to erase the
-- record of gstlal_inspiral_calc_likelihood from the file's metadata,
-- tricking it into believing the file has not been processed yet.  For
-- safety, this script also deletes the likleihood ratios from the
-- coinc_event table.


DELETE FROM
	process_params
WHERE
	process_id IN (
	SELECT
		process_id
	FROM
		process
	WHERE
		program == "gstlal_inspiral_calc_likelihood"
	);

DELETE FROM
	search_summary
WHERE
	process_id IN (
	SELECT
		process_id
	FROM
		process
	WHERE
		program == "gstlal_inspiral_calc_likelihood"
	);

DELETE FROM
	process
WHERE
	program == "gstlal_inspiral_calc_likelihood";

UPDATE
	coinc_event
SET
	likelihood = NULL;
