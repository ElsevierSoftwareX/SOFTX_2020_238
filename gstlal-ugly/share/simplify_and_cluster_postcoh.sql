-- Copyright (C) 2011--2012,2014,2015  Kipp Cannon, Chad Hanna
-- Copyright (C) 2017 Qi Chu
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

--
-- process clean up.  same computer, unix process ID, and start time =
-- same process
--

SELECT
	"process table cleanup"
FROM
	postcoh;

CREATE INDEX tmpindex ON process (node, unix_procid, start_time);
CREATE TEMPORARY TABLE _idmap_ AS
	SELECT
		old.process_id AS old,
		MIN(new.process_id) AS new
	FROM
		process AS old
		JOIN process AS new ON (
			new.node == old.node
			AND new.unix_procid == old.unix_procid
			AND new.start_time == old.start_time
		)
	GROUP BY
		old.process_id;
DROP INDEX tmpindex;
CREATE INDEX tmpindex ON _idmap_ (old);

UPDATE segment SET process_id = (SELECT new FROM _idmap_ WHERE old == process_id);
UPDATE segment_definer SET process_id = (SELECT new FROM _idmap_ WHERE old == process_id);
UPDATE segment_summary SET process_id = (SELECT new FROM _idmap_ WHERE old == process_id);

DELETE FROM process WHERE process_id IN (SELECT old FROM _idmap_ WHERE old != new);
DELETE FROM process_params WHERE process_id NOT IN (SELECT process_id FROM process);

DROP INDEX tmpindex;
DROP TABLE _idmap_;

--
-- segment_definer clean up.  NOTE:  this assumes no meaningful information
-- is stored in the version and comment columns, and will scramble it if
-- there is.  that is, two segment_definer rows are considered to be
-- equivalent even if their version and comment values differ, and the one
-- with the higher ID will be discarded.
--

SELECT
	"segment_definer table cleanup"
FROM
	postcoh;

CREATE TEMPORARY TABLE _idmap_ AS
	SELECT
		old.segment_def_id AS old,
		MIN(new.segment_def_id) AS new
	FROM
		segment_definer AS old
		JOIN segment_definer AS new ON (
			new.ifos == old.ifos
			AND new.name == old.name
		)
	GROUP BY
		old.segment_def_id;
CREATE INDEX tmpindex ON _idmap_ (old);

UPDATE segment_summary SET segment_def_id = (SELECT new FROM _idmap_ WHERE old == segment_def_id);
UPDATE segment SET segment_def_id = (SELECT new FROM _idmap_ WHERE old == segment_def_id);
DELETE FROM segment_definer WHERE segment_def_id IN (SELECT old FROM _idmap_ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE _idmap_;

--
-- segment clean up.  NOTE:  this assumes that nothing references segment
-- rows by ID, so that redundant rows can be deleted without correcting
-- references to their IDs in other tables
--

SELECT
	"segment table cleanup"
FROM
	postcoh;


DELETE FROM
	segment
WHERE
	EXISTS (
		SELECT
			*
		FROM
			segment AS other
		WHERE
			other.segment_def_id == segment.segment_def_id
			AND other.start_time == segment.start_time
			AND other.start_time_ns == segment.start_time_ns
			AND other.end_time == segment.end_time
			AND other.end_time_ns == segment.end_time_ns
			AND other.segment_id < segment.segment_id
	);

--
-- time_slide clean up
--

--
-- begin clustering
--


SELECT
	"Number of coincs before clustering: " || count(*)
FROM
	postcoh;



--
-- create a look-up table of info required for clustering
--

CREATE TEMPORARY TABLE _cluster_info_ AS
	SELECT
		(postcoh.end_time - (SELECT MIN(end_time) FROM postcoh)) + 1e-9 * postcoh.end_time_ns AS myend_time,
		postcoh.end_time as end_time,
		postcoh.end_time_ns as end_time_ns,
		postcoh.pivotal_ifo as pivotal_ifo,
		postcoh.far AS ranking_stat,
		postcoh.cohsnr AS cohsnr
	FROM
		postcoh;

CREATE INDEX tmpindex1 ON _cluster_info_ (end_time, end_time_ns);
CREATE INDEX tmpindex2 ON _cluster_info_ (myend_time, ranking_stat);

--
-- delete postcoh that are within 4 s of postcohs with higher ranking
-- statistic in the same category.  break ties by root-sum-square of SNRs,
-- FIXME: should use postcoh.id to replace the conditions of where
--

DELETE FROM
	postcoh
WHERE
	EXISTS (
		SELECT
			*
		FROM
			_cluster_info_ AS _cluster_info_a_
			JOIN _cluster_info_ AS _cluster_info_b_ ON (
				(_cluster_info_b_.myend_time BETWEEN _cluster_info_a_.myend_time - 1.0 AND _cluster_info_a_.myend_time + 1.0)
				AND (_cluster_info_b_.ranking_stat < _cluster_info_a_.ranking_stat OR
					_cluster_info_b_.ranking_stat == _cluster_info_a_.ranking_stat AND (_cluster_info_b_.cohsnr >  _cluster_info_a_.cohsnr))
			)
		WHERE 
			(_cluster_info_a_.end_time == postcoh.end_time AND _cluster_info_a_.end_time_ns == postcoh.end_time_ns AND _cluster_info_a_.pivotal_ifo == postcoh.pivotal_ifo AND _cluster_info_a_.ranking_stat == postcoh.far)
	);
DROP INDEX tmpindex1;
DROP INDEX tmpindex2;
DROP TABLE _cluster_info_;

-- 
-- SELECT
-- 	"Number of coincs after clustering: " || count(*)
-- FROM
-- 	postcoh;

--
-- shrink the file
--

VACUUM;
