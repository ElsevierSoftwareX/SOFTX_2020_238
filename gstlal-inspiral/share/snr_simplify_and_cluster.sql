-- Copyright (C) 2011--2012,2014,2015  Kipp Cannon, Chad Hanna
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

CREATE INDEX tmpindex ON process (node, unix_procid, start_time);
CREATE TEMPORARY TABLE ___idmap___ AS
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
CREATE INDEX tmpindex ON ___idmap___ (old);
CREATE INDEX tmpindex2 ON ___idmap___ (new, old);

UPDATE coinc_event SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);
UPDATE segment SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);
UPDATE segment_definer SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);
UPDATE segment_summary SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);
UPDATE sngl_inspiral SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);
UPDATE time_slide SET process_id = (SELECT new FROM ___idmap___ WHERE old == process_id);

DELETE FROM process WHERE process_id IN (SELECT old FROM ___idmap___ WHERE old != new);
DELETE FROM process_params WHERE process_id NOT IN (SELECT process_id FROM process);

DROP INDEX tmpindex;
DROP INDEX tmpindex2;
DROP TABLE ___idmap___;

--
-- coinc_definer clean up
--

CREATE TEMPORARY TABLE ___idmap___ AS
	SELECT
		old.coinc_def_id AS old,
		MIN(new.coinc_def_id) AS new
	FROM
		coinc_definer AS old
		JOIN coinc_definer AS new ON (
			new.search == old.search
			AND new.search_coinc_type == old.search_coinc_type
		)
	GROUP BY
		old.coinc_def_id;
CREATE INDEX tmpindex ON ___idmap___ (old);
CREATE INDEX tmpindex2 ON ___idmap___ (new, old);

UPDATE coinc_event SET coinc_def_id = (SELECT new FROM ___idmap___ WHERE old == coinc_def_id);
DELETE FROM coinc_definer WHERE coinc_def_id IN (SELECT old FROM ___idmap___ WHERE old != new);

DROP INDEX tmpindex;
DROP INDEX tmpindex2;
DROP TABLE ___idmap___;

--
-- segment_definer clean up.  NOTE:  this assumes no meaningful information
-- is stored in the version and comment columns, and will scramble it if
-- there is.  that is, two segment_definer rows are considered to be
-- equivalent even if their version and comment values differ, and the one
-- with the higher ID will be discarded.
--

CREATE TEMPORARY TABLE ___idmap___ AS
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
CREATE INDEX tmpindex ON ___idmap___ (old);

UPDATE segment_summary SET segment_def_id = (SELECT new FROM ___idmap___ WHERE old == segment_def_id);
UPDATE segment SET segment_def_id = (SELECT new FROM ___idmap___ WHERE old == segment_def_id);
DELETE FROM segment_definer WHERE segment_def_id IN (SELECT old FROM ___idmap___ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE ___idmap___;

--
-- segment clean up.  NOTE:  this assumes that nothing references segment
-- rows by ID, so that redundant rows can be deleted without correcting
-- references to their IDs in other tables
--

CREATE INDEX tmpindex ON segment(segment_def_id, start_time, start_time_ns, end_time, end_time_ns);

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

DROP INDEX tmpindex;
--
-- time_slide clean up
--

CREATE TEMPORARY TABLE ___idmap___ AS
	SELECT
		time_slide_id AS old,
		(SELECT group_concat(instrument || "=" || offset) FROM time_slide AS time_slide_a WHERE time_slide_a.time_slide_id == time_slide.time_slide_id ORDER BY instrument) AS repr,
		NULL AS new
	FROM
		time_slide
	GROUP BY
		time_slide_id;
CREATE INDEX tmpindex ON ___idmap___ (repr, old);

UPDATE ___idmap___ SET new = (SELECT MIN(old) FROM ___idmap___ AS a WHERE a.repr == ___idmap___.repr);
DROP INDEX tmpindex;
CREATE INDEX tmpindex ON ___idmap___ (old);

UPDATE coinc_event SET time_slide_id = (SELECT ___idmap___.new FROM ___idmap___ WHERE ___idmap___.old == time_slide_id);
DELETE FROM time_slide WHERE time_slide_id IN (SELECT old FROM ___idmap___ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE ___idmap___;


--
-- begin clustering
--


SELECT
	"Number of coincs before clustering: " || count(*)
FROM
	coinc_event;

--
-- create a look-up table of info required for clustering
--

CREATE INDEX tmpindex1 ON coinc_event_map (coinc_event_id);
CREATE INDEX tmpindex2 ON coinc_event (coinc_event_id);

CREATE TEMPORARY TABLE _cluster_info_ AS
	SELECT
		coinc_event.coinc_event_id AS coinc_event_id,
		coinc_event.time_slide_id AS category,
		(coinc_inspiral.end_time - (SELECT MIN(end_time) FROM coinc_inspiral)) + 1e-9 * coinc_inspiral.end_time_ns AS end_time,
		coinc_inspiral.snr AS snr,
		-- NOTE FIXME we are using bank_chisq to hold a chisq weighted snr value
		(SELECT SUM(sngl.bank_chisq * sngl.bank_chisq) FROM sngl_inspiral as sngl WHERE sngl.event_id IN (SELECT event_id FROM coinc_event_map WHERE coinc_event_map.coinc_event_id == coinc_event.coinc_event_id)) AS ranking_stat
	FROM
		coinc_event
		JOIN coinc_inspiral ON (
			coinc_inspiral.coinc_event_id == coinc_event.coinc_event_id
		);
DROP INDEX tmpindex1;
DROP INDEX tmpindex2;

CREATE INDEX tmpindex1 ON _cluster_info_ (coinc_event_id);
CREATE INDEX tmpindex2 ON _cluster_info_ (category, end_time, ranking_stat);

--
-- delete coincs that are within 1 s of coincs with higher ranking
-- statistic in the same category.  break ties by root-sum-square of SNRs,
-- break ties by coinc_event_id
--

DELETE FROM
	coinc_event
WHERE
	EXISTS (
		SELECT
			*
		FROM
			_cluster_info_ AS _cluster_info_a_
			JOIN _cluster_info_ AS _cluster_info_b_ ON (
				_cluster_info_b_.category == _cluster_info_a_.category
				AND (_cluster_info_b_.end_time BETWEEN _cluster_info_a_.end_time - 0.1 AND _cluster_info_a_.end_time + 0.1)
				AND (_cluster_info_b_.ranking_stat > _cluster_info_a_.ranking_stat OR
					_cluster_info_b_.ranking_stat == _cluster_info_a_.ranking_stat AND (_cluster_info_b_.snr >  _cluster_info_a_.snr OR _cluster_info_b_.snr == _cluster_info_a_.snr AND (_cluster_info_b_.coinc_event_id >  _cluster_info_a_.coinc_event_id))
				)
			)
		WHERE
			_cluster_info_a_.coinc_event_id == coinc_event.coinc_event_id
	);
DROP INDEX tmpindex1;
DROP INDEX tmpindex2;
DROP TABLE _cluster_info_;

SELECT
	"Number of coincs after clustering: " || count(*)
FROM
	coinc_event;

--
-- delete unused coinc_inspiral rows
--

DELETE FROM
	coinc_inspiral
WHERE
	coinc_event_id NOT IN (
		SELECT
			coinc_event_id
		FROM
			coinc_event
	);

--
-- delete unused coinc_event_map rows
--

DELETE FROM
	coinc_event_map
WHERE
	coinc_event_id NOT IN (
		SELECT
			coinc_event_id
		FROM
			coinc_event
	);

--
-- delete unused sngl_inspiral rows
--

DELETE FROM
	sngl_inspiral
WHERE
	sngl_inspiral.snr < (
		SELECT
			-- if they aren't all the same err on the size of
			-- saving disk space
			MAX(value)
		FROM
			process_params
		WHERE
			program == "gstlal_inspiral"
			AND param == "--singles-threshold"
	)
	AND event_id NOT IN (
		SELECT
			event_id
		FROM
			coinc_event_map
		WHERE
			table_name == 'sngl_inspiral'
	);

--
-- shrink the file
--

VACUUM;
