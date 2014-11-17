-- Copyright (C) 2011--2012,2014  Kipp Cannon, Chad Hanna, Melissa Frei
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


DROP table if EXISTS sim_inspiral;

--
-- coinc_definer clean up
--

CREATE TEMPORARY TABLE _idmap_ AS
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
CREATE INDEX tmpindex ON _idmap_ (old);

UPDATE coinc_event SET coinc_def_id = (SELECT new FROM _idmap_ WHERE old == coinc_def_id);
DELETE FROM coinc_definer WHERE coinc_def_id IN (SELECT old FROM _idmap_ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE _idmap_;

--
-- segment_definer clean up.  NOTE;  this assumes no meaningful information
-- is stored in the version and comment columns, and will scramble it if
-- there is.  that is, two segment_definer rows are considered to be
-- equivalent even if their version and comment values differ, and the one
-- with the higher ID will be discarded.
--

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
-- time_slide clean up
--

CREATE TEMPORARY TABLE _idmap_ AS
	SELECT
		time_slide_id AS old,
		(SELECT group_concat(instrument || "=" || offset) FROM time_slide AS time_slide_a WHERE time_slide_a.time_slide_id == time_slide.time_slide_id ORDER BY instrument) AS repr,
		NULL AS new
	FROM
		time_slide
	GROUP BY
		time_slide_id;
CREATE INDEX tmpindex ON _idmap_ (repr, old);

UPDATE _idmap_ SET new = (SELECT MIN(old) FROM _idmap_ AS a WHERE a.repr == _idmap_.repr);
DROP INDEX tmpindex;
CREATE INDEX tmpindex ON _idmap_ (old);

UPDATE coinc_event SET time_slide_id = (SELECT _idmap_.new FROM _idmap_ WHERE _idmap_.old == time_slide_id);
DELETE FROM time_slide WHERE time_slide_id IN (SELECT old FROM _idmap_ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE _idmap_;


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

CREATE TEMPORARY TABLE _cluster_info_ AS
	SELECT
		coinc_event.coinc_event_id AS coinc_event_id,
		coinc_event.time_slide_id AS category,
		(coinc_inspiral.end_time - (SELECT MIN(end_time) FROM coinc_inspiral)) + 1e-9 * coinc_inspiral.end_time_ns AS end_time,
		coinc_event.likelihood AS ranking_stat
	FROM
		coinc_event
		JOIN coinc_inspiral ON (
			coinc_inspiral.coinc_event_id == coinc_event.coinc_event_id
		);
CREATE INDEX tmpindex1 ON _cluster_info_ (coinc_event_id);
CREATE INDEX tmpindex2 ON _cluster_info_ (category, end_time, ranking_stat);

--
-- delete coincs that are within 4 s of coincs with higher SNR in the same
-- category
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
				AND (_cluster_info_b_.end_time BETWEEN _cluster_info_a_.end_time - 4.0 AND _cluster_info_a_.end_time + 4.0)
				AND _cluster_info_b_.ranking_stat > _cluster_info_a_.ranking_stat
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
	event_id NOT IN (
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
