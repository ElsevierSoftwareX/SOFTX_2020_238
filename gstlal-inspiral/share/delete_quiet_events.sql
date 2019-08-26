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
-- begin deleting quiet events
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
		coinc_inspiral.end_time AS end_time,	--- only keep the integer part
		coinc_event.likelihood AS ranking_stat,
		coinc_inspiral.combined_far as far,
		coinc_inspiral.snr AS snr
	FROM
		coinc_event
		JOIN coinc_inspiral ON (
			coinc_inspiral.coinc_event_id == coinc_event.coinc_event_id
		);
CREATE INDEX tmpindex1 ON _cluster_info_ (coinc_event_id);
CREATE INDEX tmpindex2 ON _cluster_info_ (category, end_time, ranking_stat);

--
-- delete all events with combined far < 1e-4 or NULL
--
DELETE FROM
	coinc_event
WHERE
	EXISTS (
		SELECT
			*
		FROM
			_cluster_info_ AS _cluster_info_a_
		WHERE
			_cluster_info_a_.coinc_event_id == coinc_event.coinc_event_id AND (_cluster_info_a_.far > 1e-3 OR _cluster_info_a_.far IS NULL)
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
-- process clean up.
--


CREATE INDEX tmpindex ON process (process_id);
CREATE INDEX tmpindex2 ON process_params (process_id);
DELETE FROM process WHERE process_id NOT IN (SELECT process_id FROM coinc_event);
DELETE FROM process_params WHERE process_id NOT IN (SELECT process_id FROM process);
DROP INDEX tmpindex;
DROP INDEX tmpindex2;

--
-- shrink the file
--

VACUUM;
