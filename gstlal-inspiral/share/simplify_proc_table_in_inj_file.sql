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

UPDATE sim_inspiral SET process_id = (SELECT new FROM _idmap_ WHERE old == process_id);

DELETE FROM process WHERE process_id IN (SELECT old FROM _idmap_ WHERE old != new);
DELETE FROM process_params WHERE process_id NOT IN (SELECT process_id FROM process);

DROP INDEX tmpindex;
DROP TABLE _idmap_;

VACUUM;
