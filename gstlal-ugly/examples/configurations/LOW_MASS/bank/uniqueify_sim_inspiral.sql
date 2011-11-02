--
-- sim_inspiral clean up
--

CREATE INDEX tmpindex ON sim_inspiral (end_time_gmst);
CREATE TEMPORARY TABLE _idmap_ AS
	SELECT
		old.simulation_id AS old,
		MIN(new.simulation_id) AS new
	FROM
		sim_inspiral AS old
		JOIN sim_inspiral AS new ON (
			new.end_time_gmst == old.end_time_gmst
		)
	GROUP BY
		old.simulation_id;
DROP INDEX tmpindex;
CREATE INDEX tmpindex ON _idmap_ (old);

UPDATE coinc_event_map SET event_id = (SELECT new FROM _idmap_ WHERE old == event_id) WHERE table_name == "sim_inspiral";
DELETE FROM sim_inspiral WHERE simulation_id IN (SELECT old FROM _idmap_ WHERE old != new);

DROP INDEX tmpindex;
DROP TABLE _idmap_;
