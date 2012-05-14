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
