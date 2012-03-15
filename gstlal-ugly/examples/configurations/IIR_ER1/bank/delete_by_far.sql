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
		(coinc_event.time_slide_id || ";" || coinc_event.instruments) AS category,
		(coinc_inspiral.end_time - (SELECT MIN(end_time) FROM coinc_inspiral)) + 1e-9 * coinc_inspiral.end_time_ns AS end_time,
		coinc_inspiral.false_alarm_rate AS false_alarm_rate
	FROM
		coinc_event
		JOIN coinc_inspiral ON (
			coinc_inspiral.coinc_event_id == coinc_event.coinc_event_id
		);
CREATE INDEX tmpindex1 ON _cluster_info_ (coinc_event_id);
CREATE INDEX tmpindex2 ON _cluster_info_ (category, end_time, false_alarm_rate);

--
-- FIXME a no op right now, do we need to do this?
-- delete coincs that are within 10 s of coincs with a certain FAP in the same
-- category
--

DELETE FROM
	coinc_event
WHERE
	EXISTS (
		SELECT
			*
		FROM
			_cluster_info_
		WHERE
			_cluster_info_.false_alarm_rate >= 1.0
		AND
			_cluster_info_.coinc_event_id == coinc_event.coinc_event_id
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
