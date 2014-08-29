-- gstlal_inspiral_calc_likelihood, by default, will not re-process a file it
-- has already processed.  This behaviour allows a sort of "check-pointing"
-- feature where by if a job is evicted from the cluster and started up again,
-- it will skip through files it has already processed and continue with the
-- ones that it hadn't yet gotten to.  The --force command line option can be
-- used to force it to reprocess files, but this disables the check-pointing
-- effect and is not suitable when doing a bulk re-analysis of
-- previously-processed files.  In that case, use this script to erase the
-- record of gstlal_inspiral_calc_likelihood from the file's metadata,
-- tricking it into believing the file has not been processed yet.

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
