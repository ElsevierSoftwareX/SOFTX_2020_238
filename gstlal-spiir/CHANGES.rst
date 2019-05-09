#########
Changelog
#########

spiir-O3-v2 (2018-05-09)
==================

- Apply a simple cut for triggers before uploading them to GraceDB. Check if the FAR is less than 1e-6Hz and coherent SNR is less than 8. If so, do not upload.

- Update the configuration file for planned changes to use CLEAN channels.

spiir-O3 (2018-04-03)
==================

- Fix the coa_phases of SPIIR triggers for Bayestar skymaps.

spiir-ER14-v3 (2018-03-31)
==================

- Pipeline configuration: Added ``gen_generic.sh`` for an example generic configuration file.

- Fixed a bug in the SPIIR skymap generation input reading in the ``gstlal_inspiral_postcohspiir_lvalert_plotter`` file.

- Added Patrick's lvshm patch to deal with cut-off streaming data. Tested during ER13.

- Fix a bug to prevent single FAR overflow.

- Hard-code the eff_distance value to be nan in the ``postcoh_finalsink.py`` until it is solved.

spiir-ER14-v2 (2018-03-19)
==================

- Fix nan values for coa_phase of SPIIR triggers.

- Remove redundant skymap uploading, use only lvalert to receive trigger information and generate SPIIR skymaps.

- Pipeline configuration: Added the missing input for SPIIR skymap uploading in the configuration files.

spiir-ER14 (2018-12-08 preliminary sign-off)
==================
