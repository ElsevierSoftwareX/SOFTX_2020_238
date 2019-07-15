gstlal-modules :
	# FIXME FIXME FIXME total hack to work around unique python packaging
	#mv ../gstlal/python/__init__.py ../gstlal/python/__init__.py.bk
	sphinx-apidoc -e -o source/gstlal/python-modules ../gstlal/python ../gstlal/python/misc.py ../gstlal/python/bottle.py ../gstlal/python/coherent_null.py ../gstlal/python/matplotlibhelper.py
	#mv ../gstlal/python/__init__.py.bk ../gstlal/python/__init__.py
	sed -i "1s/python/\`\`gstlal\`\`/" source/gstlal/python-modules/modules.rst

gstlal-inspiral-modules :
	sphinx-apidoc -e -o source/gstlal-inspiral/python-modules ../gstlal-inspiral/python ../gstlal-inspiral/python/cbc_template_iir.py ../gstlal-inspiral/python/spawaveform.py ../gstlal-inspiral/python/emcee.py ../gstlal-inspiral/python/spiirparts.py ../gstlal-inspiral/python/snglinspiraltable.py
	sed -i "1s/python/\`\`gstlal-inspiral\`\`/" source/gstlal-inspiral/python-modules/modules.rst

gstlal-calibration-modules :
	sphinx-apidoc -e -o source/gstlal-calibration/python-modules ../gstlal-calibration/python
	sed -i "1s/python/\`\`gstlal-calibration\`\`/" source/gstlal-calibration/python-modules/modules.rst

gstlal-burst-modules :
	sphinx-apidoc -e -o source/gstlal-burst/python-modules ../gstlal-burst/python ../gstlal-burst/python/excesspower ../gstlal-burst/python/fxtools/sngltriggertable.py
	sed -i "1s/python/\`\`gstlal-burst\`\`/" source/gstlal-burst/python-modules/modules.rst

gstlal-ugly-modules :
	sphinx-apidoc -e -o source/gstlal-ugly/python-modules ../gstlal-ugly/python ../gstlal-ugly/python/gviz_api.py ../gstlal-ugly/python/coherent_inspiral_metric.py  ../gstlal-ugly/python/coherent_inspiral_metric_plots.py ../gstlal-ugly/python/coherent_inspiral_metric_detector_details.py ../gstlal-ugly/python/aggregator.py 
	sed -i "1s/python/\`\`gstlal-ugly\`\`/" source/gstlal-ugly/python-modules/modules.rst

gstlal-bin :
	./sphinx-bindoc source/gstlal/bin ../gstlal/bin gstlal_ligo_data_find_check
	sed -i "1s/bin/\`\`gstlal\`\`/" source/gstlal/bin/bin.rst

gstlal-inspiral-bin :
	./sphinx-bindoc source/gstlal-inspiral/bin ../gstlal-inspiral/bin gstlal_inspiral_flopulator gstlal_inspiral_lvalert_omegascan gstlal_ll_inspiral_calculate_range gstlal_ll_inspiral_daily_page_online gstlal_iir_inspiral gstlal_inspiral_iir_bank_pipe gstlal_ll_inspiral_save_state gstlal_inspiral_marginalize_likelihoods_online gstlal_inspiral_plot_extrinsic_params gstlal_ll_inspiral_get_urls gstlal_inspiral_plot_svd_bank gstlal_inspiral_lvalert_background_plotter 
	sed -i "1s/bin/\`\`gstlal-inspiral\`\`/" source/gstlal-inspiral/bin/bin.rst

gstlal-calibration-bin :
	./sphinx-bindoc source/gstlal-calibration/bin ../gstlal-calibration/bin
	sed -i "1s/bin/\`\`gstlal-calibration\`\`/" source/gstlal-calibration/bin/bin.rst

gstlal-burst-bin :
	./sphinx-bindoc source/gstlal-burst/bin ../gstlal-burst/bin gstlal_excesspower gstlal_excesspower_trigvis
	sed -i "1s/bin/\`\`gstlal-burst\`\`/" source/gstlal-burst/bin/bin.rst

gstlal-ugly-bin :
	./sphinx-bindoc source/gstlal-ugly/bin ../gstlal-ugly/bin gstlal_h1h2_coh_frames gstlal_geo_player gstlal_ll_inspiral_daily_summary_page gstlal_cache_to_segments gstlal_inspiral_plot_efficiency gstlal_burst_injections_by_rate_plot gstlal_display_from_logs gstlal_inspiral_bankviz gstlal_ll_inspiral_regen_reg_from_condor_q gstlal_cbc_sensitivity_vs_flow_at_fixed_computational_cost gstlal_inspiral_check_livetimes gstlal_rds_gen gstlal_injections_by_local_rate_plot gstlal_ninja_smooth_reference_psd gstlal_shift_frames 
	sed -i "1s/bin/\`\`gstlal-ugly\`\`/" source/gstlal-ugly/bin/bin.rst

html : gstlal-modules gstlal-inspiral-modules gstlal-calibration-modules gstlal-ugly-modules gstlal-burst-modules gstlal-bin gstlal-inspiral-bin gstlal-calibration-bin gstlal-ugly-bin gstlal-burst-bin
	sphinx-build -b html source _build

clean :
	rm -rf source/gstlal/python-modules/*
	rm -rf source/gstlal-*/python-modules/*
	rm -rf source/gstlal/bin/*
	rm -rf source/gstlal-*/bin/*
	rm -rf _build/*
	rm -rf _build/.doctrees
	rm -rf _build/.buildinfo
	rm -rf _build/.nojekyll
