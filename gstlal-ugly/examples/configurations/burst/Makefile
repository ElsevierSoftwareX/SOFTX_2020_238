injections: 
	lalapps_binj  --gps-end-time=10000 --gps-start-time=100  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj0.xml --population="all_sky_sinegaussian" --time-step=30 --q=5
	lalapps_binj  --gps-end-time=10000 --gps-start-time=105  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj1.xml --population="all_sky_sinegaussian" --time-step=30 --q=6
	lalapps_binj  --gps-end-time=10000 --gps-start-time=110  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj2.xml --population="all_sky_sinegaussian" --time-step=30 --q=7
	lalapps_binj  --gps-end-time=10000 --gps-start-time=115  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj3.xml --population="all_sky_sinegaussian" --time-step=30 --q=8
	lalapps_binj  --gps-end-time=10000 --gps-start-time=120  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj4.xml --population="all_sky_sinegaussian" --time-step=30 --q=9
	lalapps_inspinj --i-distr uniform --max-mass1 99.0 --max-mass2 99.0 --m-distr componentMass --disable-spin  --min-mtotal 25.0 --gps-start-time 951868815 --user-tag EOBNR_THREE_INJ --max-mtotal 100.0 --f-lower 10 --time-step 64.78 --min-mass2 1.0 --taper-injection start --l-distr random --min-mass1 1.0 --min-distance 1000 --waveform EOBNRpseudoFourPN --gps-end-time 953078415 --seed 103 --max-distance 750000 --d-distr log10 --output inj5.xml
	./fixinsp.py inj5.xml
	ligolw_add --ilwdchar-compat inj0.xml inj1.xml inj2.xml inj3.xml inj4.xml inj5.xml -o injections.xml
	rm inj0.xml inj1.xml inj2.xml inj3.xml inj4.xml inj5.xml
clean:
	rm -f injections.xml
