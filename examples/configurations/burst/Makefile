injections: 
	lalapps_binj  --gps-end-time=10000 --gps-start-time=100  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj0.xml --population="all_sky_sinegaussian" --time-step=30 --q=5
	lalapps_binj  --gps-end-time=10000 --gps-start-time=105  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj1.xml --population="all_sky_sinegaussian" --time-step=30 --q=6
	lalapps_binj  --gps-end-time=10000 --gps-start-time=110  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj2.xml --population="all_sky_sinegaussian" --time-step=30 --q=7
	lalapps_binj  --gps-end-time=10000 --gps-start-time=115  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj3.xml --population="all_sky_sinegaussian" --time-step=30 --q=8
	lalapps_binj  --gps-end-time=10000 --gps-start-time=120  --max-frequency=1000 --min-frequency=40 --max-hrss=5e-23 --min-hrss=1e-23 --output=inj4.xml --population="all_sky_sinegaussian" --time-step=30 --q=9
	ligolw_add inj0.xml inj1.xml inj2.xml inj3.xml inj4.xml -o sine_gauss_test.xml
	rm inj0.xml inj1.xml inj2.xml inj3.xml inj4.xml

clean:
	rm -f sine_gauss_test.xml
