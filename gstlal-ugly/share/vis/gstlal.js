/*
 * Tab opening function
 */

function openGstlalTab(evt, cityName) {
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the link that opened the tab
    document.getElementById(cityName).style.display = "block";
    evt.currentTarget.className += " active";
}


/*
 * Google charts
 */


var latency_wrapper;
var snr_wrapper;
var horizon_wrapper;
var noise_wrapper;
var psd_wrapper;
var noise_gauge_wrapper;
var range_gauge_wrapper;

function drawLatency() {
// Draw a column chart
latency_wrapper = new google.visualization.ChartWrapper({
	chartType: 'CandlestickChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=0',
	query: 'select latency_history',
	refreshInterval: 5,
	options: {	title: 'Latency', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] },
			hAxis: {slantedText: true, slantedTextAngle: 90},
			chartArea: {left:25,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
	},
	containerId: 'visualization',
});
latency_wrapper.draw();
}


function drawSNR() {
	snr_wrapper = new google.visualization.ChartWrapper({
	chartType: 'CandlestickChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=1',
	query: 'select snr_history',
	refreshInterval: 5,
	options: {	title: 'SNR', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] },
			hAxis: {slantedText: true, slantedTextAngle: 90},
			chartArea: {left:30,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
	},
	containerId: 'snr',
});
snr_wrapper.draw();
}


function drawHorizon() {
	horizon_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=2',
	query: 'select horizon_history',
	refreshInterval: 5,
	options: {	title: 'Horizon', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64,128] },
			chartArea: {left:30,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}}
	},
	containerId: 'horizon',
});
horizon_wrapper.draw();
}


function drawNoise() {
	noise_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=3',
	query: 'select noise',
	refreshInterval: 5,
	options: {	title: 'Noise', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:-0.1, max:2}},
			chartArea: {left:30,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}}
	},
	containerId: 'noise',
});
noise_wrapper.draw();
}


function drawPSD() {
	psd_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=4',
	query: 'select psd',
	refreshInterval: 5,
	options: {	title: 'Amplitude Spectral Density', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{max:1e-18, min:1e-24}, format: 'scientific'},
			hAxis: {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:10, max:2048}},
			chartArea: {left:50,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}}
	},
	containerId: 'psd',
});
psd_wrapper.draw();
}

function drawNoiseGauge() {
	noise_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=5',
	query: 'select noise where now',
	refreshInterval: 5,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 1, redTo: 2,
		yellowFrom: 0.3, yellowTo: 1,
		greenFrom: -0.1, greenTo: 0.3,
		minorTicks: 5,
		max: 2,
		min: -0.2
		},
	containerId: 'noise_gauge',
	});

	noise_gauge_wrapper.draw();
}

function drawRangeGauge() {
	range_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?reqId=6',
	query: 'select horizon_history where now',
	refreshInterval: 5,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 0, redTo: 20,
		yellowFrom: 20, yellowTo: 50,
		greenFrom: 50, greenTo: 100,
		minorTicks: 5,
		max: 100,
		min: 0
		},
	containerId: 'range_gauge',
	});

	range_gauge_wrapper.draw();
}
