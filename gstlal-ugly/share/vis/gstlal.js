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

// Give these wrappers global scope
var latency_status_by_nodes_wrapper;
var latency_history_wrapper;
var latency_gauge_wrapper;

var snr_status_by_nodes_wrapper;
var snr_history_wrapper;

var horizon_wrapper;
var psd_wrapper;
var range_gauge_wrapper;

var noise_wrapper;
var noise_gauge_wrapper;


/*
 * Charts about latency
 */


function drawLatencyStatusByNodes() {
	// Draw a column chart
	latency_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:0',
		query: 'select latency_history where status by node',
		refreshInterval: 5,
		options: {title: 'Latency', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] },
			  hAxis: {slantedText: true, slantedTextAngle: 90},
			  chartArea: {left:25,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
		},
		containerId: 'latency_status_by_nodes_wrapper',
	});
	latency_status_by_nodes_wrapper.draw();
}


function drawLatencyHistory() {
	// Draw a column chart
	latency_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1',
		query: 'select latency_history where node is all',
		refreshInterval: 5,
		options: {title: 'Latency', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] },
			  chartArea: {left:25,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
		},
		containerId: 'latency_history_wrapper',
	});
	latency_history_wrapper.draw();
}


function drawLatencyGauge() {
	latency_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:2',
	query: 'select latency_history where now',
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
	containerId: 'latency_gauge_wrapper',
	});
	latency_gauge_wrapper.draw();
}


/*
 * Charts about SNR
 * NOTE these start numbering reqId at 100
 */


function drawSNRStatusByNodes() {
	snr_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:100',
		query: 'select snr_history where status by node',
		refreshInterval: 5,
		options: {title: 'SNR', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] },
			  hAxis: {slantedText: true, slantedTextAngle: 90},
			  chartArea: {left:30,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
		},
		containerId: 'snr_status_by_nodes_wrapper',
	});
	snr_status_by_nodes_wrapper.draw();
}


function drawSNRHistory() {
	snr_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:101',
		query: 'select snr_history where node is all',
		refreshInterval: 5,
		options: {title: 'SNR', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] },
			  chartArea: {left:30,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
		},
		containerId: 'snr_history_wrapper',
	});
	snr_history_wrapper.draw();
}


function drawHorizon() {
	horizon_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:102',
	query: 'select horizon_history',
	refreshInterval: 5,
	options: {	title: 'Horizon', 
			animation: {duration: 1000, easing: 'inAndOut'},
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
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:200',
	query: 'select noise',
	refreshInterval: 5,
	options: {	title: 'Noise', 
			animation: {duration: 1000, easing: 'inAndOut'},
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
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:300',
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
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:400',
	query: 'select noise where now',
	refreshInterval: 5,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 50, redTo: 100,
		yellowFrom: 10, yellowTo: 50,
		greenFrom: 0, greenTo: 10,
		minorTicks: 5,
		max: 100,
		min: 0
		},
	containerId: 'noise_gauge',
	});

	noise_gauge_wrapper.draw();
}


function drawRangeGauge() {
	range_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:500',
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
