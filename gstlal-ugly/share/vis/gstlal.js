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

function updateClock ( )
 	{
 	var currentTime = new Date ( );
  	var currentHours = currentTime.getHours ( );
  	var currentMinutes = currentTime.getMinutes ( );
  	var currentSeconds = currentTime.getSeconds ( );

  	// Pad the minutes and seconds with leading zeros, if required
  	currentMinutes = ( currentMinutes < 10 ? "0" : "" ) + currentMinutes;
  	currentSeconds = ( currentSeconds < 10 ? "0" : "" ) + currentSeconds;

  	// Choose either "AM" or "PM" as appropriate
  	var timeOfDay = ( currentHours < 12 ) ? "AM" : "PM";

  	// Convert the hours component to 12-hour format if needed
  	currentHours = ( currentHours > 12 ) ? currentHours - 12 : currentHours;

  	// Convert an hours component of "0" to "12"
  	currentHours = ( currentHours == 0 ) ? 12 : currentHours;

  	// Compose the string for display
  	var currentTimeString = currentHours + ":" + currentMinutes + ":" + currentSeconds + " " + timeOfDay;
  
	var timeInMs = "" + Math.floor((Date.now() - 315964800000 + 18000)/1000.) + "&nbsp;" + currentTimeString;
  	$("#clock").html(timeInMs);
   	//$("#clock").html(currentTimeString);
   	  	
 }

$(document).ready(function()
{
   setInterval('updateClock()', 1000);
});

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


function drawLatencyStatusByNodes(gps, duration) {
	latency_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:0',
		query: 'select latency_history where status by node',
		refreshInterval: 30,
		options: {title: 'Latency', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] },
			  hAxis: {slantedText: true, slantedTextAngle: 90},
			  chartArea: {left:25,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
			  legend: {position: "in"},
		},
		containerId: 'latency_status_by_nodes_wrapper',
	});
	latency_status_by_nodes_wrapper.draw();
}


function drawLatencyHistory(gps, duration) {
	latency_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1',
		query: 'select latency_history where node is all',
		refreshInterval: 30,
		options: {title: 'Latency', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] },
			  chartArea: {left:25,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
			  legend: {position: "in"},
		},
		containerId: 'latency_history_wrapper',
	});
	latency_history_wrapper.draw();
}


function drawLatencyGauge(gps, duration) {
	latency_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:2',
	query: 'select latency_history where now',
	refreshInterval: 30,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 60, redTo: 100,
		yellowFrom: 30, yellowTo: 60,
		greenFrom: 0, greenTo: 30,
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


function drawSNRStatusByNodes(gps, duration) {
	snr_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:100',
		query: 'select snr_history where status by node',
		refreshInterval: 30,
		options: {title: 'SNR', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] },
			  hAxis: {slantedText: true, slantedTextAngle: 90},
			  chartArea: {left:30,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
			  legend: {position: "in"},
		},
		containerId: 'snr_status_by_nodes_wrapper',
	});
	snr_status_by_nodes_wrapper.draw();
}


function drawSNRHistory(gps, duration) {
	snr_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:101',
		query: 'select snr_history where node is all',
		refreshInterval: 30,
		options: {title: 'SNR', 
			  animation: {duration: 1000, easing: 'inAndOut'},
			  vAxis: {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] },
			  chartArea: {left:30,top:5,width:'100%',height:'73%'},
			  titlePosition: 'in',
			  legend: {position: "in"},
		},
		containerId: 'snr_history_wrapper',
	});
	snr_history_wrapper.draw();
}

/*
 * Charts about sensitivity
 */

function drawHorizon(gps, duration) {
	horizon_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:200',
	query: 'select horizon_history',
	refreshInterval: 5,
	options: {	title: 'Horizon', 
			animation: {duration: 1000, easing: 'inAndOut'},
			vAxis: {minValue:0, maxValue:150, textPosition: 'out', ticks: [10,30,50,70,90,110] },
			chartArea: {left:30,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}},
			legend: {position: "in"},
	},
	containerId: 'horizon',
});
horizon_wrapper.draw();
}


function drawPSD(gps, duration) {
	psd_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:201',
	query: 'select psd where now',
	refreshInterval: 10,
	options: {	title: 'Amplitude Spectral Density', 
			animation: {duration: 1000, easing: 'inAndOut', startup: true},
			vAxis: {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{max:1e-18, min:1e-24}, format: 'scientific'},
			hAxis: {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:10, max:2048}},
			chartArea: {left:50,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}},
			legend: {position: "in"},
	},
	containerId: 'psd',
});
psd_wrapper.draw();
}


function drawRangeGauge(gps, duration) {
	range_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:201',
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


/*
 * Charts about noise
 */


function drawNoise(gps, duration) {
	noise_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:300',
	query: 'select noise',
	refreshInterval: 5,
	options: {	title: 'Noise', 
			animation: {duration: 1000, easing: 'inAndOut'},
			vAxis: {textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:0, max:100}},
			chartArea: {left:30,top:5,width:'100%',height:'73%'},
			titlePosition: 'in',
			series: {0: {color: "red"}, 1: {color:"green"}},
			legend: {position: "in"},
	},
	containerId: 'noise',
});
noise_wrapper.draw();
}


function drawNoiseGauge(gps, duration) {
	noise_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:301',
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
