/*
 * Copyright (C) 2016  Kipp Cannon, Chad Hanna
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */


/*
 * Variables with global scope
 */


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

default_options = {
	title: 'Figure', 
	hAxis: { gridlines: {color:'#FFFFFF'}},
	vAxis: {textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:0, max:100}, gridlines: {color:'#FFFFFF'}},
	chartArea: {left:40,top:10,width:'100%',height:'85%', backgroundColor:'#F0F8FF'},
	titlePosition: 'in',
	series: {0: {color: "red"}, 1: {color:"green"}},
	legend: {position: "in"},
	explorer: {actions: ['dragToZoom', 'rightClickToReset']},
	dataOpacity: "0.5",
	curveType: "function",
	fontName: "verdana",
	fontSize: 12,
	lineWidth: 2,
	backgroundColor: {stroke: '#F0F8FF', fill: '#F0F8FF', strokeWidth: '10'},
};


/*
 * Utility function
 */


function clone(obj) {
	var copy;

	// Handle the 3 simple types, and null or undefined
	if (null == obj || "object" != typeof obj) return obj;

	// Handle Date
	if (obj instanceof Date) {
		copy = new Date();
		copy.setTime(obj.getTime());
		return copy;
	}

	// Handle Array
	if (obj instanceof Array) {
		copy = [];
		for (var i = 0, len = obj.length; i < len; i++) {
			copy[i] = clone(obj[i]);
		}
		return copy;
	}

	// Handle Object
	if (obj instanceof Object) {
		copy = {};
		for (var attr in obj) {
			if (obj.hasOwnProperty(attr)) copy[attr] = clone(obj[attr]);
		}
		return copy;
	}

	throw new Error("Unable to copy obj! Its type isn't supported.");
}

function openGstlalTab(evt, tabName) {
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
	document.getElementById(tabName).style.display = "block";
	evt.currentTarget.className += " active";

	// Assume the rest are charts that need to be drawn
	for (i = 2; i < arguments.length; i++) {
		arguments[i].getChart().clearChart();
		arguments[i].draw();
        }
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
  
	var timeInMs = "" + Math.floor((Date.now() - 315964800000 + 17000)/1000.) + "&nbsp;" + currentTimeString;
  	$("#clock").html(timeInMs);
   	//$("#clock").html(currentTimeString);
   	  	
 }

$(document).ready(function()
{
   setInterval('updateClock()', 1000);
});


/*
 * Charts about latency
 */


function drawLatencyStatusByNodes(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] };
	these_options.hAxis = {slantedText: true, slantedTextAngle: 90};
	these_options.title = 'Latency';

	latency_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:0&gpstime=' + gps + '&duration=' + duration,
		query: 'select latency_history where status by node',
		refreshInterval: refresh,
		options: these_options, 
		containerId: 'latency_status_by_nodes_wrapper',
	});
	latency_status_by_nodes_wrapper.draw();
}


function drawLatencyHistory(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] };
	these_options.title = 'Latency';

	latency_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1&gpstime=' + gps + '&duration=' + duration,
		query: 'select latency_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'latency_history_wrapper',
	});
	latency_history_wrapper.draw();
}


function drawLatencyGauge(gps, duration, refresh) {
	latency_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:2' + '&gpstime='  + gps + '&duration=' + duration,
	query: 'select latency_history where now',
	refreshInterval: refresh,
        options: {
		//animation: {duration: 4000, easing: 'linear'},
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


function drawSNRStatusByNodes(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.hAxis = {slantedText: true, slantedTextAngle: 90};
	these_options.title = 'SNR';

	snr_status_by_nodes_wrapper = new google.visualization.ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:100'  + '&gpstime='  + gps + '&duration=' + duration,
		query: 'select snr_history where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'snr_status_by_nodes_wrapper',
	});
	snr_status_by_nodes_wrapper.draw();
}


function drawSNRHistory(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.title = 'SNR';

	snr_history_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:101'  + '&gpstime='  + gps + '&duration=' + duration,
		query: 'select snr_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'snr_history_wrapper',
	});
	snr_history_wrapper.draw();
}

/*
 * Charts about sensitivity
 */

function drawHorizon(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {minValue:0, maxValue:150, textPosition: 'out', ticks: [10,30,50,70,90,110] };
	these_options.title = 'Horizon';
	these_options.series = {0: {color: "red"}, 1: {color:"green"}};

	horizon_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:200'  + '&gpstime='  + gps + '&duration=' + duration,
	query: 'select horizon_history',
	refreshInterval: refresh,
	options: these_options,
	containerId: 'horizon_wrapper',
});
horizon_wrapper.draw();
}


function drawPSD(gps, duration, refresh) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{max:1e-18, min:1e-24}, format: 'scientific'};
	these_options.hAxis = {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:10, max:2048}};
	these_options.title = 'Amplitude Spectral Density';
	these_options.series = {0: {color: "red"}, 1: {color:"green"}};

	psd_wrapper = new google.visualization.ChartWrapper({
	chartType: 'LineChart',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:201'  + '&gpstime='  + gps + '&duration=' + duration,
	query: 'select psd where now',
	refreshInterval: refresh,
	options: these_options,
	containerId: 'psd_wrapper',
});
psd_wrapper.draw();
}


function drawRangeGauge(gps, duration, refresh) {
	range_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:201'  + '&gpstime='  + gps + '&duration=' + duration,
	query: 'select horizon_history where now',
	refreshInterval: refresh,
        options: {
		//animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 0, redTo: 20,
		yellowFrom: 20, yellowTo: 50,
		greenFrom: 50, greenTo: 100,
		minorTicks: 5,
		max: 100,
		min: 0
		},
	containerId: 'range_gauge_wrapper',
	});

	range_gauge_wrapper.draw();
}


/*
 * Charts about noise
 */


function drawNoise(gps, duration, refresh) {
	// Setup the custom options
	var these_options = clone(default_options);
	these_options.title = "Whitened h(t)";

	noise_wrapper = new google.visualization.ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:300'  + '&gpstime='  + gps + '&duration=' + duration,
		query: 'select noise',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'noise_wrapper',
});

noise_wrapper.draw();

}


function drawNoiseGauge(gps, duration, refresh) {
	noise_gauge_wrapper = new google.visualization.ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:301'  + '&gpstime='  + gps + '&duration=' + duration,
	query: 'select noise where now',
	refreshInterval: refresh,
        options: {
		//animation: {duration: 4000, easing: 'linear'},
		width: 1800, height: 200,
		redFrom: 50, redTo: 100,
		yellowFrom: 10, yellowTo: 50,
		greenFrom: 0, greenTo: 10,
		minorTicks: 5,
		max: 100,
		min: 0
		},
	containerId: 'noise_gauge_wrapper',
	});

	noise_gauge_wrapper.draw();
}
