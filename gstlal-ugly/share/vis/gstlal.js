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
var likelihood_status_by_nodes_wrapper;
var likelihood_history_wrapper;
var far_status_by_nodes_wrapper;
var far_history_wrapper;

var horizon_wrapper;
var horizon_table_wrapper;
var psd_wrapper;
var range_gauge_wrapper;

var noise_wrapper;
var noise_table_wrapper;
var noise_gauge_wrapper;

var up_time_wrapper;
var uptime_gauge_wrapper;
var dropped_wrapper;
var ram_status_wrapper;
var time_since_last_wrapper;
var time_since_trigger_wrapper;

var charts = [];

var H1="#c0392b";
var L1="#16a085";

default_options = {
	title: 'Figure', 
	hAxis: { gridlines: {color:'#FFFFFF'}},
	vAxis: {textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:0, max:100}, gridlines: {color:'#ecf0f1'}},
	chartArea: {left:50, top:15, width:'95%', height:'75%', backgroundColor:'#F0F8FF'},
	titlePosition: 'in',
	series: {0: {color: H1}, 1: {color:L1}},
	legend: {position: "in"},
	explorer: {actions: ['dragToZoom', 'rightClickToReset']},
	dataOpacity: "0.5",
	curveType: "none",
	fontName: "verdana",
	fontSize: 12,
	lineWidth: 2,
	backgroundColor: {stroke: '#7f8c8d', fill: '#ecf0f1', strokeWidth: '10'},
	width: "90%",
	bar: {
	    groupWidth: '70%',
	}
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

	// Disable automatic queries
	for (i = 0; i < charts.length; i++) {
		charts[i].setRefreshInterval(0);
	}

	// Get all elements with class="tablinks" and remove the class "active"
	tablinks = document.getElementsByClassName("tablinks");
	for (i = 0; i < tablinks.length; i++) {
		tablinks[i].className = tablinks[i].className.replace(" active", "");
	}

	// Show the current tab, and add an "active" class to the link that opened the tab
	document.getElementById(tabName).style.display = "block";
	evt.currentTarget.className += " active";

	// Redraw and re-enable queries for this chart
	for (i = 2; i < arguments.length; i++) {
		arguments[i].clear();
		arguments[i].setRefreshInterval(5);
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
	//openGstlalTab(event, 'Status', time_since_last_wrapper, time_since_trigger_wrapper, up_time_wrapper, dropped_wrapper, ram_status_wrapper);
});


/*
 * Charts about latency
 */


/*
 * https://developers.google.com/chart/interactive/docs/reference#Query
 */

var QueryWrapper = function(query, visualization, visOptions, errorContainer) {

  this.query = query;
  this.visualization = visualization;
  this.options = visOptions || {};
  this.errorContainer = errorContainer;
  this.currentDataTable = null;

  if (!visualization || !('draw' in visualization) ||
      (typeof(visualization['draw']) != 'function')) {
    throw Error('Visualization must have a draw method.');
  }
};


/** Draws the last returned data table, if no data table exists, does nothing.*/
QueryWrapper.prototype.draw = function() {
  if (!this.currentDataTable) {
    return;
  }
  this.visualization.draw(this.currentDataTable, this.options);
};


QueryWrapper.prototype.sendAndDraw = function() {
  var query = this.query;
  var self = this;
  query.send(function(response) {self.handleResponse(response)});
};


/** Handles the query response returned by the data source. */
QueryWrapper.prototype.handleResponse = function(response) {
  this.currentDataTable = null;
  if (response.isError()) {
    this.handleErrorResponse(response);
  } else {
    this.currentDataTable = response.getDataTable();
    this.draw();
  }
};


/** Handles a query response error returned by the data source. */
QueryWrapper.prototype.handleErrorResponse = function(response) {
  var message = response.getMessage();
  var detailedMessage = response.getDetailedMessage();
  if (this.errorContainer) {
    google.visualization.errors.addError(this.errorContainer,
        message, detailedMessage, {'showInTooltip': false});
  } else {
    throw Error(message + ' ' + detailedMessage);
  }
};


/** Aborts the sending and drawing. */
QueryWrapper.prototype.abort = function() {
  this.query.abort();
};


//function ChartWrapper(chartType, dataSourceUrl, query, refreshInterval, options, containerId) {
function ChartWrapper(obj) {

	this.chartType = obj.chartType;
	this.dataSourceUrl = obj.dataSourceUrl;
	this.query = obj.query;
	this.query_object = null;
	this.refreshInterval = obj.refreshInterval;
	this.options = obj.options;
	this.containerId = obj.containerId;
	this.container = document.getElementById(this.containerId);

	command = "this.chart = new google.visualization." + this.chartType  + "(this.container)";
	eval(command);

	this.clear = function() {
		this.query_object && this.query_object.abort();
		this.chart.clearChart();
	}	

	this.draw = function() {
		this.query_object && this.query_object.abort();
		this.query_object = new google.visualization.Query(this.dataSourceUrl + "&tq=" + this.query);
		this.query_object.setRefreshInterval(this.refreshInterval);
		var queryWrapper = new QueryWrapper(this.query_object, this.chart, this.options, this.container);
		queryWrapper.sendAndDraw();
	}

	this.setRefreshInterval = function (refreshInterval) {
		this.refreshInterval = refreshInterval;
		this.query_object.setRefreshInterval(this.refreshInterval);
	}
}


function drawLatencyStatusByNodes(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] };
	these_options.title = 'Latency';

	latency_status_by_nodes_wrapper = new ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:0&gpstime=' + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select latency_history where status by node',
		refreshInterval: refresh,
		options: these_options, 
		containerId: 'latency_status_by_nodes_wrapper',
	});
	latency_status_by_nodes_wrapper.draw();
	charts.push(latency_status_by_nodes_wrapper);
}


function drawLatencyHistory(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:5, maxValue:75, textPosition: 'out', ticks: [8,16,32,64] };
	these_options.title = 'Latency';

	latency_history_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:100&gpstime=' + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select latency_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'latency_history_wrapper',
	});
	latency_history_wrapper.draw();
	charts.push(latency_history_wrapper);
}


function drawLatencyGauge(gps, duration, refresh, analysis_path, job_ids) {
	latency_gauge_wrapper = new ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:200' + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
	query: 'select latency_history where now',
	refreshInterval: refresh,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 500, height: 500,
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
	charts.push(latency_gauge_wrapper);
}


/*
 * Charts about SNR
 * NOTE these start numbering reqId at 100
 */


function drawSNRStatusByNodes(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.title = 'SNR';

	snr_status_by_nodes_wrapper = new ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:300'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select snr_history where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'snr_status_by_nodes_wrapper',
	});
	snr_status_by_nodes_wrapper.draw();
	charts.push(snr_status_by_nodes_wrapper);
}


function drawSNRHistory(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.title = 'SNR';

	snr_history_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:400'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select snr_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'snr_history_wrapper',
	});
	snr_history_wrapper.draw();
	charts.push(snr_history_wrapper);
}


function drawLikelihoodStatusByNodes(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.title = 'Likelihood';

	likelihood_status_by_nodes_wrapper = new ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:301'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select likelihood_history where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'likelihood_status_by_nodes_wrapper',
	});
	likelihood_status_by_nodes_wrapper.draw();
	charts.push(likelihood_status_by_nodes_wrapper);
}


function drawLikelihoodHistory(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:4, maxValue:150, textPosition: 'out', ticks: [4,8,16,32,64] };
	these_options.title = 'Likelihood';

	likelihood_history_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:401'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select likelihood_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'likelihood_history_wrapper',
	});
	likelihood_history_wrapper.draw();
	charts.push(likelihood_history_wrapper);
}


function drawFARStatusByNodes(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:0.00000001, maxValue:1, textPosition: 'out', ticks: [0.00000001, 0.000001, 0.0001, 0.01, 1], format: 'scientific' };
	these_options.title = 'FAR';

	far_status_by_nodes_wrapper = new ChartWrapper({
		chartType: 'CandlestickChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:302'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select far_history where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'far_status_by_nodes_wrapper',
	});
	far_status_by_nodes_wrapper.draw();
	charts.push(far_status_by_nodes_wrapper);
}


function drawFARHistory(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:0.0000001, maxValue:1, textPosition: 'out', ticks: [0.00000001, 0.000001, 0.0001, 0.01, 1], format: 'scientific' };
	these_options.title = 'FAR';

	far_history_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:402'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select far_history where node is all',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'far_history_wrapper',
	});
	far_history_wrapper.draw();
	charts.push(far_history_wrapper);
}

/*
 * Charts about sensitivity
 */

function drawHorizon(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {minValue:0, maxValue:250, textPosition: 'out'};
	these_options.title = 'Horizon';
	these_options.series = {0: {color: "red"}, 1: {color:"green"}};

	horizon_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:500'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select horizon_history',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'horizon_wrapper',
	});

	horizon_table_wrapper = new ChartWrapper({
		chartType: 'Table',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:501'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select horizon_history',
		refreshInterval: refresh,
		containerId: 'horizon_table_wrapper',
		options : { sortColumn: 1, width: "100%", page : "enable", sortAscending : false},
	});


	horizon_wrapper.draw();
	horizon_table_wrapper.draw();
	charts.push(horizon_wrapper);
	charts.push(horizon_table_wrapper);
}


function drawPSD(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{max:1e-18, min:1e-24}, format: 'scientific'};
	these_options.hAxis = {scaleType: 'log', textPosition: 'out', viewWindowMode:'explicit', viewWindow:{min:10, max:2048}};
	these_options.title = 'Amplitude Spectral Density';
	these_options.series = {lineWidth: 6, 0: {color: "red"}, 1: {color:"green"}};
	these_options.interpolateNulls = true;

	psd_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:600'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select psd' + ((gps == "-1") ? 'where now' : ''),
		refreshInterval: refresh,
		options: these_options,
		containerId: 'psd_wrapper',
	});
	psd_wrapper.draw();
	charts.push(psd_wrapper);
}


function drawRangeGauge(gps, duration, refresh, analysis_path, job_ids) {
	range_gauge_wrapper = new ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:700'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
	query: 'select horizon_history where now',
	refreshInterval: refresh,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1000, height: 1000,
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
	charts.push(range_gauge_wrapper);
}


/*
 * Charts about noise
 */


function drawNoise(gps, duration, refresh, analysis_path, job_ids) {
	// Setup the custom options
	var these_options = clone(default_options);
	these_options.title = "Whitened h(t)";

	noise_wrapper = new ChartWrapper({
		chartType: 'LineChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:800'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select noise',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'noise_wrapper',
	});

	noise_table_wrapper = new ChartWrapper({
		chartType: 'Table',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:801'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select noise',
		refreshInterval: refresh,
		containerId: 'noise_table_wrapper',
		options : { sortColumn: 1, width: "100%", page : "enable", sortAscending : false},
	});

	noise_wrapper.draw();
	noise_table_wrapper.draw();
	charts.push(noise_wrapper);
	charts.push(noise_table_wrapper);
}


function drawNoiseGauge(gps, duration, refresh, analysis_path, job_ids) {
	noise_gauge_wrapper = new ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:900'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
	query: 'select noise where now',
	refreshInterval: refresh,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 1000, height: 1000,
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
	charts.push(noise_gauge_wrapper);
}


/*
 * Charts about analysis state
 */


function drawUpTime(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.title = 'Up Time';
	these_options.vAxis = {textPosition: 'out', viewWindowMode:'explicit', gridlines: {color:'#FFFFFF'}, scaleType: 'log', minValue:1, maxValue:100000000, format: 'scientific'}
	up_time_wrapper = new ChartWrapper({
		chartType: 'ColumnChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1000'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select _state_vector_on_off_gap where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'up_time_wrapper',
	});
	up_time_wrapper.draw();
	charts.push(up_time_wrapper);
}

function drawUpTimeGauge(gps, duration, refresh, analysis_path, job_ids) {
	uptime_gauge_wrapper = new ChartWrapper({
	chartType: 'Gauge',
	dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1010'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
	query: 'select _state_vector_on_off_gap where now',
	refreshInterval: refresh,
        options: {
		animation: {duration: 4000, easing: 'linear'},
		width: 500, height: 500,
		redFrom: 0, redTo: 4,
		yellowFrom: 4, yellowTo: 12,
		greenFrom: 12, greenTo: 24,
		minorTicks: 5,
		max: 24,
		min: 0
		},
	containerId: 'uptime_gauge_wrapper',
	});

	uptime_gauge_wrapper.draw();
	charts.push(uptime_gauge_wrapper);
}

function drawDroppedData(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.title = 'Dropped Data';
	these_options.vAxis = {textPosition: 'out', viewWindowMode:'explicit', gridlines: {color:'#FFFFFF'}, scaleType: 'log', minValue:1, maxValue:100000000, format: 'scientific'}
	dropped_wrapper = new ChartWrapper({
		chartType: 'ColumnChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1100'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select _strain_add_drop where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'dropped_wrapper',
	});
	dropped_wrapper.draw();
	charts.push(dropped_wrapper);
}

function drawRAMStatus(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:1, maxValue:16, textPosition: 'out', ticks: [1,2,4,8,16] };
	these_options.title = 'RAM';

	ram_status_wrapper = new ChartWrapper({
		chartType: 'ColumnChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1200'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select ram_history where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'ram_status_wrapper',
	});
	ram_status_wrapper.draw();
	charts.push(ram_status_wrapper);
}


function drawTimeSinceLast(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:1, maxValue:1000000, textPosition: 'out', ticks: [1,10,100,1000,10000,100000], format: 'scientific'};
	these_options.title = 'Time Since Last Heartbeat';

	time_since_last_wrapper = new ChartWrapper({
		chartType: 'ColumnChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1300'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select time_since_last where status by node',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'time_since_last_wrapper',
	});
	time_since_last_wrapper.draw();
	charts.push(time_since_last_wrapper);
}


function drawTimeSinceTrigger(gps, duration, refresh, analysis_path, job_ids) {
	var these_options = clone(default_options);
	these_options.vAxis = {scaleType: 'log', minValue:1, maxValue:1000000, textPosition: 'out', ticks: [1,10,100,1000,10000,100000], format: 'scientific'};
	these_options.title = 'Time Since Last Trigger';

	time_since_trigger_wrapper = new ChartWrapper({
		chartType: 'ColumnChart',
		dataSourceUrl: 'https://ldas-jobs.ligo.caltech.edu/~gstlalcbctest/cgi-bin/gstlal_data_server_latest_by_job?tqx=reqId:1301'  + '&gpstime='  + gps + '&duration=' + duration + '&id=' + job_ids + '&dir=' + analysis_path,
		query: 'select time_since_trigger',
		refreshInterval: refresh,
		options: these_options,
		containerId: 'time_since_trigger_wrapper',
	});
	time_since_trigger_wrapper.draw();
	charts.push(time_since_trigger_wrapper);
}
