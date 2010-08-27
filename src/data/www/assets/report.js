$(document).ready(function(){
	$("head").append(
		'<link rel="stylesheet" type="text/css" href="assets/report.css" />' +
		'<link rel="stylesheet" type="text/css" href="assets/fancybox/jquery.fancybox-1.3.1.css" />' +
		'<script type="text/javascript" src="assets/fancybox/jquery.fancybox-1.3.1.pack.js" />' +
		'<script type="text/javascript" src="assets/autoupdate.js" />'
	);
	$("section h1").toggle(
		function () {
			$(this).siblings().slideUp();
			$(this).attr("title", "Click to show the contents of this section.");
		},
		function () {
			$(this).siblings().slideDown();
			$(this).attr("title", "Click to hide the contents of this section.");
		}
	);
	$("section h1").attr("title", "Click to hide the contents of this section.");
	$("figure").each(function() {
		 var rel = 'gallery' + Math.random();
		 $(this).find("img").wrap(function() {return '<a href="' + $(this).attr('src') + '" rel="' + rel + '" />';});
	});
	$("figure a").fancybox({'cyclic': true, 'autoScale': true, 'transitionIn': 'elastic', 'transitionOut': 'elastic', 'changeFade': 100});
	$("img.autoupdating").autoupdate({'interval': 8000});
});
