$(document).ready(function(){
	$("head").append(
		'<link rel="stylesheet" type="text/css" href="assets/report.css" />' +
		'<link rel="stylesheet" type="text/css" href="assets/fancybox/jquery.fancybox-1.3.1.css" />' +
		'<script type="text/javascript" src="assets/fancybox/jquery.fancybox-1.3.1.pack.js" />'
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
	$("figure img").fancybox();
});
