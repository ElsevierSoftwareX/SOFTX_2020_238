/*
 * jQuery plugin for automatically updating images.
 *
 * To use, just include this script in your web page by adding the following
 * markup inside <head>:
 *
 * <script type="text/javascript" src="autoupdate.js"></script>
 * <script type="text/javascript">
 *
 *		$(document).ready(function() {
 *
 *			// Autoupdate all images with the attribute class="autoupdating"
 *			$('img.autoupdating").autoupdate();
 *
 *			// Autoupdate all images with the attribute id="webcam", once every 0.5 seconds
 *			$('img#webcam").autoupdate( {'interval': 500} );
 *
 *		};
 *
 * </script>
 *
 * Copyright (C) 2010 Leo Singer
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


(function($) {
	$.fn.autoupdate = function(settings) {

		var config = {'interval': 4000};
		if (settings) $.extend(config, settings);

		var interval = config['interval'];

		this.each(function() {
			var obj = this;
			var old_src = $(this).attr('src');
			var parent = $(obj).parent();
			var old_href = undefined;
			if (parent.is('a'))
				old_href=parent.attr('href');
			var func = function() {
				window.setTimeout(function() {
					// FIXME: we could load the image into a separate img object,
					// then swap URLs; this might help to avoid flicker.
					var new_id = "#e" + Math.random();
					if ($(obj).is(':visible'))
						obj.src = old_src + new_id;
					// This is a hack to get fancybox to work right.
					var parent = $(obj).parent();
					if (parent.is('a'))
						parent.attr('href', old_href + new_id);
					func();
				}, interval);
			};
			func();
		});

		return this;
	};
})(jQuery);
