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

		var is_img = $(this).is('img');

		this.each(function() {
			var obj = this;
			var old_src = $(this).attr(is_img ? 'src' : 'href');
			var func = function() {
				window.setTimeout(function() {
					// FIXME: we could load the image into a separate img object,
					// then swap URLs; this might help to avoid flicker.
					var new_src = old_src + "#e" + Math.random();
					if (is_img && $(obj).is(':visible'))
						obj.src = new_src;
					else
						obj.href = new_src;
					func();
				}, interval);
			};
			func();
		});

		return this;
	};
})(jQuery);
