#!/usr/bin/python2.7
# implement replacements as listed on
# https://cgit.freedesktop.org/gstreamer/gstreamer/tree/docs/random/porting-to-1.0.txt

import re
import sys

line_no = 0
gobject_gst_lineno_diff = 5
import_gobject_line_no = -1
import_gst_line_no = -1
mod_str_def=' /* MOD */'
mod_str_def=''
gobject_loaded=False
gst_loaded=False
gobject_postload=False

with open(sys.argv[1]) as f:
    for line in f:
        line_no += 1
        modded = False
        mod_str=False
        cut_line = False

        # strip the crlf
        line = line.rstrip("\r\n")

        # audio/x-raw-int   -> audio/x-raw
        (mod_line, mod_num) = re.subn(r'audio/x-raw-int', r'audio/x-raw', line)
        if mod_num != 0:
            mod_str=' /* MOD: audio/x-raw-int -> audio/x-raw */'
            line = mod_line
            modded = True

        # audio/x-raw-float -> audio/x-raw
        (mod_line, mod_num) = re.subn(r'audio/x-raw-float', r'audio/x-raw', line)
        if mod_num != 0:
            mod_str=' /* MOD: audio/x-raw-float -> audio/x-raw */'
            line = mod_line
            modded = True

        # text/plain -> text/x-raw, format=utf8
        (mod_line, mod_num) = re.subn(r'text/plain', r'text/x-raw, format=utf8', line)
        if mod_num != 0:
            mod_str=' /* MOD: text/plain -> text/x-raw, format=utf8 */'
            line = mod_line
            modded = True

        ############################################################
        # GstObject
        ############################################################
        # GST_OBJECT_DISPOSING flag removed
        m = re.search('GST_OBJECT_DISPOSING', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_OBJECT_DISPOSING flag removed */'
            modded = True

        # GST_OBJECT_IS_DISPOSING removed
        m = re.search('GST_OBJECT_IS_DISPOSING', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_OBJECT_IS_DISPOSING flag removed */'
            modded = True

        # GST_OBJECT_FLOATING flag removed, GstObject is now GInitiallyUnowned
        m = re.search('GST_OBJECT_FLOATING', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_OBJECT_FLOATING flag removed, GstObject is now GInitiallyUnowned */'
            modded = True

        # GST_OBJECT_IS_FLOATING removed, use g_object_is_floating()
        m = re.search('GST_OBJECT_IS_FLOATING', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_OBJECT_IS_FLOATING removed, use g_object_is_floating() */'
            modded = True

        # GST_CLASS_GET_LOCK, GST_CLASS_LOCK, GST_CLASS_TRYLOCK, GST_CLASS_UNLOCK,
        # used to be a workaround for thread-unsafe glib < 2.8

        # gst_object_ref_sink() has gpointer as result to make it more like the
        # GObject version.

        # gst_object_sink() removed, use gst_object_ref_sink() instead.
        (mod_line, mod_num) = re.subn(r'gst_object_sink', r'gst_object_ref_sink', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_object_sink() removed, use gst_object_ref_sink() instead. */'
            line = mod_line
            modded = True

        # gst_class_signal_connect() removed, was only used for XML
        m = re.search('gst_class_signal_connect', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_class_signal_connect() removed, was only used for XML */'
            modded = True

        # parent-set and parent-unset signals removed. Use notify:parent. Currently
        # still disabled because of deep notify locking issues.            

        ############################################################
        # GstElement
        ############################################################
        # GstElementDetails is removed and replaced with more generic metadata.
        # ???
        
        # gst_element_class_set_details_simple -> gst_element_class_set_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_class_set_details_simple', r'gst_element_class_set_metadata', line)
        if mod_num != 0:
            line = mod_line
            mod_str = ' /* MOD: gst_element_class_set_details_simple -> gst_element_class_set_metadata */'
            modded = True

        # gst_element_class_set_documentation_uri -> gst_element_class_add_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_class_set_documentation_uri', r'gst_element_class_add_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_class_set_documentation_uri -> gst_element_class_add_metadata */'
            line = mod_line
            modded = True

        # gst_element_class_set_documentation_uri -> gst_element_class_add_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_class_set_documentation_uri', r'gst_element_class_add_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_class_set_documentation_uri -> gst_element_class_add_metadata */'
            line = mod_line
            modded = True

        # also gst_element_class_get_metadata()
        m = re.search('gst_element_class_get_metadata', line)
        if m != None:
            mod_str=' /* MOD, FIXME: porting documentation is unclear, just says: also gst_element_class_get_metadata */'
            modded = True

        # gst_element_factory_get_longname -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_longname', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_longname -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gst_element_factory_get_klass -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_klass', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_klass -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gst_element_factory_get_description -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_description', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_description -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gst_element_factory_get_author -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_author', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_author -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gst_element_factory_get_documentation_uri -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_documentation_uri', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_documentation_uri -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gst_element_factory_get_icon_name -> gst_element_factory_get_metadata
        (mod_line, mod_num) = re.subn(r'gst_element_factory_get_icon_name', r'gst_element_factory_get_metadata', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_factory_get_icon_name -> gst_element_factory_get_metadata */'
            line = mod_line
            modded = True

        # gstelementmetadata.h contains the keys for all standard metadata.

        # gst_element_factory_can_{src,sink}_caps() => gst_element_factory_can_{src,sink}_{any,all}_caps()
        (mod_line, mod_num) = re.subn(r'gst_element_factory_can_(.+)_caps', r'gst_element_factory_can_\1_{CHOOSE:any,all}_caps', line)
        if mod_num != 0:
            mod_str = ' /* MOD, FIXME: gst_element_factory_can_{src,sink}_caps() => gst_element_factory_can_{src,sink}_{any,all}_caps() */'
            line = mod_line
            modded = True

        # gst_element_class_add_static_pad_template (element_class, &src_template) => gst_element_class_add_pad_template (element_class, gst_static_pad_template_get (&src_template));
        (mod_line, mod_num) = re.subn(r'gst_element_class_add_static_pad_template', r'gst_element_class_add_pad_template', line)
        if mod_num != 0:
            line = mod_line
            modded = True
            mod_str = '/* MOD, FIXME: gst_element_class_add_static_pad_template (element_class, &src_template) => gst_element_class_add_pad_template (element_class, gst_static_pad_template_get (&src_template)); */'

        # gst_element_lost_state_full() -> gst_element_lost_state()
        (mod_line, mod_num) = re.subn(r'gst_element_lost_state_full', r'gst_element_lost_state', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_lost_state_full() -> gst_element_lost_state() */'
            line = mod_line
            modded = True

        # gst_element_lost_state() -> gst_element_lost_state(, TRUE)
        (mod_line, mod_num) = re.subn(r'gst_element_lost_state\((.+)\)', r'gst_element_lost_state(\1, TRUE)', line)
        if mod_num != 0:
            mod_str = ' /* MOD: gst_element_lost_state() -> gst_element_lost_state(, TRUE) */'
            line = mod_line
            modded = True
        else:
            m = re.search('gst_element_lost_state', line)
            if m != None:
                mod_str=' /* MOD, FIXME: gst_element_lost_state() -> gst_element_lost_state(, TRUE) */'
                modded = True

	# Element metadata and pad templates are inherited from parent classes and
	# should be added in class_init instead of base_init.

	# Elements that change the duration must post DURATION messages on the
	# bus when the duration changes in PAUSED or PLAYING.

	# gst_element_found_tags() and gst_element_found_tags_for_pad() are gone, just
	# push the tag event.
        m = re.search('gst_element_found_tags', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_element_found_tags() and gst_element_found_tags_for_pad() are gone, just push the tag event. */'
            modded = True

        m = re.search('gst_element_found_tags_for_pad', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_element_found_tags() and gst_element_found_tags_for_pad() are gone, just push the tag event. */'
            modded = True

        # request_new_pad_full() -> request_new_pad()
        (mod_line, mod_num) = re.subn(r'request_new_pad_full', r'request_new_pad', line)
        if mod_num != 0:
            mod_str = ' /* MOD: request_new_pad_full() -> request_new_pad() */'
            line = mod_line
            modded = True
        
            
        ############################################################
        # GstPad
        ############################################################

        # gst_pad_get_caps() was replaced by gst_pad_query_caps(), it
        # does not return writable caps anymore and an explicit
        # gst_caps_make_writable() needs to be performed. This was the functionality
        # of gst_pad_get_caps_reffed(), which is removed now.
        #
        # Actually do not change, just
        m = re.search('gst_pad_get_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_get_caps() was replaced by gst_pad_query_caps(), it does not return writable caps anymore and an explicit gst_caps_make_writable() needs to be performed. This was the functionality of gst_pad_get_caps_reffed(), which is removed now. */'
            modded = True

	# A similar change was done for gst_pad_peer_get_caps() and
	# gst_pad_peer_get_caps_reffed()
        m = re.search('gst_pad_get_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_get_caps() was replaced by gst_pad_query_caps(), it does not return writable caps anymore and an explicit gst_caps_make_writable() needs to be performed. This was the functionality of gst_pad_get_caps_reffed(), which is removed now. A similar change was done for gst_pad_peer_get_caps() and gst_pad_peer_get_caps_reffed() */'
            modded = True

	# gst_pad_set_bufferalloc_function(), gst_pad_alloc_buffer() and
	# gst_pad_alloc_buffer_and_set_caps() are removed. Use the ALLOCATION query
	# now when negotiating formats to obtain a reference to a bufferpool object
	# that can be used to allocate buffers using gst_buffer_pool_acquire_buffer().
        m = re.search('(gst_pad_set_bufferalloc_function|gst_pad_alloc_buffer|gst_pad_alloc_buffer_and_set_caps)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_bufferalloc_function(), gst_pad_alloc_buffer() and gst_pad_alloc_buffer_and_set_caps() are removed. Use the ALLOCATION query now when negotiating formats to obtain a reference to a bufferpool object that can be used to allocate buffers using gst_buffer_pool_acquire_buffer(). */'
            modded = True

	# gst_pad_set_setcaps_function() => GST_EVENT_CAPS in event handler
        m = re.search('gst_pad_set_setcaps_function', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_setcaps_function() => GST_EVENT_CAPS in event handler */'
            modded = True

	# gst_pad_set_getcaps_function() => GST_QUERY_CAPS in query handler
        m = re.search('gst_pad_set_setcaps_function', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_getcaps_function() => GST_QUERY_CAPS in query handler */'
            modded = True

	# gst_pad_set_acceptcaps_function() => GST_QUERY_ACCEPT_CAPS in query handler
        m = re.search('gst_pad_set_acceptcaps_function', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_acceptcaps_function() => GST_QUERY_ACCEPT_CAPS in query handler */'
            modded = True

	# removed sched_private, it should not be used, use g_object_set_qdata() or
	# use element_private.
        m = re.search('sched_private', line)
        if m != None:
            mod_str=' /* MOD, FIXME: removed sched_private, it should not be used, use g_object_set_qdata() or use element_private. */'
            modded = True        

	# Removed GST_PAD_CAPS() use gst_pad_get_current_caps() to get a handle to the
	# currently configured caps.
        m = re.search('GST_PAD_CAPS', line)
        if m != None:
            mod_str=' /* MOD, FIXME: Removed GST_PAD_CAPS() use gst_pad_get_current_caps() to get a handle to the currently configured caps. */'
            modded = True        

	# gst_pad_get_pad_template_caps() and gst_pad_get_pad_template()
	# return a new reference of the caps or template now and the return
	# value needs to be unreffed after usage.
        m = re.search('(gst_pad_get_pad_template_caps|gst_pad_get_pad_template)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_get_pad_template_caps() and gst_pad_get_pad_template() return a new reference of the caps or template now and the return value needs to be unreffed after usage. */'
            modded = True        

	# gst_pad_set_caps() now pushes a CAPS event for backward compatibility.
	# Consider sending the CAPS event yourself. It is not possible anymore to set
	# NULL caps.
        m = re.search('gst_pad_set_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_caps() now pushes a CAPS event for backward compatibility. Consider sending the CAPS event yourself. It is not possible anymore to set NULL caps. */'
            modded = True        

	# gst_pad_set_checkgetrange_function() and gst_pad_check_pull_range() are
	# gone, use the SCHEDULING query now.
        m = re.search('(gst_pad_set_checkgetrange_function|gst_pad_check_pull_range)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_checkgetrange_function() and gst_pad_check_pull_range() are gone, use the SCHEDULING query now. */'
            modded = True                

	# gst_pad_set_blocked(), gst_pad_set_blocked_async(),
	# gst_pad_set_blocked_async_full() are removed, use the gst_pad_add_probe()
	# method with the GST_PAD_PROBE_TYPE_BLOCK to get the same result as the async
	# blocking version. There is no more sync version of blocking, this is in
	# general dangerous and can be implemented using the callbacks if needed.
        m = re.search('(gst_pad_set_blocked|gst_pad_set_blocked_async|gst_pad_set_blocked_async_full|gst_pad_add_probe)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_set_blocked(), gst_pad_set_blocked_async(), gst_pad_set_blocked_async_full() are removed, use the gst_pad_add_probe() method with the GST_PAD_PROBE_TYPE_BLOCK to get the same result as the async blocking version. There is no more sync version of blocking, this is in general dangerous and can be implemented using the callbacks if needed. */'
            modded = True                

	# gst_pad_add_data_probe(), gst_pad_add_data_probe_full(),
	# gst_pad_remove_data_probe(), gst_pad_add_event_probe(),
	# gst_pad_add_event_probe_full(), gst_pad_remove_event_probe(),
	# gst_pad_add_buffer_probe(), gst_pad_add_buffer_probe_full(),
	# gst_pad_remove_buffer_probe() are removed. Use gst_pad_add_probe() and
	# gst_pad_remove_probe() for equivalent functionality.
        m = re.search('(gst_pad_add_data_probe|gst_pad_add_data_probe_full|gst_pad_remove_data_probe|gst_pad_add_event_probe|gst_pad_add_event_probe_full|gst_pad_remove_event_probe|gst_pad_add_buffer_probe|gst_pad_add_buffer_probe_full|gst_pad_remove_buffer_probe)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_add_data_probe(), gst_pad_add_data_probe_full(), gst_pad_remove_data_probe(), gst_pad_add_event_probe(), gst_pad_add_event_probe_full(), gst_pad_remove_event_probe(), gst_pad_add_buffer_probe(), gst_pad_add_buffer_probe_full(), gst_pad_remove_buffer_probe() are removed. Use gst_pad_add_probe() and gst_pad_remove_probe() for equivalent functionality. */'
            modded = True                
        
	# The have-data signal was removed from pads, it was never supposed to be used
	# without calling the _add_.*_probe() methods.
        # ???

	# The request-link signal was removed. It was never used.

	# gst_pad_get_negotiated_caps() -> gst_pad_get_current_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_get_negotiated_caps', r'gst_pad_get_current_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_get_negotiated_caps() -> gst_pad_get_current_caps() */'
            line = mod_line
            modded = True

	# GST_FLOW_UNEXPECTED -> GST_FLOW_EOS
        (mod_line, mod_num) = re.subn(r'GST_FLOW_UNEXPECTED', r'GST_FLOW_EOS', line)
        if mod_num != 0:
            mod_str=' /* MOD: GST_FLOW_UNEXPECTED -> GST_FLOW_EOS */'
            line = mod_line
            modded = True

	# GST_FLOW_WRONG_STATE -> GST_FLOW_FLUSHING
        (mod_line, mod_num) = re.subn(r'GST_FLOW_WRONG_STATE', r'GST_FLOW_FLUSHING', line)
        if mod_num != 0:
            mod_str=' /* MOD: GST_FLOW_WRONG_STATE -> GST_FLOW_FLUSHING */'
            line = mod_line
            modded = True

	# GstActivateMode -> GstPadMode
        (mod_line, mod_num) = re.subn(r'GstActivateMode', r'GstPadMode', line)
        if mod_num != 0:
            mod_str=' /* MOD: GstActivateMode -> GstPadMode */'
            line = mod_line
            modded = True

        # GST_ACTIVATE_* -> GST_PAD_MODE_*
        (mod_line, mod_num) = re.subn(r'GST_ACTIVATE_(.+)', r'GST_PAD_MODE_\1', line)
        if mod_num != 0:
            mod_str=' /* MOD: GST_ACTIVATE_* -> GST_PAD_MODE_* */'
            line = mod_line
            modded = True

	# gst_pad_activate_{pull,push}() -> gst_pad_activate_mode()
        (mod_line, mod_num) = re.subn(r'gst_pad_activate_push', r'gst_pad_activate_mode', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_activate_{pull,push}() -> gst_pad_activate_mode() */'
            line = mod_line
            modded = True
        (mod_line, mod_num) = re.subn(r'gst_pad_activate_pull', r'gst_pad_activate_mode', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_activate_{pull,push}() -> gst_pad_activate_mode() */'
            line = mod_line
            modded = True

	# The GstPadAcceptCapsFunction was removed and replaced with a
	# GST_QUERY_ACCEPT_CAPS query.
        m = re.search('GstPadAcceptCapsFunction', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The GstPadAcceptCapsFunction was removed and replaced with a GST_QUERY_ACCEPT_CAPS query. */'
            modded = True

	# The GstPadFixateCapsFunction was removed. It has no replacement, you can
	# simply do the fixation in the element or use a vmethod from the base class
	# if appropriate.
        m = re.search('GstPadFixateCapsFunction', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The GstPadFixateCapsFunction was removed. It has no replacement, you can simply do the fixation in the element or use a vmethod from the base class if appropriate. */'
            modded = True

	# The GstPadGetCapsFunction was removed and replaced with a GST_QUERY_CAPS
	# query. The query takes a GstCaps* parameter to inform the other side about
	# the possible caps and preferences.  
        m = re.search('GstPadGetCapsFunction', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The GstPadGetCapsFunction was removed and replaced with a GST_QUERY_CAPS query. The query takes a GstCaps* parameter to inform the other side about the possible caps and preferences. */'
            modded = True

	# gst_pad_proxy_getcaps() -> gst_pad_proxy_query_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_proxy_getcaps', r'gst_pad_proxy_query_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_proxy_getcaps() -> gst_pad_proxy_query_caps() */'
            line = mod_line
            modded = True

	# gst_pad_get_caps() -> gst_pad_query_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_get_caps', r'gst_pad_query_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_get_caps() -> gst_pad_query_caps() */'
            line = mod_line
            modded = True

	# gst_pad_peer_get_caps() -> gst_pad_peer_query_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_peer_get_caps', r'gst_pad_peer_query_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_peer_get_caps() -> gst_pad_peer_query_caps() */'
            line = mod_line
            modded = True

        # gst_pad_accept_caps() -> gst_pad_query_accept_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_accept_caps', r'gst_pad_query_accept_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_accept_caps() -> gst_pad_query_accept_caps() */'
            line = mod_line
            modded = True

        # gst_pad_peer_accept_caps() -> gst_pad_peer_query_accept_caps()
        (mod_line, mod_num) = re.subn(r'gst_pad_peer_accept_caps', r'gst_pad_peer_query_accept_caps', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_peer_accept_caps() -> gst_pad_peer_query_accept_caps() */'
            line = mod_line
            modded = True

        # gst_pad_query_peer_*() -> gst_pad_peer_query_*()
        (mod_line, mod_num) = re.subn(r'gst_pad_query_peer_(.+)', r'gst_pad_peer_query_\1', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_pad_query_peer_*() -> gst_pad_peer_query_*() */'
            line = mod_line
            modded = True

	# GstPadFlags: GST_PAD_* -> GST_PAD_FLAG_*
        (mod_line, mod_num) = re.subn(r'GST_PAD_(.+)', r'GST_PAD_FLAG_\1', line)
        if mod_num != 0:
            mod_str=' /* MOD: GstPadFlags: GST_PAD_* -> GST_PAD_FLAG_* */'
            line = mod_line
            modded = True

        ############################################################
        # GstPadTemplate
        ############################################################
        # gst_pad_template_get_caps() returns a new reference of the caps
	# and the return value needs to be unreffed after usage.
        m = re.search('gst_pad_template_get_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_template_get_caps() returns a new reference of the caps and the return value needs to be unreffed after usage. */'
            modded = True

	# gst_pad_template_new() does not take ownership of the caps anymore.
        m = re.search('gst_pad_template_new', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_pad_template_new() does not take ownership of the caps anymore. */'
            modded = True

	# GstPadTemplate is now created with a floating ref and
	# gst_element_class_add_pad_template() takes ownership of this floating ref.
        m = re.search('gst_element_class_add_pad_template', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstPadTemplate is now created with a floating ref and gst_element_class_add_pad_template() takes ownership of this floating ref. */'
            modded = True

	# GstPadTemplate instances are considered immutable and must not be
	# changed.
        m = re.search('GstPadTemplate', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstPadTemplate instances are considered immutable and must not be changed. */'
            modded = True

        ############################################################
        # GstMiniObject
        ############################################################
	# A miniobject is now a simple refcounted structure holding the information
	# common to buffers, events, messages, queries and caps.

	# There is no more GST_TYPE_MINIOBJECT as the type for subclasses.
	# G_TYPE_BOXED can be used as the type of all GstMiniObject based types such
	# as buffers, events, messages, caps, etc. Signals, for example, would use the
	# boxed type if the argument include GstMiniObject derived types.
        m = re.search('GST_TYPE_MINIOBJECT', line)
        if m != None:
            mod_str=' /* MOD, FIXME: There is no more GST_TYPE_MINIOBJECT as the type for subclasses. G_TYPE_BOXED can be used as the type of all GstMiniObject based types such as buffers, events, messages, caps, etc. Signals, for example, would use the boxed type if the argument include GstMiniObject derived types. */'
            modded = True

	# gst_mini_object_new() is removed. You would allocate memory with the
	# methods specific for the derived type.
        m = re.search('gst_mini_object_new', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_mini_object_new() is removed. You would allocate memory with the methods specific for the derived type. */'
            modded = True

	# GstParamSpecMiniObject is removed, use boxed param spec now with the GType
	# of the specific GstMiniObject derived type. Also
	# gst_param_spec_mini_object().
        m = re.search('GstParamSpecMiniObject', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstParamSpecMiniObject is removed, use boxed param spec now with the GType of the specific GstMiniObject derived type. Also gst_param_spec_mini_object(). */'
            modded = True

	# gst_param_spec_mini_object() -> g_param_spec_boxed()
        (mod_line, mod_num) = re.subn(r'gst_param_spec_mini_object', r'g_param_spec_boxed', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_param_spec_mini_object() -> g_param_spec_boxed() */'
            modded = True

	# The specific gst_value_*_mini_object() methods are removed, used the generic
	# boxed methods instead.
        m = re.search('gst_value_(.+)_mini_object', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The specific gst_value_*_mini_object() methods are removed, used the generic boxed methods instead.*/'
            modded = True

	# gst_value_set_mini_object() -> g_value_set_boxed()
        (mod_line, mod_num) = re.subn(r'gst_value_set_mini_object', r'g_value_set_boxed', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_value_set_mini_object() -> g_value_set_boxed() */'
            modded = True
            
	# gst_value_take_mini_object() -> g_value_take_boxed()
        (mod_line, mod_num) = re.subn(r'gst_value_take_mini_object', r'g_value_take_boxed', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_value_take_mini_object() -> g_value_take_boxed() */'
            line = mod_line
            modded = True
            
	# gst_value_take_get_object() -> g_value_get_boxed()
        (mod_line, mod_num) = re.subn(r'gst_value_take_get_object', r'g_value_get_boxed', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_value_take_get_object() -> g_value_get_boxed() */'
            line = mod_line
            modded = True
            
	# gst_value_take_dup_object() -> g_value_dup_boxed()
        (mod_line, mod_num) = re.subn(r'gst_value_take_dup_object', r'g_value_dup_boxed', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_value_take_dup_object() -> g_value_dup_boxed() */'
            line = mod_line
            modded = True
            
	# GST_VALUE_HOLDS_MINI_OBJECT() was removed, use G_VALUE_HOLDS_BOXED() or
	# type-specific GST_VALUE_HOLDS_{BUFFER,CAPS,etc.}() instead.
        m = re.search('GST_VALUE_HOLDS_MINI_OBJECT', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_VALUE_HOLDS_MINI_OBJECT() was removed, use G_VALUE_HOLDS_BOXED() or type-specific GST_VALUE_HOLDS_{BUFFER,CAPS,etc.}() instead. */'
            modded = True

	# The GST_MINI_OBJECT_READONLY flag was removed as it used to mark the
	# memory in buffers as READONLY. Marking memory READONLY can now be done
	# with the GstMemory API. Writability of miniobjects is now either done
	# by using the refcount or by using exclusive locking.
        m = re.search('GST_MINI_OBJECT_READONLY', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The GST_MINI_OBJECT_READONLY flag was removed as it used to mark the memory in buffers as READONLY. Marking memory READONLY can now be done with the GstMemory API. Writability of miniobjects is now either done by using the refcount or by using exclusive locking. */'
            modded = True

        ############################################################
        # GstBuffer
        ############################################################

	# A GstBuffer is now a simple boxed type this means that subclassing is not
	# possible anymore. 
	 
	# To add data to the buffer you would now use gst_buffer_insert_memory() with
	# a GstMemory object containing the data. Multiple memory blocks can added to
	# a GstBuffer that can then be retrieved with gst_buffer_peek_memory().
	 
	# GST_BUFFER_DATA(), GST_BUFFER_MALLOCDATA(), GST_BUFFER_FREE_FUNC() and
	# GST_BUFFER_SIZE() are gone, along with the fields in GstBuffer.
        m = re.search('(GST_BUFFER_DATA|GST_BUFFER_MALLOCDATA|GST_BUFFER_FREE_FUNC|GST_BUFFER_SIZE)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_BUFFER_DATA(), GST_BUFFER_MALLOCDATA(), GST_BUFFER_FREE_FUNC() and GST_BUFFER_SIZE() are gone, along with the fields in GstBuffer. */'
            modded = True
        
	# The most common way to access all the data in a buffer is by using
	# gst_buffer_map() and gst_buffer_unmap(). These calls require you to specify
	# the access mode required to the data and will automatically merge and return
	# a writable copy of the data.

	# GST_BUFFER_SIZE() can be replaced with gst_buffer_get_size() but if also
	# access to the data is required, gst_buffer_map() can return both the size
	# and data in one go.
        m = re.search('GST_BUFFER_SIZE', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_BUFFER_SIZE() can be replaced with gst_buffer_get_size() but if also access to the data is required, gst_buffer_map() can return both the size and data in one go. */'
            modded = True

	# The buffer must be writable (gst_buffer_is_writable()) in order to modify
	# the fields, metadata or buffer memory. gst_buffer_make_writable() will not
	# automatically make a writable copy of the memory but will instead increase
	# the refcount of the memory. The _map() and _peek_memory() methods will
	# automatically create writable copies when needed.
        m = re.search('gst_buffer_make_writable', line)
        if m != None:
            mod_str=' /* MOD, FIXME: The buffer must be writable (gst_buffer_is_writable()) in order to modify the fields, metadata or buffer memory. gst_buffer_make_writable() will not automatically make a writable copy of the memory but will instead increase the refcount of the memory. The _map() and _peek_memory() methods will automatically create writable copies when needed. */'
            modded = True

	# gst_buffer_make_metadata_writable() is gone, you can replace this safely
	# with gst_buffer_make_writable().
        (mod_line, mod_num) = re.subn(r'gst_buffer_make_metadata_writable', r'gst_buffer_make_writable', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_buffer_make_metadata_writable() is gone, you can replace this safely with gst_buffer_make_writable(). */'
            line = mod_line
            modded = True

	# gst_buffer_copy_metadata() is gone, use gst_buffer_copy_into() instead and
	# mind use GST_BUFFER_COPY_METADATA instead of the former GST_BUFFER_COPY_ALL.
        m = re.search('gst_buffer_copy_metadata', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_buffer_copy_metadata() is gone, use gst_buffer_copy_into() instead and mind use GST_BUFFER_COPY_METADATA instead of the former GST_BUFFER_COPY_ALL. */'
            modded = True

	# gst_buffer_create_sub() is gone and can be safely replaced with
	# gst_buffer_copy_region().
        (mod_line, mod_num) = re.subn(r'gst_buffer_create_sub', r'gst_buffer_copy_region', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_buffer_create_sub() is gone and can be safely replaced with gst_buffer_copy_region(). */'
            line = mod_line
            modded = True
        
	# Changing the size of the buffer data can be done with gst_buffer_resize(),
	# which will also update the metadata fields correctly. gst_buffer_set_size()
	# is #defined to a special case of gst_buffer_resize() with a 0 offset.

	# gst_buffer_try_new_and_alloc() is replaced with gst_buffer_new_and_alloc(),
	# which now returns NULL when memory allocation fails.
        (mod_line, mod_num) = re.subn(r'gst_buffer_try_new_and_alloc', r'gst_buffer_new_and_alloc', line)
        if mod_num != 0:
            mod_str=' /* MOD, FIXME: gst_buffer_try_new_and_alloc() is replaced with gst_buffer_new_and_alloc(), which now returns NULL when memory allocation fails. */'
            line = mod_line
            modded = True

	# GST_BUFFER_CAPS() is gone, caps are not set on buffers anymore but are set
	# on the pads where the buffer is pushed on. Likewise GST_BUFFER_COPY_CAPS is
	# not needed anymore. gst_buffer_get/set_caps() are gone too.
        m = re.search('GST_BUFFER_CAPS', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_BUFFER_CAPS() is gone, caps are not set on buffers anymore but are set on the pads where the buffer is pushed on. Likewise GST_BUFFER_COPY_CAPS is not needed anymore. gst_buffer_get/set_caps() are gone too. */'
            modded = True
        
	# GST_BUFFER_TIMESTAMP is gone, use GST_BUFFER_PTS or GST_BUFFER_DTS instead.
	# Likewise GST_BUFFER_TIMESTAMP_IS_VALID() was changed to
	# GST_BUFFER_PTS_IS_VALID and GST_BUFFER_DTS_IS_VALID
        m = re.search('GST_BUFFER_TIMESTAMP', line)
        if m != None:
            mod_str=' /* MOD, FIXME:  GST_BUFFER_TIMESTAMP is gone, use GST_BUFFER_PTS or GST_BUFFER_DTS instead. Likewise GST_BUFFER_TIMESTAMP_IS_VALID() was changed to GST_BUFFER_PTS_IS_VALID and GST_BUFFER_DTS_IS_VALID*/'
            modded = True

	# gst_buffer_join() was renamed to gst_buffer_append() and the memory is not
	# directly merged but appended.
        m = re.search('gst_buffer_join', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_buffer_join() was renamed to gst_buffer_append() and the memory is not directly merged but appended. */'
            modded = True

	# gst_buffer_merge() was removed, it is the same as gst_buffer_join() but
	# without taking ownership of the arguments. Caller code should ref themselves
	# when needed. Note that the extra refs might force slower paths in
	# gst_buffer_join().
        m = re.search('gst_buffer_merge', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_buffer_merge() was removed, it is the same as gst_buffer_join() but without taking ownership of the arguments. Caller code should ref themselves when needed. Note that the extra refs might force slower paths in gst_buffer_join(). */'
            modded = True

	# gst_buffer_is_span() and gst_buffer_span() are removed, use
	# gst_buffer_merge() and gst_buffer_resize() for the same effect. Merging and
	# spanning is delayed until the buffer is mapped and in some cases no merging
	# of memory is needed at all when the element can deal with individual memory
	# chunks.        
        m = re.search('(gst_buffer_is_span|gst_buffer_span)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_buffer_is_span() and gst_buffer_span() are removed, use gst_buffer_merge() and gst_buffer_resize() for the same effect. Merging and spanning is delayed until the buffer is mapped and in some cases no merging of memory is needed at all when the element can deal with individual memory */'
            modded = True

        ############################################################
        # GstBufferList
        ############################################################

	# The GstBufferList object is much simplified because most of the
	# functionality in the groups is now part of the GstMemory in buffers.

	# The object is reduced to encapsulating an array of buffers that you can send
	# with the regular gst_pad_push_list. The iterator is not needed anymore
	# because you can simply use gst_buffer_list_length() and gst_buffer_list_get()
	# to iterate the array.

	# For dealing with the groups, it's now needed to add the memory blocks to
	# GstBuffer and use the normal buffer API to get and merge the groups.

	# gst_buffer_list_sized_new() -> gst_buffer_list_new_sized()
        (mod_line, mod_num) = re.subn(r'gst_buffer_list_sized_new', r'gst_buffer_list_new_sized', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_buffer_list_sized_new() -> gst_buffer_list_new_sized() */'
            modded = True
        
	# gst_buffer_list_len() -> gst_buffer_list_length()        
        (mod_line, mod_num) = re.subn(r'gst_buffer_list_len', r'gst_buffer_list_length', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_buffer_list_len() -> gst_buffer_list_length() */'
            modded = True

        ############################################################
        # GstStructure
        ############################################################
	# The GArray of the structure fields are moved to private part and are not
	# accessible from the application anymore. Use the methods to retrieve and
	# modify fields from the array.

	# gst_structure_empty_new() -> gst_structure_new_empty()
        (mod_line, mod_num) = re.subn(r'gst_structure_empty_new', r'gst_structure_new_empty', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_structure_empty_new() -> gst_structure_new_empty() */'
            modded = True
        
	# gst_structure_id_empty_new() -> gst_structure_new_id_empty()
        (mod_line, mod_num) = re.subn(r'gst_structure_id_empty_new', r'gst_structure_new_id_empty', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_structure_id_empty_new() -> gst_structure_new_id_empty() */'
            modded = True

        # gst_structure_id_new() -> gst_structure_new_id()
        (mod_line, mod_num) = re.subn(r'gst_structure_id_new', r'gst_structure_new_id', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_structure_id_new() -> gst_structure_new_id() */'
            modded = True

        ############################################################
        # GstEvent 
        ############################################################
	# Boxed types derived from GstMiniObject.

	# GST_EVENT_SRC is removed. Don't use this anymore.
        m = re.search('GST_EVENT_SRC', line)
        if m != None:
            mod_str=" /* MOD, FIXME: GST_EVENT_SRC is removed. Don't use this anymore. */"
            modded = True

	# gst_event_new_qos_full() -> gst_event_new_qos()
        (mod_line, mod_num) = re.subn(r'gst_event_new_qos_full', r'gst_event_new_qos', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_event_new_qos_full() -> gst_event_new_qos() */'
            modded = True
        
	# gst_event_parse_qos_full() -> gst_event_parse_qos()
        (mod_line, mod_num) = re.subn(r'gst_event_parse_qos_full', r'gst_event_parse_qos', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_event_parse_qos_full() -> gst_event_parse_qos() */'
            modded = True

	# The GstStructure is removed from the public API, use the getters to get
	# a handle to a GstStructure.
        m = re.search('GstStructure', line)
        if m != None:
            mod_str=" /* MOD, FIXME: The GstStructure is removed from the public API, use the getters to get a handle to a GstStructure. */"
            modded = True
        
	# GST_EVENT_NEWSEGMENT -> GST_EVENT_SEGMENT
        (mod_line, mod_num) = re.subn(r'GST_EVENT_NEWSEGMENT', r'GST_EVENT_SEGMENT', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_EVENT_NEWSEGMENT -> GST_EVENT_SEGMENT */'
            modded = True

	# gst_event_new_new_segment () -> gst_event_new_segment() and it takes a
	# GstSegment structure as an argument.
        m = re.search('gst_event_new_new_segment', line)
        if m != None:
            mod_str=" /* MOD, FIXME: gst_event_new_new_segment () -> gst_event_new_segment() and it takes a GstSegment structure as an argument. */"
            modded = True
        
	# gst_event_parse_new_segment() -> gst_event_parse_segment() to retrieve the
	# GstSegment structure from the event.
        m = re.search('gst_event_parse_new_segment', line)
        if m != None:
            mod_str=" /* MOD, FIXME: gst_event_parse_new_segment() -> gst_event_parse_segment() to retrieve the GstSegment structure from the event. */"
            modded = True

	# gst_event_copy_segment() to fill a GstSegment structure.
        m = re.search('gst_event_copy_segment', line)
        if m != None:
            mod_str=" /* MOD, FIXME: unclear in the porting page: gst_event_copy_segment() to fill a GstSegment structure. */"
            modded = True

	# gst_event_new_flush_stop() now takes a boolean, which in most cases should
	# be TRUE   
        m = re.search('gst_event_new_flush_stop', line)
        if m != None:
            mod_str=" /* MOD, FIXME: gst_event_new_flush_stop() now takes a boolean, which in most cases should be TRUE  */"
            modded = True

        ############################################################
        # GstQuery
        ############################################################
	# Boxed types derived from GstMiniObject.

	# The GstStructure is removed from the public API, use the getters to get
	# a handle to a GstStructure.
        m = re.search('GstStructure', line)
        if m != None:
            mod_str=" /* MOD, FIXME: The GstStructure is removed from the public API, use the getters to get a handle to a GstStructure. */"
            modded = True

	# gst_query_new_application() -> gst_query_new_custom()
        (mod_line, mod_num) = re.subn(r'gst_query_new_application', r'gst_query_new_custom', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_query_new_application() -> gst_query_new_custom() */'
            line = mod_line
            modded = True

	# gst_query_parse_formats_length() -> gst_query_parse_n_formats()
        (mod_line, mod_num) = re.subn(r'gst_query_parse_formats_length', r'gst_query_parse_n_formats', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_query_parse_formats_length() -> gst_query_parse_n_formats() */'
            line = mod_line
            modded = True

        # gst_query_parse_formats_nth() -> gst_query_parse_nth_format()
        (mod_line, mod_num) = re.subn(r'gst_query_parse_formats_nth', r'gst_query_parse_nth_format', line)
        if mod_num != 0:
            mod_str=' /* MOD: gst_query_parse_formats_nth() -> gst_query_parse_nth_format() */'
            line = mod_line
            modded = True

	# Some query utility functions no longer use an inout parameter for the
	# destination/query format:
	# 
	#   - gst_pad_query_position()
	#   - gst_pad_query_duration()
	#   - gst_pad_query_convert()
	#   - gst_pad_query_peer_position()
	#   - gst_pad_query_peer_duration()
	#   - gst_pad_query_peer_convert()
	#   - gst_element_query_position()
	#   - gst_element_query_duration()
	#   - gst_element_query_convert()
        m = re.search('(gst_pad_query_position|gst_pad_query_duration|gst_pad_query_convert|gst_pad_query_peer_position|gst_pad_query_peer_duration|gst_pad_query_peer_convert|gst_element_query_position|gst_element_query_duration|gst_element_query_convert)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: Some query utility functions no longer use an inout parameter for the destination/query format: gst_pad_query_position(), gst_pad_query_duration(), gst_pad_query_convert(), gst_pad_query_peer_position(), gst_pad_query_peer_duration(), gst_pad_query_peer_convert(), gst_element_query_position(), gst_element_query_duration(), gst_element_query_convert() */'
            modded = True

	# gst_element_get_query_types() and gst_pad_get_query_types() with associated
	# functions were removed.
        m = re.search('(gst_element_get_query_types|gst_pad_get_query_types)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_element_get_query_types() and gst_pad_get_query_types() with associated functions were removed. */'
            modded = True

        ############################################################
        # GstBufferList
        ############################################################
        # Is now a boxed type derived from GstMiniObject.

        ############################################################
        # GstMessage
        ############################################################
	# Is now a boxed type derived from GstMiniObject

	# The GstStructure is removed from the public API, use the getters to get
	# a handle to a GstStructure.
        # (already handled above)

	# GST_MESSAGE_DURATION -> GST_MESSAGE_DURATION_CHANGED
        (mod_line, mod_num) = re.subn(r'GST_MESSAGE_DURATION', r'GST_MESSAGE_DURATION_CHANGED', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_MESSAGE_DURATION -> GST_MESSAGE_DURATION_CHANGED */'
            modded = True

	# gst_message_parse_duration() was removed (not needed any longer, do
	# a duration query to query the updated duration)
        m = re.search('gst_message_parse_duration', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_message_parse_duration() was removed (not needed any longer, do a duration query to query the updated duration) */'
            modded = True

        ############################################################
        # GstCaps
        ############################################################
        # Is now a boxed type derived from GstMiniObject.

	# GST_VIDEO_CAPS_xxx -> GST_VIDEO_CAPS_MAKE(xxx)
        (mod_line, mod_num) = re.subn(r'GST_VIDEO_CAPS_(.+)', r'GST_VIDEO_CAPS_MAKE(\1)', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_VIDEO_CAPS_xxx -> GST_VIDEO_CAPS_MAKE(xxx) */'
            modded = True

	# Some caps functions now take ownership of the input argument, for
	# efficiency reasons (to avoid unnecessary copies to make them writable):
	# 
	#   gst_caps_normalize (caps)          =>   caps = gst_caps_normalize (caps)
	#   gst_caps_do_simplify (caps)        =>   caps = gst_caps_simplify (caps)
	#   gst_caps_merge (caps,caps2)        =>   caps = gst_caps_merge (caps,caps2)
	#   gst_caps_merge_structure (caps,st) =>   caps = gst_caps_merge_structure (caps,st)
	#   gst_caps_truncate (caps)           =>   caps = gst_caps_truncate (caps)
	# 
	# The compiler should warn about unused return values from these functions,
	# which may help find the places that need to be updated.
        m = re.search('(gst_caps_normalize|gst_caps_do_simplify|gst_caps_merge|gst_caps_merge_structure|gst_caps_truncate)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: Some caps functions now take ownership of the input argument, for efficiency reasons (to avoid unnecessary copies to make them writable): gst_caps_normalize (caps) => caps = gst_caps_normalize (caps); gst_caps_do_simplify (caps) => caps = gst_caps_simplify (caps); gst_caps_merge (caps,caps2) => caps = gst_caps_merge (caps,caps2); gst_caps_merge_structure (caps,st) => caps = gst_caps_merge_structure (caps,st); gst_caps_truncate (caps) => caps = gst_caps_truncate (caps). The compiler should warn about unused return values from these functions, which may help find the places that need to be updated. */'
            modded = True

	# Removed functions:
	# 
	#   gst_caps_union() -> gst_caps_merge():  Be careful because _merge takes
        #      ownership of the arguments.
        m = re.search('gst_caps_union', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_caps_union() -> gst_caps_merge():  Be careful because _merge takes ownership of the arguments. */'
            modded = True

        ############################################################
        # GstClock
        ############################################################
 	# gst_clock_id_wait_async_full() was renamed to gst_clock_id_wait_async() and
 	# the old gst_clock_id_wait_async() function was removed.
        (mod_line, mod_num) = re.subn(r'gst_clock_id_wait_async_full', r'gst_clock_id_wait_async', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_clock_id_wait_async_full() was renamed to gst_clock_id_wait_async() */'
            modded = True
        m = re.search('gst_clock_id_wait_async', line)
        if m != None:
            mod_str=' /* MOD, FIXME: the old gst_clock_id_wait_async() function was removed. */'
            modded = True

        ############################################################
        # GstSegment
        ############################################################
	# abs_rate was removed from the public fields, it can be trivially calculated
	# from the rate field.
        m = re.search('abs_rate', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstSegment: abs_rate was removed from the public fields, it can be trivially calculated from the rate field. */'
            modded = True

	# Also segment accumulation was removed from the segment event. This means
	# that now the source/demuxer/parser needs to add the elapsed time of the
	# previous segment themselves (this must be added to segment->base). If this
	# is not done, looped playback wont work.
        # ???

	# accum was renamed to base. last_stop was renamed to position.
        m = re.search('accum', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstSegment: accum was renamed to base. */'
            modded = True
        m = re.search('last_stop', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstSegment: last_stop was renamed to position. */'
            modded = True

	# The segment info now contains all the information needed to convert buffer
	# timestamps to running_time and stream_time. There is no more segment
	# accumulation, the GstSegment is completely self contained.

	# gst_segment_set_duration() and gst_segment_set_last_stop() are removed,
	# simply modify the structure members duration and position respectively.
        m = re.search('(gst_segment_set_duration|gst_segment_set_last_stop)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_segment_set_duration() and gst_segment_set_last_stop() are removed, simply modify the structure members duration and position respectively. */'
            modded = True

	# gst_segment_set_newsegment() is removed, it was used to accumulate segments
	# and is not needed anymore, use gst_segment_copy_into() or modify the segment
	# values directly.
        m = re.search('gst_segment_set_newsegment', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_segment_set_newsegment() is removed, it was used to accumulate segments and is not needed anymore, use gst_segment_copy_into() or modify the segment values directly. */'
            modded = True

	# gst_segment_set_seek() -> gst_segment_do_seek(). Updates the segment values
	# with seek parameters.
        m = re.search('gst_segment_set_seek', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_segment_set_seek() -> gst_segment_do_seek(). Updates the segment values with seek parameters. */'
            modded = True

        ############################################################
        # GstPluginFeature
        ############################################################
        # GST_PLUGIN_FEATURE_NAME() was removed, use GST_OBJECT_NAME() instead.
        (mod_line, mod_num) = re.subn(r'GST_PLUGIN_FEATURE_NAME', r'GST_OBJECT_NAME', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_PLUGIN_FEATURE_NAME() was removed, use GST_OBJECT_NAME() instead. */'
            modded = True

        ############################################################
        # GstTypeFind
        ############################################################
        # gst_type_find_peek() returns a const guint8 * now.
        m = re.search('gst_type_find_peek', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_type_find_peek() returns a const guint8 * now. */'
            modded = True        

        ############################################################
        # GstTask
        ############################################################
        # gst_task_create() -> gst_task_new()
        (mod_line, mod_num) = re.subn(r'gst_task_create', r'gst_task_new', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_task_create() -> gst_task_new() */'
            modded = True

        ############################################################
        # GstAudio
        ############################################################
	#     GstBaseAudioSink -> GstAudioBaseSink
	#     GstBaseAudioSrc -> GstAudioBaseSrc
	#     ...
        (mod_line, mod_num) = re.subn(r'GstBaseAudio(.+)', r'GstAudioBase\1', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GstBaseAudioSink -> GstAudioBaseSink; GstBaseAudioSrc -> GstAudioBaseSrc; ... */'
            modded = True

        ############################################################
        # GstAdapter
        ############################################################
	# gst_adapter_peek() is removed, use gst_adapter_map() and gst_adapter_unmap()
	# to get access to raw data from the adapter.
        m = re.search('gst_adapter_peek', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_adapter_peek() is removed, use gst_adapter_map() and gst_adapter_unmap() to get access to raw data from the adapter. */'
            modded = True        
        
	# Arguments changed from guint to gsize.
        m = re.search('GstAdapter', line)
        if m != None:
            mod_str=' /* MOD, FIXME: Arguments changed from guint to gsize. */'
            modded = True        

	# gst_adapter_prev_timestamp() is removed and should be replaced with
	# gst_adapter_prev_pts() and gst_adapter_prev_dts().
        m = re.search('gst_adapter_prev_timestam', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_adapter_prev_timestamp() is removed and should be replaced with gst_adapter_prev_pts() and gst_adapter_prev_dts(). */'
            modded = True

        ############################################################
        # GstBitReader, GstByteReader, GstByteWriter
        ############################################################
	# gst_*_reader_new_from_buffer(), gst_*_reader_init_from_buffer() removed, get
	# access to the buffer data with _map() and then use the _new() functions.
        m = re.search('gst_(.+)_reader_new_from_buffer', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_*_reader_new_from_buffer(), gst_*_reader_init_from_buffer() removed, get access to the buffer data with _map() and then use the _new() functions. */'
            modded = True
        m = re.search('gst_(.+)_reader_init_from_buffer', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_*_reader_new_from_buffer(), gst_*_reader_init_from_buffer() removed, get access to the buffer data with _map() and then use the _new() functions. */'
            modded = True

	# gst_byte_reader_new_from_buffer() and gst_byte_reader_init_from_buffer()
	# removed, get access to the buffer data and then use the _new() functions.
        m = re.search('gst_byte_reader_new_from_buffer', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_byte_reader_new_from_buffer() and gst_byte_reader_init_from_buffer() removed, get access to the buffer data and then use the _new() functions. */'
            modded = True
        m = re.search('gst_byte_reader_init_from_buffer', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_byte_reader_new_from_buffer() and gst_byte_reader_init_from_buffer() removed, get access to the buffer data and then use the _new() functions. */'
            modded = True

        ############################################################
        # GstCollectPads
        ############################################################
	# gst_collect_pads_read() removed, use _read_buffer() or _take_buffer() and
	# then use the memory API to get to the memory.        
        m = re.search('gst_collect_pads_read', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_collect_pads_read() removed, use _read_buffer() or _take_buffer() and then use the memory API to get to the memory. */'
            modded = True

        ############################################################
        # GstBaseSrc, GstBaseTransform, GstBaseSink
        ############################################################
	# GstBaseSrc::get_caps(), GstBaseTransform::transform_caps() and
	# GstBaseSink::get_caps() now take a filter GstCaps* parameter to
	# filter the caps and allow better negotiation decisions.
        m = re.search('(get_caps|transform_caps|get_caps)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstBaseSrc::get_caps(), GstBaseTransform::transform_caps() and GstBaseSink::get_caps() now take a filter GstCaps* parameter to filter the caps and allow better negotiation decisions. */'
            modded = True

        ############################################################
        # GstBaseSrc
        ############################################################
	# When overriding GstBaseTransform::fixate() one should chain up to the parent
	# implementation.
        m = re.search('fixrate', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstBaseSrc: When overriding GstBaseTransform::fixate() one should chain up to the parent implementation. */'
            modded = True

        ############################################################
        # GstBaseTransform
        ############################################################
	# GstBaseTransform::transform_caps() now gets the complete caps passed
	# instead of getting it passed structure by structure.
        m = re.search('transform_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstBaseTransform::transform_caps() now gets the complete caps passed instead of getting it passed structure by structure. */'
            modded = True        

	# GstBaseTransform::event() was renamed to sink_event(). The old function
	# uses the return value to determine if the event should be forwarded or not.
	# The new function has a default implementation that always forwards the event
	# and the return value is simply returned as a result from the event handler.
	# The semantics of the sink_event are thus the same as those for the src_event
	# function.        
        m = re.search('GstBaseTransform.*event', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstBaseTransform::event() was renamed to sink_event(). The old function uses the return value to determine if the event should be forwarded or not. The new function has a default implementation that always forwards the event and the return value is simply returned as a result from the event handler. The semantics of the sink_event are thus the same as those for the src_event function. */'
            modded = True

        ############################################################
        # GstImplementsInterface
        ############################################################
	# GstImplementsInterface has been removed. Interfaces need to be updated to either have
	# is_ready/usable/available() methods, or have GError arguments
	# to their methods so we can return an appropriate error if a
	# particular interface isn't supported for a particular device.
        m = re.search('GstImplementsInterface', line)
        if m != None:
            mod_str=" /* MOD, FIXME: GstImplementsInterface has been removed. Interfaces need to be updated to either have is_ready/usable/available() methods, or have GError arguments to their methods so we can return an appropriate error if a particular interface isn't supported for a particular device. */"
            modded = True

        ############################################################
        # GstIterator
        ############################################################
	# uses a GValue based API now that is similar to the 0.10 API but
	# allows bindings to properly use GstIterator and prevents complex
	# return value ownership issues.

        ############################################################
        # GstNavigationInterface
        ############################################################
	# Now part of the video library in gst-plugins-base, and the interfaces
	# library no longer exists.        
        m = re.search('GstNavigationInterface', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstNavigationInterface: Now part of the video library in gst-plugins-base, and the interfaces library no longer exists. */'
            modded = True

        ############################################################
        # GstMixerInterface / GstTunerInterface
        ############################################################
	# Removed - no replacement?
        m = re.search('(GstMixerInterface|GstTunerInterface)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstMixerInterface|GstTunerInterface: Removed - no replacement? */'
            modded = True

        ############################################################
        # GstXOverlay interface
        ############################################################
	# Renamed to GstVideoOverlay, and now part of the video library in
	# gst-plugins-base, as the interfaces library no longer exists.
        m = re.search('GstXOverlay', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstXOverlay: Renamed to GstVideoOverlay, and now part of the video library in gst-plugins-base, as the interfaces library no longer exists. */'
            modded = True

        ############################################################
        # GstPropertyProbe interface
        ############################################################
	# Removed - no replacement in 1.0.x and 1.2.x, but since 1.4 there is
	# a more featureful replacement for device discovery and feature querying,
	# provided by GstDeviceMonitor, GstDevice, and friends. See the
	# "GStreamer Device Discovery and Device Probing" documentation at
	# http://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-device-probing.html
        m = re.search('GstPropertyProbe', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstPropertyProbe: Removed - no replacement in 1.0.x and 1.2.x, but since 1.4 there is a more featureful replacement for device discovery and feature querying, provided by GstDeviceMonitor, GstDevice, and friends. See the "GStreamer Device Discovery and Device Probing" documentation at http://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-device-probing.html */'
            modded = True

        ############################################################
        # GstURIHandler
        ############################################################
	# gst_uri_handler_get_uri() and the get_uri vfunc now return a copy of
	# the URI string
        m = re.search('gst_uri_handler_get_uri', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_uri_handler_get_uri() and the get_uri vfunc now return a copy of the URI string */'
            modded = True

	# gst_uri_handler_set_uri() and the set_uri vfunc now take an additional
	# GError argument so the handler can notify the caller why it didn't
	# accept a particular URI.
        m = re.search(' gst_uri_handler_set_uri', line)
        if m != None:
            mod_str=" /* MOD, FIXME: gst_uri_handler_set_uri() and the set_uri vfunc now take an additional GError argument so the handler can notify the caller why it didn't accept a particular URI. */"
            modded = True

	# gst_uri_handler_set_uri() now checks if the protocol of the URI passed
	# is one of the protocols advertised by the uri handler, so set_uri vfunc
	# implementations no longer need to check that as well.
        m = re.search('gst_uri_handler_set_uri', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_uri_handler_set_uri() now checks if the protocol of the URI passed is one of the protocols advertised by the uri handler, so set_uri vfunc implementations no longer need to check that as well. */'
            modded = True

        ############################################################
        # GstTagList
        ############################################################
	# is now an opaque mini object instead of being typedefed to a GstStructure.

	# While it was previously okay (and in some cases required because of
	# missing taglist API) to cast a GstTagList to a GstStructure or use
	# gst_structure_* API on taglists, you can no longer do that. Doing so will
	# cause crashes.
        m = re.search('GstTagList.*GstStructure', line)
        if m != None:
            mod_str=' /* MOD, FIXME: While it was previously okay (and in some cases required because of missing taglist API) to cast a GstTagList to a GstStructure or use gst_structure_* API on taglists, you can no longer do that. Doing so will cause crashes. */'
            modded = True
        m = re.search('GstStructure.*GstTagList', line)
        if m != None:
            mod_str=' /* MOD, FIXME: While it was previously okay (and in some cases required because of missing taglist API) to cast a GstTagList to a GstStructure or use gst_structure_* API on taglists, you can no longer do that. Doing so will cause crashes. */'
            modded = True

	# Also, tag lists are refcounted now, and can therefore not be freely
	# modified any longer. Make sure to call
	# 
	#   taglist = gst_tag_list_make_writable (taglist);
	# 
	# before adding, removing or changing tags in the taglist.
        # ???

	# gst_tag_list_new() has been renamed to gst_tag_list_new_empty().
        (mod_line, mod_num) = re.subn(r'gst_tag_list_new', r'gst_tag_list_new_empty', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_tag_list_new() has been renamed to gst_tag_list_new_empty(). */'
            modded = True
        
	# gst_tag_list_new_full*() have been renamed to gst_tag_list_new*().
        (mod_line, mod_num) = re.subn(r'gst_tag_list_new_full', r'gst_tag_list_new', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_tag_list_new_full*() have been renamed to gst_tag_list_new*(). */'
            modded = True

        # gst_tag_list_free() has been replaced by gst_tag_list_unref().
        (mod_line, mod_num) = re.subn(r'gst_tag_list_free', r'gst_tag_list_unref', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_tag_list_free() has been replaced by gst_tag_list_unref(). */'
            modded = True

	# GST_TAG_IMAGE, GST_TAG_PREVIEW_IMAGE, GST_TAG_ATTACHMENT: many tags that
	# used to be of type GstBuffer are now of type GstSample (which is basically
	# a struct containing a buffer alongside caps and some other info).
        m = re.search('(GST_TAG_IMAGE|GST_TAG_PREVIEW_IMAGE|GST_TAG_ATTACHMENT)', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_TAG_IMAGE, GST_TAG_PREVIEW_IMAGE, GST_TAG_ATTACHMENT: many tags that used to be of type GstBuffer are now of type GstSample (which is basically a struct containing a buffer alongside caps and some other info). */'
            modded = True        

	# gst_tag_list_get_buffer() => gst_tag_list_get_sample()
        (mod_line, mod_num) = re.subn(r'gst_tag_list_get_buffer', r'gst_tag_list_get_sample', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_tag_list_get_buffer() => gst_tag_list_get_sample() */'
            modded = True

	# gst_is_tag_list() => GST_IS_TAG_LIST ()
        (mod_line, mod_num) = re.subn(r'gst_is_tag_list', r'GST_IS_TAG_LIST', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_is_tag_list() => GST_IS_TAG_LIST () */'
            modded = True

        ############################################################
        # GstController (TODO)
        ############################################################
	# has now been merged into GstObject. It does not exists as a individual
	# object anymore. In addition core contains a GstControlSource base class and
	# the GstControlBinding. The actual control sources are in the controller
	# library as before. The 2nd big change is that control sources generate
	# a sequence of gdouble values and those are mapped to the property type and
	# value range by GstControlBindings.
	# 
	# For plugins the effect is that gst_controller_init() is gone and
	# gst_object_sync_values() is taking a GstObject * instead of GObject *.
	# 
	# For applications the effect is larger. The whole gst_controller_* API is
	# gone and now available in simplified form under gst_object_*. ControlSources
	# are now attached via GstControlBinding to properties. There are no GValue
	# arguments used anymore when programming control sources.
	# 
	# A simple way to attach a ControlSource to a property is:
	# gst_object_add_control_binding (object,
	#   gst_direct_control_binding_new (object, property_name, control_source));
	#   
	# gst_controller_set_property_disabled ->
	#   gst_object_set_control_binding_disabled
	# 
	# gst_object_get_value_arrays has been removed. Loop over the controlled
	# properties fetch the value array. Also GstValueArray is gone. The fields of
	# GstValueArray are now passed directly to gst_object_get_value_array as
	# arguments.
	# 
	# GstInterpolationControlSource has been split. There is a new 
	# GstTimedValueControlSource baseclass and 2 sub classes: 
	# GstInterpolationControlSource and GstTriggerControlSource. The API for setting
	# and getting the timestamps is in GstTimedValueControlSource.
	# 
	# gst_interpolation_control_source_set_interpolation_mode() has been removed.
	# Set the "mode" gobject property on the control-source instead. The possible
	# enum values have been renamed from GST_INTERPOLATE_XXX to
	# GST_INTERPOLATION_MODE_XXX.

        ############################################################
        # GstRegistry
        ############################################################
        # gst_registry_get_default() -> gst_registry_get()
        (mod_line, mod_num) = re.subn(r'gst_registry_get_default', r'gst_registry_get', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_registry_get_default() -> gst_registry_get() */'
            modded = True
        
        # gst_default_registry_*(...) -> gst_registry_*(gst_registry_get(), ...)
        (mod_line, mod_num) = re.subn(r'gst_default_registry_(.+)\s*\(', r'gst_registry_\1(gst_registry_get(),', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_default_registry_*(...) -> gst_registry_*(gst_registry_get(), ...) */'
            modded = True

        ############################################################
        # GstValue
        ############################################################
	# GST_TYPE_DATE -> G_TYPE_DATE
        (mod_line, mod_num) = re.subn(r'GST_TYPE_DATE', r'G_TYPE_DATE', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_TYPE_DATE -> G_TYPE_DATE */'
            modded = True
        
	# GST_VALUE_HOLDS_DATE(value) -> G_VALUE_HOLDS(value,G_TYPE_DATE)
        (mod_line, mod_num) = re.subn(r'GST_VALUE_HOLDS_DATE\s*\((.+)\)', r'G_VALUE_HOLDS(\1,G_TYPE_DATE)', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_VALUE_HOLDS_DATE(value) -> G_VALUE_HOLDS(value,G_TYPE_DATE) */'
            modded = True
        else:
            m = re.search('GST_VALUE_HOLDS_DATE', line)
            if m != None:
                mod_str=' /* MOD, FIXME: GST_VALUE_HOLDS_DATE(value) -> G_VALUE_HOLDS(value,G_TYPE_DATE) */'
                modded = True

        # gst_value_set_date() -> g_value_set_boxed()
        (mod_line, mod_num) = re.subn(r'gst_value_set_date', r'g_value_set_boxed', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_value_set_date() -> g_value_set_boxed() */'
            modded = True
        
	# gst_value_get_date() -> g_value_get_boxed()        
        (mod_line, mod_num) = re.subn(r'gst_value_get_date', r'g_value_get_boxed', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: gst_value_get_date() -> g_value_get_boxed() */'
            modded = True

        ############################################################
        # GError/GstGError
        ############################################################
	# GstGError -> GError
        (mod_line, mod_num) = re.subn(r'GstGError', r'GError', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GstGError -> GError */'
            modded = True
        
	# GST_TYPE_G_ERROR / gst_g_error_get_type() -> G_TYPE_ERROR        
        (mod_line, mod_num) = re.subn(r'GST_TYPE_G_ERROR', r'G_TYPE_ERROR', line)
        if mod_num != 0:
            line = mod_line
            mod_str=' /* MOD: GST_TYPE_G_ERROR / gst_g_error_get_type() -> G_TYPE_ERROR */'
            modded = True
        m = re.search('gst_g_error_get_type', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GST_TYPE_G_ERROR / gst_g_error_get_type() -> G_TYPE_ERROR */'
            modded = True

        ############################################################
        # GstVideo
        ############################################################
	# GstXOverlay interface -> renamed to GstVideoOverlay, and now part of
	# the video library in gst-plugins-base, as the interfaces library
	# no longer exists.
        m = re.search('GstXOverlay', line)
        if m != None:
            mod_str=' /* MOD, FIXME: GstXOverlay interface -> renamed to GstVideoOverlay, and now part of the video library in gst-plugins-base, as the interfaces library no longer exists. */'
            modded = True

	# gst_video_format_parse_caps() -> use gst_video_info_from_caps() and
	#     then GstVideoInfo.        
        m = re.search('gst_video_format_parse_caps', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_video_format_parse_caps() -> use gst_video_info_from_caps() and then GstVideoInfo.  */'
            modded = True

        ############################################################
        # GstChildProxy
        ############################################################
	# gst_child_proxy_lookup() can no longer be called on GObjects that
	# do not implement the GstChildProxy interface. Use
	#   g_object_class_find_property (G_OBJECT_GET_CLASS (obj), "foo")
	# instead for non-childproxy objects.
        m = re.search('gst_child_proxy_lookup', line)
        if m != None:
            mod_str=' /* MOD, FIXME: gst_child_proxy_lookup() can no longer be called on GObjects that do not implement the GstChildProxy interface. Use g_object_class_find_property (G_OBJECT_GET_CLASS (obj), "foo") instead for non-childproxy objects. */'
            modded = True

        ############################################################
        # "codec-data" and "streamheader" field in GstCaps (not implemented yet!)
        ############################################################
	# codec-data and stream headers are no longer in GstCaps, but sent as
	# part of a STREAM CONFIG event (which should be sent after the initial
	# CAPS event if needed).
        # ???

        ############################################################
        #
        # soft changes
        #
        ############################################################
        
        # m = re.search('', line)
        # if m != None:
        #     mod_str=' /* MOD, FIXME:  */'
        #     modded = True
        
        # (mod_line, mod_num) = re.subn(r'', r'', line)
        # if mod_num != 0:
        #     line = mod_line
        #     mod_str=' /* MOD: */'
        #     modded = True

        
            
        # Does not match any rule above, just print
        if cut_line == False:
            if modded == False:
                print line
            else:
                if mod_str == False:
                    print line+mod_str_def
                else:
                    print line+mod_str

# print 'line_no: ', line_no


