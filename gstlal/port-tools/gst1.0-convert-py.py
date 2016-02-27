#!/usr/bin/python2.7
# implement replacements as listed on
# https://wiki.ubuntu.com/Novacut/GStreamer1.0

# Usage, e.g. from gstlal/python directory:
# find . ! -name '*.py' -print0 | xargs -0 -n 1 -I {} bash -c 'file {} | grep -i python | cut -d: -f1' | xargs -n 1 -I {} bash -c 'echo {}; ../port-tools/gst1.0-convert-py.py {} > {}.tmp; ! diff -q {} {}.tmp && mv {}.tmp {}; rm -f {}.tmp'; find . -name '*.py' -print0 | xargs -0 -n 1 -I {} bash -c 'echo {}; ../port-tools/gst1.0-convert-py.py {} > {}.tmp; ! diff -q {} {}.tmp && mv {}.tmp {}; rm -f {}.tmp'

# From Chad:
#* gst.event_new_seek() -> Gst.Event.new_seek()
# gst.Pipeline("blah") -> Gst.Pipeline(name="blah")
# gst.element_factory_make() -> Gst.ElementFactory.make()
#* gst.STATE_PLAYING -> gst.State.PLAYING (and similarly for others)
#* gst.STATE_CHANGE_FAILURE -> Gst.StateChangeReturn.FAILURE
#* gst.MESSAGE_EOS -> Gst.MessageType.EOS
# gst.Caps(str) -> Gst.Caps.from_string(str)
# gst.FORMAT_TIME -> Gst.Format.TIME
#* gst.SEEK_FLAG_FLUSH -> Gst.SeekFlags.FLUSH
#* gst.SEEK_TYPE_SET -> Gst.SeekType.SET
 

import re
import sys

line_no = 0
gobject_gst_lineno_diff = 5
import_gobject_line_no = -1
import_gst_line_no = -1
mod_str_def=' # MOD'
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

        # Is this just a comment? If so, leave as is
        m = re.search('^\s*\#', line)
        if m != None:
            print line
            continue

        # Remove import pygst
        m = re.search('import\s+pygst', line)
        if m != None:
            # print 'Found import gobject in line: [', line, ']'
            cut_line = True
        m = re.search('pygst\s*\.\s*require', line)
        if m != None:
            # print 'Found import gobject in line: [', line, ']'
            cut_line = True
            

        # Imports
        # Replace:
        #     import gobject
        #     import gst
        # With:
        #     import gi
        #     gi.require_version('Gst', '1.0')
        #     from gi.repository import GObject, Gst
        m = re.search('import\s+gobject', line)
        if m != None:
            # print 'Found import gobject in line: [', line, ']'
            import_gobject_line_no = line_no
            cut_line = True
            
        m = re.search('import\s+gst', line)
        if m != None:
            # print 'Found import gst in line: [', line, ']'
            import_gst_line_no = line_no
            cut_line = True
            
        if import_gobject_line_no > 0 and import_gst_line_no > 0:
            # print '** Found both'
            import_gobject_line_no = -1
            import_gst_line_no = -1
            print 'import gi'
            print "gi.require_version('Gst', '1.0')"
            print 'from gi.repository import GObject, Gst'
            gobject_loaded=True
            if gobject_postload != False:
                print gobject_postload

        if import_gobject_line_no > 0 and (line_no - import_gobject_line_no) > 5:
            mod_str=' # MOD: Error: found import gobject but expected import gst to be within ['+str(gobject_gst_lineno_diff)+'] lines.'
            modded=True


        # Gst.init()
        # You should call GObject.threads_init() in your module-scope, right after your imports. Unlike the static bindings, you also need to call Gst.init(). So replace:
        #     gobject.threads_init()
        # With:
        #     GObject.threads_init()
        #     Gst.init(None)
        m = re.search('^([ \t]*)gobject\s*\.\s*threads_init\s*\(\s*\)', line)
        if m != None:
            if gobject_loaded == True:
                print m.group(1) + 'GObject.threads_init()'
                print m.group(1) + 'Gst.init(None)'
            else:
                gobject_postload=m.group(1) + 'GObject.threads_init()'+'\n'+m.group(1) + 'Gst.init(None)'
            continue


        # element_factory_make()
        # Replace:
        #    src = gst.element_factory_make('filesrc')
        # With:
        #    src = Gst.ElementFactory.make('filesrc', None)
        # Note that unlike element_factory_make(), ElementFactory.make() will return None rather than raising an exception when the element is not found.
        #
        # Note: only add ", None" if there is only argument (only)
        m = re.search('element_factory_make\s*\((.+)\)', line)
        if m != None:
            # is there more than one argument there?
            m2 = re.search('\,', m.group(1))
            # Yes: more than one argument
            if m2 != None:
                (mod_line, mod_num) = re.subn(r'element_factory_make', r"ElementFactory.make", line)
                if mod_num != 0:
                    line = mod_line
                    modded=True
            # No: only one argument, add a ", None"
            else:
                (mod_line, mod_num) = re.subn(r'element_factory_make\s*\((.+)\)', r"ElementFactory.make(\1, None)", line)
                if mod_num != 0:
                    line = mod_line
                    # mod_str = ' # MOD: Added ", None" to the argument list'
                    modded=True

        # gst.event_new_seek() -> Gst.Event.new_seek()
        # (from Chad)
        (mod_line, mod_num) = re.subn(r'event_new_seek\s*\((.*)\)', r"Event.new_seek(\1)", line)
        if mod_num != 0:
            line = mod_line
            modded = True


        # element_link_many()
        # There is no equivalent to gst.element_link_many(), so replace:
        #    gst.element_link_many(one, two, three)
        # With:
        #    one.link(two)
        #    two.link(three)
        m = re.search('element_link_many', line)
        if m != None:
            mod_str=' # MOD: Error line ['+str(line_no)+']: element_link_many not yet implemented. See web page **'
            modded=True

        # Pipeline.add(one, two)
        # Bin.add() can't be overloaded to add multiple elements at once, so replace:
        #    pipeline = gst.Pipeline()
        #    pipeline.add(one, two)
        # With:
        #    pipeline = Gst.Pipeline()
        #    pipeline.add(one)
        #    pipeline.add(two)
        m = re.search('\.add\s*\((.+)\)', line)
        if m != None:
            # is there more than one argumen there?
            m2 = re.search('\,', m.group(1))
            if m2 != None:
                mod_str=' # MOD: Found an add with multiple args: ['+str(line)+' ], args = ['+str(m.group(1))+']'
                modded=True

        # one.link(two, mycaps)
        # Replace:
        #    one.link(two, mycaps)
        # With:
        #    one.link_filtered(two, mycaps)
        m = re.search('\.\s*link\s*\((.+)\)', line)
        if m != None:
            # do caps appear here?
            m2 = re.search('caps', m.group(1))
            if m2 != None:
                (mod_line, mod_num) = re.subn(r'link', r"link_filtered", line)
                if mod_num != 0:
                    line = mod_line
                    modded = True

        # From Chad:
        # gst.STATE_CHANGE_FAILURE -> Gst.StateChangeReturn.FAILURE
        (mod_line, mod_num) = re.subn(r'STATE_CHANGE_FAILURE', r'StateChangeReturn.FAILURE', line)
        if mod_num != 0:
            line = mod_line
            modded = True
        (mod_line, mod_num) = re.subn(r'STATE_CHANGE_SUCCESS', r'StateChangeReturn.SUCCESS', line)
        if mod_num != 0:
            line = mod_line
            modded = True


        # STATE_PLAYING
        # Replace:
        #     pipeline.set_state(gst.STATE_PLAYING)
        # With:
        #     pipeline.set_state(Gst.State.PLAYING)
        # And so on. Some search-and-replace is helpful here, basically:
        #     gst.STATE_* => Gst.State.*
        (mod_line, mod_num) = re.subn(r'STATE_', r'State.', line)
        if mod_num != 0:
            line = mod_line
            modded = True


        # From Chad:
        # gst.MESSAGE_EOS -> Gst.MessageType.EOS
        (mod_line, mod_num) = re.subn(r'MESSAGE_EOS', r'MessageType.EOS', line)
        if mod_num != 0:
            line = mod_line
            modded = True


        # SEEK_FLAG_FLUSH
        # Replace something like this:
        #     pipeline.seek_simple(
        #         gst.FORMAT_TIME,
        #         gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT,
        #         nanoseconds
        #     )
        # With this:
        #     pipeline.seek_simple(
        #         Gst.Format.TIME,
        #         Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
        #         nanoseconds
        #     )
        # Search and replace:
        #     gst.SEEK_FLAG_* => Gst.SeekFlags.*
        (mod_line, mod_num) = re.subn(r'SEEK_FLAG_', r'SeekFlags.', line)
        if mod_num != 0:
            line=mod_line
            modded=True


        # From Chad:
        # gst.SEEK_TYPE_SET -> Gst.SeekType.SET
        (mod_line, mod_num) = re.subn(r'SEEK_TYPE_', r'SeekType.', line)
        if mod_num != 0:
            line=mod_line
            modded=True


        # FORMAT_TIME
        # Search and replace:
        #     gst.FORMAT_* => Gst.Format.*
        (mod_line, mod_num) = re.subn(r'gst\.FORMAT_', r'Gst.Format.', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # GhostPad
        # You've got to use GhostPad.new(). So replace:
        #     ghost = gst.GhostPad('sink', mysinkpad)
        # With:
        #     ghost = Gst.GhostPad.new('sink', mysinkpad)
        (mod_line, mod_num) = re.subn(r'gst\s*\.\s*GhostPad', r'Gst.GhostPad.new', line)
        if mod_num != 0:
            line = mod_line
            modded = True

        # Pad.get_caps()
        # In callbacks for "pad-added" events and similar, it's common to use the string representation of the pad caps as a way to decide whether to link a pad and what to link the pad to.
        # You need to use Pad.query_caps() instead of Pad.get_caps(), and the returned object is no longer array-like. So in general replace this pattern:
        #     def on_pad_added(element, pad):
        #        string = pad.get_caps()[0].get_name()
        #        if string.startswith('audio/'):
        #            <link to some audio elements>
        # With this:
        #     def on_pad_added(element, pad):
        #        string = pad.query_caps(None).to_string()
        #        if string.startswith('audio/'):
        #            <link to some audio elements>
        (mod_line, mod_num) = re.subn(r'\.\s*get_caps\s*[\(\)]*\s*\[\s*0\s*\]', r'.query_caps(None)', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        (mod_line, mod_num) = re.subn(r'\.\s*get_caps\s*[\(\)]*', r'.query_caps(None)', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # Element.get_pad()
        # Element.get_pad() has been renamed Element.get_static_pad(). So replace something like this:
        #     src = gst.element_factory_make('filesink')
        #     pad = src.get_pad('sink')
        # With this:
        #     src = Gst.ElementFactory.make('filesink', None)
        #     pad = src.get_static_pad('sink')
        (mod_line, mod_num) = re.subn(r'\.\s*get_pad\s*\(', r'.get_static_pad(', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # Buffer.data
        # In a "preroll-handoff" or "handoff" callback, it's common to want access to the buffer data. A common use case is saving a JPEG thumbnail to a file, for example.
        # Replace something like this:
        #     def on_preroll_handoff(element, buf, pad):
        #         data = buf.data
        # With this:
        #     def on_preroll_handoff(element, buf, pad):
        #         data = buf.extract_dup(0, buf.get_size())
        # ??? How to implement ???

        # Buffer.timestamp
        # Buffer.timestamp has been renamed to Buffer.pts (Presentation Time Stamp). So replace something like this:
        #     def on_handoff(element, buf, pad):
        #         timestamp = buf.timestamp
        # With this:
        #     def on_handoff(element, buf, pad):
        #         timestamp = buf.pts
        # There is also the new Buffer.dts (Decode Time Stamp). Often pts and dts will be the same, but B-frames are an example of when they'll be different. 
        (mod_line, mod_num) = re.subn(r'\.\s+timestamp', r'.pts', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # Pad.get_negotiated_caps()
        # Pad.get_negotiated_caps() has been renamed Pad.get_current_caps(), so replace:
        #     caps = pad.get_negotiated_caps()
        # With:
        #     caps = pad.get_current_caps()
        (mod_line, mod_num) = re.subn(r'get_negotiated_caps', r'get_current_caps', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # caps[0]
        # Caps objects are not array-like from PyGI, so replace:
        #     cap = caps[0]
        # With:
        #     cap = caps.get_structure(0)
        # ???


        # caps[0]['foo']
        # Individual capability structures are not dictionary-like objects from PyGI, so you need to use type-appropriate accessor methods instead. For example, replace this:
        #     framerate = caps[0]['framerate']
        #     width = caps[0]['width']
        #     height = caps[0]['height']
        # With:
        #     (success, num, denom) = caps.get_structure(0).get_fraction('framerate')
        #     (success, width) = caps.get_structure(0).get_int('width')
        #     (success, height) = caps.get_structure(0).get_int('height')
        # ???

        # query_new_duration()
        # Replace:
        #     query = gst.query_new_duration(Gst.FORMAT_TIME)
        # With:
        #     query = Gst.Query.new_duration(Gst.Format.TIME)
        (mod_line, mod_num) = re.subn(r'gst\s*\.\*query_new_duration', r'Gst.Query.new_duration', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # audio/x-raw, video/x-raw
        # "audio/x-raw-int" and "audio/x-raw-float" have been condensed into a unified "audio/x-raw" with a flexible format description.
        # Likewise, "video/x-raw-yuv" and "video/x-raw-rgb" have been condensed into a unified "video/x-raw" with a flexible format description.
        # This is much nicer because in GStreamer 0.10 certain element details were leaked that really shouldn't have been. For example, if you wanted filter caps to force a certain sample-rate, you previously needed to know whether, say, an audio encoder took audio/x-raw-int or audio/x-raw-float.
        # So now you can replace say:
        #     caps = gst.caps_from_string('audio/x-raw-float, rate=(int)44100')
        # With:
        #     caps = Gst.caps_from_string('audio/x-raw, rate=(int)44100')
        # As you're only interested in specifying the rate anyway, it's much nicer to be truly abstracted from the format details. 
        (mod_line, mod_num) = re.subn(r'x-raw-int', r'x-raw', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        (mod_line, mod_num) = re.subn(r'x-raw-float', r'x-raw', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # element_register()
        # The gst.element_register() function had some magic to create an in-process plugin on the fly if needed (assuming Jason understood this detail correctly when talking with Edward Hervey about it).
        # You can still register in-process elements with PyGI, but you need to use a bit more of the C API now and explicitly register your plugin, and then register your elements. So replace something like this:
        #     gobject.type_register(DmediaSrc)
        #     gst.element_register(DmediaSrc, 'dmediasrc')
        # With something like this:
        #     def plugin_init(plugin, userarg):
        #         DmediaSrcType = GObject.type_register(DmediaSrc)
        #         Gst.Element.register(plugin, 'dmediasrc', 0, DmediaSrcType)
        #         return True
        #     version = Gst.version()
        #     Gst.Plugin.register_static_full(
        #         version[0],  # GST_VERSION_MAJOR
        #         version[1],  # GST_VERSION_MINOR
        #         'dmedia',
        #         'dmedia src plugin',
        #         plugin_init,
        #         '12.06',
        #         'LGPL',
        #         'dmedia',
        #         'dmedia',
        #         'https://launchpad.net/dmedia',
        #         None,
        #     )
        m = re.search('type_register', line)
        if m != None:
            mod_str=' # MOD: Found type_register in line: ['+str(line)+']'
            modded=True

        m = re.search('element_register', line)
        if m != None:
            modstr=' # MOD: Found element_register in line: ['+ str(line)+ ']'
            modded=True

        # PAD_SRC, PAD_ALWAYS
        # Search and replace:
        #     gst.PAD_SRC => Gst.PadDirection.SRC
        #     gst.PAD_SINK => Gst.PadDirection.SINK
        #     gst.PAD_ALWAYS => Gst.PadPresence.ALWAYS
        #     gst.PAD_REQUEST => Gst.PadPresence.REQUEST
        #     gst.PAD_SOMETIMES => Gst.PadPresence.SOMETIMES
        (mod_line, mod_num) = re.subn(r'\.\s*PAD_SRC', r'.PadDirection.SRC', line)
        if mod_num != 0:
            line=mod_line
            modded=True
        (mod_line, mod_num) = re.subn(r'\.\s*PAD_SINK', r'.PadDirection.SINK', line)
        if mod_num != 0:
            line=mod_line
            modded=True
        (mod_line, mod_num) = re.subn(r'\.\s*PAD_ALWAYS', r'.PadPresence.ALWAYS', line)
        if mod_num != 0:
            line=mod_line
            modded=True
        (mod_line, mod_num) = re.subn(r'\.\s*PAD_REQUEST', r'.PadPresence.REQUEST', line)
        if mod_num != 0:
            line=mod_line
            modded=True
        (mod_line, mod_num) = re.subn(r'\.\s*PAD_SOMETIMES', r'.PadPresence.SOMETIMES', line)
        if mod_num != 0:
            line=mod_line
            modded=True
        
        # URIHandler
        # With PyGST, you can do something like this to implement the GstUriHandler interface in a custom element:
        #     class DmediaSrc(gst.Bin, gst.URIHandler):
        #         @classmethod
        #         def do_get_type_full(cls):
        #             return gst.URI_SRC
        #         @classmethod
        #         def do_get_protocols_full(cls):
        #             return ['dmedia']
        #         def do_set_uri(self, uri):
        #             if not uri.startswith('dmedia://'):
        #                 return False
        #             self.uri = uri
        #             return True
        #         def do_get_uri(self):
        #             return self.uri
        # (Note many details were left out above, see plugin-0.10 for the full working example.)
        # Currently it seems this isn't possible with PyGI + GStreamer 1.0. See bug 679181.
        # m = re.search('URI', line)
        # if m != None:
        #     mod_str=' MOD: Found URI in line: ['+line+']'
        #     modded=True

        # decodebin2
        # The "decodebin2" element has been renamed to "decodebin", and the old "decodebin" element has been removed. 
        (mod_line, mod_num) = re.subn(r'decodebin2', r'decodebin', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # playbin2
        # The "playbin2" element has been renamed to "playbin", and the old "playbin" element has been removed. 
        (mod_line, mod_num) = re.subn(r'playbin2', r'playbin', line)
        if mod_num != 0:
            line=mod_line
            modded=True

        # ffmpegcolorspace
        # The "ffmpegcolorspace" element has been replaced with the new "videoconvert" element. 
        m = re.search('ffmpegcolorspace', line)
        if m != None:
            mod_str=' # MOD: Found ffmpegcolorspace in line: ['+str(line)+']'
            modded=True


        # Still any gst's? Change them to Gst
        (mod_line, mod_num) = re.subn(r'gst\s*\.', r'Gst.', line)
        if mod_num != 0:
            line = mod_line
            modded = True
            
        # Still any gst's? Change them to Gst
        # (mod_line, mod_num) = re.subn(r'gst([^l])', r'Gst\1', line)
        # if mod_num != 0:
        #     # Wait: if this line has an "import", then do not do anything
        #     m = re.search('import', line)
        #     if m == None:
        #         # Also: do not modify if comes after a comment
        #         m = re.search('\#.*gst([^l])', line)
        #         if m == None:
        #             # And do not change if appears in quotation marks
        #             m = re.search('"\s*gst([^l])\s*"', line)
        #             if m == None:
        #                 line = mod_line
        #                 modded = True

        # Still any gobject's? Change them to GObject
        (mod_line, mod_num) = re.subn(r'gobject\s*\.', r'GObject.', line)
        if mod_num != 0:
            line = mod_line
            modded = True
        
        # Still any gobject's? Change them to GObject
        # (mod_line, mod_num) = re.subn(r'gobject', r'GObject', line)
        # if mod_num != 0:
        #     # Wait: if this line has an "import", then do not do anything
        #     m = re.search('import', line)
        #     if m == None:
        #         # Also: do not modify if comes after a comment
        #         m = re.search('\#.*gobject', line)
        #         if m == None:
        #             # And to not change if appears in quotation marks
        #             m = re.search('"\s*gobject\s*"', line)
        #             if m == None:
        #                 line = mod_line
        #                 modded = True

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
