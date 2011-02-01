all: gab-rtsp-server

gab-rtsp-server: gab-rtsp-server.c
	gcc $(shell pkg-config --cflags --libs gst-rtsp-server-0.10) $^ -o $@

clean:
	rm -f gab-rtsp-server
