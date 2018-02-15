FROM containers.ligo.org/lscsoft/lalsuite:nightly
RUN echo "Building GstLAL..."

LABEL name="GstLAL Runtime Debian" \
      maintainer="Alexander Pace <alexander.pace@ligo.org>" \
      date="20180108" \
      support="Reference Platform"

RUN apt-get update && apt-get install --assume-yes build-essential

# Install GstLAL dependencies:
RUN  apt-get --assume-yes install \
      doxygen \
      gtk-doc-tools \
      libfftw3-dev \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      liborc-0.4-0 \
      gobject-introspection \
      python-gobject-dev \
      python-numpy \
      python-scipy \
      lscsoft-gds

RUN rm /opt/lalsuite/lib/*.la

ENV PKG_CONFIG_PATH="/opt/gstlal/lib/pkgconfig${PKG_CONFIG_PATH}" \
    PATH="/opt/gstlal/bin:${PATH}" \
    GI_TYPELIB_PATH="/opt/gstlal/lib/girepository-1.0:${GI_TYPELIB_PATH}" \
    GST_PLUGIN_PATH="/opt/gstlal/lib/gstreamer-1.0:${GST_PLUGIN_PATH}" \
    PKG_CONFIG_PATH="/opt/gstlal/lib/pkgconfig:$/opt/gstlal/lib64/pkgconfig:${PKG_CONFIG_PATH}" \
    PYTHONPATH="/opt/gstlal/lib64/python2.7/site-packages:/opt/gstlal/lib/python2.7/site-packages:${PYTHONPATH}"

COPY gstlal /tmp/gstlal/
COPY gstlal-ugly /tmp/gstlal-ugly/
COPY gstlal-calibration /tmp/gstlal-calibration/
COPY gstlal-inspiral /tmp/gstlal-inspiral/
RUN cd /tmp/gstlal && ./00init.sh && ./configure --prefix=/opt/gstlal && make && make install
RUN cd /tmp/gstlal-ugly && ./00init.sh && ./configure --prefix=/opt/gstlal && make && make install
RUN cd /tmp/gstlal-calibration && ./00init.sh && ./configure --prefix=/opt/gstlal && make && make install
RUN cd /tmp/gstlal-inspiral && ./00init.sh && ./configure --prefix=/opt/gstlal && make && make install


ENTRYPOINT ["/bin/bash"]
