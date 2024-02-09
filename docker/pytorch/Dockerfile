FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG FFMPEG_HEADERS_VER=n12.0.16.1
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers \
    && git checkout ${FFMPEG_HEADERS_VER} \
    && make install

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
    yasm \
    libx264-dev \
    libgnutls28-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PREFIX=/usr/
ARG CCAP=75
ARG FFMPEG_VER=n4.4.2

RUN wget -q https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_VER}.tar.gz \
    && tar -xf ${FFMPEG_VER}.tar.gz \
    && cd FFmpeg-${FFMPEG_VER} \
    && ./configure \
    --prefix=$PREFIX \
    --extra-cflags='-I/usr/local/cuda/include' \
    --extra-ldflags='-L/usr/local/cuda/lib64' \
    --nvccflags="-gencode arch=compute_${CCAP},code=sm_${CCAP} -O2" \
    --disable-doc \
    --enable-decoder=aac \
    --enable-decoder=h264 \
    --enable-decoder=h264_cuvid \
    --enable-decoder=rawvideo \
    --enable-indev=lavfi \
    --enable-encoder=libx264 \
    --enable-encoder=h264_nvenc \
    --enable-demuxer=mov \
    --enable-muxer=mp4 \
    --enable-filter=scale \
    --enable-filter=testsrc2 \
    --enable-protocol=file \
    --enable-protocol=https \
    --enable-gnutls \
    --enable-shared \
    --enable-gpl \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-libx264 \
    --enable-nvenc \
    --enable-cuvid \
    --enable-nvdec \
    && make clean \
    && make -j$(nproc) \
    && make install

RUN git clone https://github.com/pytorch/audio \
    && cd audio \
    && git checkout v2.0.2 \
    && export BUILD_SOX=0 \
    && export BUILD_KALDI=0 \
    && export BUILD_RNNT=0 \
    && export USE_FFMPEG=1 \
    && export USE_ROCM=0 \
    && export USE_CUDA=1 \
    && python setup.py develop

COPY requirements.txt .
RUN python -m pip uninstall -y $(pip list --format=freeze | grep opencv) \
    && rm -rf /usr/local/lib/python3.8/dist-packages/cv2/ \
    && python -m pip install -r requirements.txt

ENTRYPOINT ["python"]
