# Fixed OCR engine for Wajib DocBench. Build: docker build -t wajib-docbench-ocr:1 ocr/
# Base image pinned by digest (ubuntu:24.04 as pulled 2026-11-20). Tesseract comes from apt
# (expected 5.3.4); the language data is tessdata_best at a pinned commit, not the apt packs,
# which track the less accurate tessdata_fast. Every download is checked by sha256.
FROM ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289

ARG TESSDATA_COMMIT=e12c65a915945e4c28e237a9b52bc4a8f39a0cec
ARG ARA_SHA256=ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896
ARG ENG_SHA256=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/opt/tessdata
RUN mkdir -p "$TESSDATA_PREFIX" \
    && curl -fsSL -o "$TESSDATA_PREFIX/ara.traineddata" \
       "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_COMMIT}/ara.traineddata" \
    && curl -fsSL -o "$TESSDATA_PREFIX/eng.traineddata" \
       "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_COMMIT}/eng.traineddata" \
    && echo "${ARA_SHA256}  $TESSDATA_PREFIX/ara.traineddata" | sha256sum -c - \
    && echo "${ENG_SHA256}  $TESSDATA_PREFIX/eng.traineddata" | sha256sum -c - \
    && echo "$TESSDATA_COMMIT" > "$TESSDATA_PREFIX/COMMIT"

# The repo is mounted at /work at run time; the entrypoint is tesseract itself.
WORKDIR /work
ENTRYPOINT ["tesseract"]
