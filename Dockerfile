# syntax=docker/dockerfile:experimental
FROM quay.io/unstructured-io/base-images:rocky8.7-3 as base

ARG PIP_VERSION

# Set up environment
ENV HOME /home/
WORKDIR ${HOME}
RUN mkdir ${HOME}/.ssh && chmod go-rwx ${HOME}/.ssh \
  &&  ssh-keyscan -t rsa github.com >> /home/.ssh/known_hosts
ENV PYTHONPATH="${PYTHONPATH}:${HOME}"
ENV PATH="/home/usr/.local/bin:${PATH}"

FROM base as deps
# Copy and install Unstructured
COPY requirements requirements

RUN python3.8 -m pip install pip==${PIP_VERSION} && \
  dnf -y groupinstall "Development Tools" && \
  pip install --no-cache -r requirements/base.txt && \
  pip install --no-cache -r requirements/test.txt && \
  pip install --no-cache -r requirements/dev.txt && \
  pip install "unstructured.PaddleOCR" && \
  dnf -y groupremove "Development Tools" && \
  dnf clean all

FROM deps as code
ARG PACKAGE_NAME=unstructured_inference
COPY unstructured_inference unstructured_inference

#CMD ["pytest -m \"not slow\" test_${PACKAGE_NAME} --cov=${PACKAGE_NAME} --cov-report term-missing"]
CMD ["/bin/bash"]
#CMD ["bash -c pytest test_unstructured_inference"]
