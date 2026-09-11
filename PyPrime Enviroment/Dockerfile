FROM condaforge/miniforge3:latest

COPY environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml && conda clean --all --yes

ENV PATH=/opt/conda/envs/PyPrimePortable/bin:$PATH
WORKDIR /workspace

COPY test_environment.py /tmp/test_environment.py
RUN python /tmp/test_environment.py

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
