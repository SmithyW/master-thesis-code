FROM python:3.11.9

ARG JUPYTERHUB_VERSION=5.1.0

RUN pip install --upgrade pip
RUN pip install jupyterhub==$JUPYTERHUB_VERSION
RUN pip install \
jupyterlab-git \
notebook \
mlflow \
mlflow-oauth-keycloak-auth \
pandas \
numpy \
dvc \
prefect

RUN useradd -m jovyan
ENV HOME=/home/jovyan
WORKDIR $HOME
USER jovyan

COPY --chown=jovyan:jovyan ./jupyter_server_config.py /home/jovyan/.jupyter/
RUN mkdir /home/jovyan/work
RUN chown -R jovyan:jovyan /home/jovyan/work 

CMD ["jupyterhub-singleuser"]
