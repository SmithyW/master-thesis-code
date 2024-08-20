FROM quay.io/jupyterhub/jupyterhub:5.1.0

ENV OAUTH2_AUTHORIZE_URL=https://auth.ecample.com/realms/mlops/protocol/openid-connect/auth
ENV OAUTH2_TOKEN_URL=https://auth.example.com/realms/mlops/protocol/openid-connect/token

COPY jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py
COPY requirements.txt .

RUN chmod 644 /etc/jupyterhub/jupyterhub_config.py
RUN pip install -r requirements.txt

CMD ["jupyterhub", "-f", "/etc/jupyterhub/jupyterhub_config.py"] 
