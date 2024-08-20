from subprocess import check_call
from oauthenticator.generic import GenericOAuthenticator
import os

c = get_config()  # noqa

c.Application.log_level = 'INFO'

c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = os.environ["DOCKER_NOTEBOOK_IMAGE"]

network_name = os.environ["DOCKER_NETWORK_NAME"]
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.network_name = network_name

notebook_dir = os.environ.get('DOCKER_NOTEBOOK_DIR', '/home/jovyan/work')
c.DockerSpawner.notebook_dir = notebook_dir
c.DockerSpawner.volumes = {'jupyterhub-user-{username}': notebook_dir,
                           'jupyterhub-config-{username}': '/home/jovyan/.jupyter'}
c.DockerSpawner.remove = True

c.JupyterHub.hub_ip = "jupyterhub"
c.JupyterHub.hub_port = 8080

c.DockerSpawner.mem_limit = '3G'

# Konfiguration von Keycloak als OAuth Provider
c.JupyterHub.authenticator_class = GenericOAuthenticator
c.GenericOAuthenticator.client_id = 'jupyterhub'
c.GenericOAuthenticator.client_secret = ''
c.GenericOAuthenticator.token_url = 'https://auth.example.com/realms/mlops/protocol/openid-connect/token'
c.GenericOAuthenticator.userdata_url = 'https://auth.example.com/realms/mlops/protocol/openid-connect/userinfo'
c.GenericOAuthenticator.userdata_params = {'state': 'state'}
c.GenericOAuthenticator.username_key = 'preferred_username'
c.GenericOAuthenticator.login_service = 'Keycloak'
c.GenericOAuthenticator.oauth_callback_url = 'http://jupyterhub.example.com/hub/oauth_callback'
c.GenericOAuthenticator.authorize_url = 'https://auth.example.com/realms/mlops/protocol/openid-connect/auth'
c.GenericOAuthenticator.logout_redirect_url = 'https://auth.example.com/realms/mlops/protocol/openid-connect/logout?redirect_url=http://jupyterhub.example.com/'
c.GenericOAuthenticator.scope = ['openid']
c.GenericOAuthenticator.allow_all = True
c.GenericOAuthenticator.auto_login = True

# Netwerkbezogene Konfiguration
c.JupyterHub.bind_url = 'http://0.0.0.0:8000'
c.JupyterHub.proxy_class = 'jupyterhub.proxy.ConfigurableHTTPProxy'
c.JupyterHub.public_url = 'https://jupyterhub.example.com'

# Anlegen neuer Benutzer beim erstmaligen Anmelden


def pre_spawn_hook(spawner):
    username = spawner.user.name
    try:
        check_call(['useradd', '-ms', '/bin/bash', username])
    except Exception as e:
        print(f'{e}')


c.DockerSpawner.pre_spawn_hook = pre_spawn_hook
