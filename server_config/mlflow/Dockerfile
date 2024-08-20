FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY .mlflow.env .
COPY requirements.txt .
COPY startup_mlflow.sh .

RUN chmod +x ./startup_mlflow.sh

# Microsoft ODBC Driver Installation
ENV ACCEPT_EULA=Y

RUN apt update && apt install -y curl

RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc
RUN curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list
RUN apt update
RUN ACCEPT_EULA=${ACCEPT_EULA} apt install -y msodbcsql18 mssql-tools18
RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc

# Install required python packages
RUN pip install -r requirements.txt

CMD ["/bin/bash", "./startup_mlflow.sh"]

EXPOSE 5000
