FROM python:3-alpine

MAINTAINER Leonardo Apolonio
ADD . /code
WORKDIR /code
RUN pip install pipenv
RUN pipenv install
ADD asset/vocab.txt asset/vocab.txt
CMD ["python", "run_app.py"]
