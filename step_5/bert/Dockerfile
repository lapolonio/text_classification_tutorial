FROM python:3.7

MAINTAINER Leonardo Apolonio
ADD . /code
WORKDIR /code
RUN pip install pipenv
RUN pipenv install
ADD asset/vocab.txt asset/vocab.txt
CMD ["pipenv", "run", "python", "run_app.py"]
