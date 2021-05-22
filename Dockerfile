FROM python:3.7
MAINTAINER daniilbond1

WORKDIR /usr/project

COPY . .

EXPOSE 8888
RUN pip install --no-cache-dir -r requirements.txt
CMD ["/usr/project/in.sh"]
