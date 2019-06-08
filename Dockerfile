
FROM ubuntu:latest
RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas matplotlib seaborn plotly sklearn
COPY class7.py .
COPY class-svc.py .
COPY class7-kn.py .




CMD ["python3","-u","class7.py"]
CMD ["python3","-u","class-svc.py"]
CMD ["python3","-u","class7-kn.py"]




