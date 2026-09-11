FROM ultralytics/ultralytics:8.3.161

WORKDIR /app
COPY train.py sku110k.yaml ./

ENTRYPOINT ["python", "train.py"]
