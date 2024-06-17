FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY with_parameter.py .

RUN pip install pandas numpy scikit-learn pyarrow

ENTRYPOINT ["python", "with_parameter.py"]
