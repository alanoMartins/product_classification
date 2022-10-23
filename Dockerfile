FROM tensorflow/serving:latest

WORKDIR /

COPY models/models.config /models/models.config
COPY models/bert /models/bert/1/

CMD ["--model_config_file=/models/models.config", "--allow_version_labels_for_unavailable_models=true"]