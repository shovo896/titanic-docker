# Titanic API (Docker)

Docker image: [shovo896/titanic-api](https://hub.docker.com/repository/docker/shovo896/titanic-api/general)

This container runs a FastAPI service that trains a Titanic survival model at startup and exposes prediction endpoints.

## Pull image

```bash
docker pull shovo896/titanic-api:latest
```

## Run container

```bash
docker run --rm -p 8000:8000 shovo896/titanic-api:latest
```

Then open:
- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Predict endpoint

`POST /predict`

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }'
```

Response includes:
- `survived` (boolean)
- `prediction` (0 or 1)
- `probability` (confidence score)
