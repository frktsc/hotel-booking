from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "LogisticRegression"
