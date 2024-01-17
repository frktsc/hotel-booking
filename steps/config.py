from pydantic import BaseModel
from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    model_name: str = "LogisticRegression"
