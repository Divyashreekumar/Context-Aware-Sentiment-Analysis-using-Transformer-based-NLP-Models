from transformers import pipeline
from typing import List, Dict, Any
from pydantic  import BaseModel, Field

from enum import Enum
from transformers import AutoTokenizer



class predictModel(BaseModel):
    label : str = Field(...,description="The label of the text")
    score : float = Field(...,description="The score of the text")
saved_model = pipeline('text-classification',
                       model = 'intent classification1')

target_map = { 'LABEL_1': "Positive", 'LABEL_0': "Negative", 'LABEL_2': "Neutral","LABEL_3":"Sarcasm"}
# index2label = {v: k for k, v in target_map.items()}

def predict_text(text : str|List[str])->List[Dict[str, Any]]:
    # Predict the intent of the text
    prediction = saved_model(text)
    # print(prediction)
    # for idx,i in enumerate(prediction):
        # print(idx,i)
        # print(i["label"],i["score"])
    return [predictModel(label = target_map[pred["label"]], score = pred['score']) for pred in prediction]


