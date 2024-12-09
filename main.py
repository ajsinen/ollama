import json
from typing import Optional
import ollama
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

# CHECK FOR OLLAMA
ollama_url = "http://localhost:11434/api/generate"


class Resp(BaseModel):
    message: list


class Req(BaseModel):
    prompt: Optional[str]


class ModelFile(BaseModel):
    model: str
    parameter: str
    system: str
    newModelName: str


@app.get("/")
async def default():
    return JSONResponse(content={"message": "Hello World! AI is Running"})


async def return_resp(message, retcode):
    response = {
        "response": message,
        "retcode": retcode
    }
    return response


@app.get("/test/ai")
async def ask_ai():
    print("asking Jarvis")
    ans = ollama.chat(
        model="jarvis",
        messages=[
            {
                "role": "user",
                "content": "what is your name"
            },
        ],
    )
    resp = ans.get('message').get('content')
    return resp


@app.post("/test/ai/v2")
async def ask_ai(rb: Req):
    print("asking Jarvis")
    ans = ollama.chat(
        model="jarvis",
        messages=[
            {
                "role": "user",
                "content": rb.prompt
            },
        ],
    )
    resp = ans.get('message').get('content')

    response = await return_resp(resp, 200)

    return Response(content=json.dumps(response), media_type="application/json", status_code=200)


# =================================================================================================
# === The Ollama Python Library's API is designed around the Ollama
# =================================================================================================

# == Ollama Generate Function Example
@app.get("/generate/sample")
async def ollama_generate_function():
    res = ollama.generate(
        model="jarvis",
        prompt="why is the sky blue",
    )
    print(ollama.show("llama3.2"))
    response = await return_resp(res.get('response'), 200)
    return Response(content=json.dumps(response), media_type="application/json")


@app.post("/create/modelfile")
async def create_modelfile(rb: ModelFile):
    modelfile = f"""
        FROM {rb.model}
        SYSTEM {rb.system}
        PARAMETER {rb.parameter}
    """
    ollama.create(model=f"{rb.newModelName}", modelfile=modelfile)
    return await return_resp("successfully created", 200)


@app.get("/ollama/list")
async def create_modelfile():
    list = ollama.list()

    models = list.get('models', [])
    list = []
    for model in models:
        name = model.get('model')
        list.append(name)
    resp = await return_resp(list, 200)
    return Response(content=json.dumps(resp), media_type="application/json")


@app.delete("/ollama/delete/model")
async def delete_ollama_model(model_name: str):
    ollama.delete(model_name)
    resp = await return_resp(f"successfully deleted {model_name}", 200)
    return resp
