import importlib
import os
from fastapi import APIRouter, FastAPI

app = FastAPI()

hide_router = ["_init.py"]

# for automatic route registration
route_files = [file for file in os.listdir("app/routes") if file.endswith(".py")]
for file in route_files:
    if file in hide_router:
        continue
    module_name = f"app.routes.{file[:-3]}"
    module = importlib.import_module(module_name)

    if hasattr(module, "router") and isinstance(module.router, APIRouter):
        app.include_router(module.router)


@app.get("/")
async def root():
    return {"message": "Healtcheck Passed"}
