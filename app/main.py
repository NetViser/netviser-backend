import importlib
import os
from fastapi import APIRouter, FastAPI
from app.routes.xai.xai_routes import router as xai_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Include the XAI router
app.include_router(xai_router)

origins = [
    "http://localhost:3000",  # Frontend URL
    # Add other origins if necessary
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

hide_router = ["_init.py"]

# For automatic route registration
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
    return {"message": "Healthcheck Passed"}
