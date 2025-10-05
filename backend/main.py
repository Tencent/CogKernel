import uvicorn
import os
import argparse
import shutil
import uuid
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from cognitive_kernel.cognitive_kernel import (
    CognitiveKernel,
    FILE_LOCATIONS,
    GLOBAL_DB_LOCATIONS,
    DB_LOCATIONS,
    CHARACTER_POOL_PATH,
    CUSTOMIZED_CHARACTER_POOL_PATH,
    TOP_K_SENTENCES,
)
import auth
from database import (
    update_or_create_session,
    get_sessions_by_username,
    get_session_by_id,
    archive_session_by_id,
    get_rawdata_by_message_id,
    update_or_create_annotation,
    get_all_annotations,
    get_anno_by_username_and_date_range,
    get_annotation_counts_by_username,
)
from cognitive_kernel.file_loader import read_file_content
from starlette.websockets import WebSocketDisconnect
import asyncio

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Config ---
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_local_module", type=int, default=1000)
    parser.add_argument("--local_module_path", type=str, default="/app/Database_local/")
    parser.add_argument("--log_path", type=str, default="/app/Logs/")
    parser.add_argument(
        "--global_config_file_path",
        type=str,
        default="/app/cognitive_kernel/memory_kernel/knowledge_engine_config/global_configs.jsonl",
    )
    parser.add_argument(
        "--default_config_file_path",
        type=str,
        default="/app/cognitive_kernel/memory_kernel/knowledge_engine_config/default_configs.json",
    )
    return parser.parse_args()

args = get_args()

# --- Constants ---
SERVICE_URLS = json.load(
    open(os.environ.get("KR_SERVICE_IP_FILE", "/app/service_url_config.json"), "r")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "ck")
ACTIVATE_KE = os.environ.get("ACTIVATE_KE", "True").lower() in ("true", "1", "t")
ACTIVATE_SHORT_FEEDBACK = os.environ.get("ACTIVATE_SHORT_FEEDBACK", "True").lower() in (
    "true", "1", "t"
)
UPLOAD_DIR = "/app/static/uploads"
MAX_CUSTOMIZED_CHARACTER = int(os.environ.get("MAX_CUSTOMIIZED_CHARACTER", 1))
SERVICE_IP = os.environ.get("SERVICE_IP", "127.0.0.1:8081")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))

# --- Models ---
class AnnotationModel(BaseModel):
    session_id: str
    message_id: str
    username: str
    old_message: str
    suggestion: str
    annotations: str
    created_time: str
    updated_time: str

class ErrorResponse(BaseModel):
    status: str
    error: Optional[str] = None

class SuccessResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None

# --- FastAPI App ---
app = FastAPI()
app.include_router(auth.router)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
app.mount("/avatar", StaticFiles(directory="/app/static/avatar"), name="avatars")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- Cognitive Kernel ---
current_cognitive_kernel = CognitiveKernel(
    args,
    memory_inference_urls=SERVICE_URLS,
    model_name=MODEL_NAME,
    service_ip=SERVICE_IP,
)

# --- WebSocket ---
async def generate_messages(websocket: WebSocket, data_info: dict, mode: str):
    try:
        async for message in current_cognitive_kernel.generate_for_demo(
            messages=data_info["messages"],
            CKStatus=data_info["CKStatus"],
            username=data_info["username"],
            message_id=data_info["currentMessageId"],
            mode=mode,
        ):
            await asyncio.sleep(0.001)
            await websocket.send_text(message)
        await websocket.send_text("[save_message]")
    except asyncio.CancelledError:
        logger.info("Message generation cancelled.")
        await websocket.send_text("[task_cancelled]")
    except Exception as e:
        logger.error(f"Error in message generation: {e}")
        await websocket.send_text(f"[error]{str(e)}")

@app.websocket("/setup_ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    message_task = None
    try:
        while True:
            try:
                data = await websocket.receive_text()
                data_info = json.loads(data)
                logger.info(f"Received WebSocket data: {data_info}")
                if data_info["action"] == "generation":
                    async for message in current_cognitive_kernel.generate_for_demo(
                        messages=data_info["messages"],
                        CKStatus=data_info["CKStatus"],
                        username=data_info["username"],
                        message_id=data_info["currentMessageId"],
                        mode="generation",
                    ):
                        await asyncio.sleep(0.001)
                        await websocket.send_text(message)
                    await websocket.send_text("[save_message]")
                elif data_info["action"] == "regeneration":
                    if message_task:
                        message_task.cancel()
                    message_task = asyncio.create_task(
                        generate_messages(websocket, data_info, mode="regeneration")
                    )
                elif data_info["action"] == "stop" and message_task:
                    message_task.cancel()
                    message_task = None
                    await websocket.send_text("[task_cancelled]")
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed.")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        if message_task:
            message_task.cancel()
        logger.info("WebSocket connection fully closed.")

# --- API Endpoints ---
@app.post("/generate_ck")
async def generate_ck(input_info: dict):
    return StreamingResponse(
        current_cognitive_kernel.generate_for_demo(
            messages=input_info["messages"],
            CKStatus=input_info["CKStatus"],
            username=input_info["username"],
            message_id=input_info["currentMessageId"],
        ),
        media_type="application/json",
    )

@app.post("/inference_api")
async def inference_api(input_info: dict):
    messages = input_info["messages"]
    if "full_info" in input_info:
        response = await current_cognitive_kernel.inference_api(
            messages, input_info["full_info"]
        )
    else:
        response = await current_cognitive_kernel.inference_api(messages)
    return JSONResponse(content=response, media_type="application/json")

@app.post("/inference_api_call_web")
async def inference_api_call_web(input_info: dict):
    response = await current_cognitive_kernel.inference_api_call_web(
        query=input_info["query"],
        target_url=input_info["target_url"],
        session_id=input_info["session_id"],
        message_id=input_info["message_id"],
        username=input_info["username"],
        max_steps=input_info["max_steps"],
        storage_state=input_info.get("storage_state"),
        geo_location=input_info.get("geo_location"),
    )
    return JSONResponse(content=response, media_type="application/json")

@app.post("/inference_api_upload_file")
async def inference_api_upload_file(file: UploadFile = File(...)):
    try:
        file_location = f"{FILE_LOCATIONS}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)
        if ACTIVATE_KE:
            current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            db_location = f"{DB_LOCATIONS}/{current_timestamp}.{file.filename}.db"
            db_name = f"{current_timestamp}.{file.filename}"
            file_content = read_file_content(
                file_location=file_location, file_name=file.filename
            )
            selected_sentences = [tmp["text"] for tmp in file_content[:TOP_K_SENTENCES]]
            selected_meta_data = [tmp["meta"] for tmp in file_content[:TOP_K_SENTENCES]]
            current_cognitive_kernel.update_knowledge_engine(
                db_name=db_name,
                db_path=db_location,
                sentences=selected_sentences,
                metadata=selected_meta_data,
            )
        else:
            db_location = ""
            db_name = ""
        uploaded_info = {
            "file_location": file_location,
            "db_location": db_location,
            "file_name": file.filename,
            "db_name": db_name,
        }
        return JSONResponse(content=uploaded_info, media_type="application/json")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

@app.post("/inference_api_history")
async def inference_api_history(input_info: dict):
    retrieved_info = await current_cognitive_kernel.inference_api_history_retrieval(
        input_info["candidate_history_messages"], input_info["target_query"]
    )
    return JSONResponse(content=retrieved_info, media_type="application/json")

@app.post("/clean_up_ck")
async def clean_up_ck(input_info: dict):
    current_cognitive_kernel.clean_up(
        CKStatus=input_info["CKStatus"],
        username=input_info["username"],
    )
    return JSONResponse(content={"data": "success"}, media_type="application/json")

@app.post("/retrieve_history")
async def retrieve_history(input_info: dict):
    sessions_info = get_sessions_by_username(
        model_name=input_info["model_name"], username=input_info["username"]
    )
    return JSONResponse(content={"data": sessions_info}, media_type="application/json")

@app.post("/retrieve_message_session_by_id")
async def retrieve_message_session_by_id(input_info: dict):
    session_info = get_session_by_id(session_id=input_info["session_id"])
    return JSONResponse(content={"data": session_info}, media_type="application/json")

@app.post("/archive_message_session_by_id")
async def archive_message_session_by_id(input_info: dict):
    try:
        archive_session_by_id(session_id=input_info["session_id"])
        return JSONResponse(
            content={"status": "success"}, media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error archiving session: {e}")
        return JSONResponse(
            content={"status": "failed", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

@app.post("/save_message_to_db")
async def save_message_to_db(input_info: dict):
    try:
        update_or_create_session(
            session_id=input_info["session_id"],
            username=input_info["username"],
            model_name=input_info["model_name"],
            messages=input_info["messages"],
            updated_time=input_info["updated_time"],
        )
        return JSONResponse(content={"data": "success"}, media_type="application/json")
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return JSONResponse(
            content={"data": "failed", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

@app.post("/retrieve_rawdata_by_message_id")
async def retrieve_rawdata_by_message_id(input_info: dict):
    rawdata_info = get_rawdata_by_message_id(message_id=input_info["currentMessageId"])
    return JSONResponse(content={"data": rawdata_info}, media_type="application/json")

@app.post("/submit_annotation")
async def submit_annotation(input_info: dict):
    try:
        update_or_create_annotation(
            session_id=input_info["session_id"],
            message_id=input_info["currentMessageId"],
            username=input_info["username"],
            tag=input_info["tag"],
            for_evaluation=input_info["for_evaluation"],
            old_message=input_info["oldKnowledge"],
            suggestion=input_info["Suggestion"],
            messages_in_train_format=input_info["messages_in_train_format"],
            updated_time=input_info["updated_time"],
        )
        current_cognitive_kernel.update_online_feedback_db(input_info)
        return JSONResponse(content={"data": "success"}, media_type="application/json")
    except Exception as e:
        logger.error(f"Error submitting annotation: {e}")
        return JSONResponse(
            content={"data": "failed", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

@app.get("/download_annotations", response_model=List[AnnotationModel])
async def download_annotations(
    username: str, start_date: str, end_date: str, download_type: str
):
    if download_type == "all":
        annotations = get_all_annotations()
    else:
        annotations = get_anno_by_username_and_date_range(
            username, start_date, end_date
        )
    return annotations

@app.get("/annotation_statistics")
async def annotation_statistics(username: str, current_time: str):
    annotation_statistics = get_annotation_counts_by_username(username, current_time)
    return JSONResponse(
        content={"data": annotation_statistics}, media_type="application/json"
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), ckStatus: str = Form(...)):
    try:
        ck_status_data = json.loads(ckStatus)
        file_location = f"{FILE_LOCATIONS}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)
        if ACTIVATE_KE:
            db_location = f"{DB_LOCATIONS}/{ck_status_data['session_id']}_{file.filename}.db"
            db_name = f"{file.filename}_{ck_status_data['session_id']}"
            file_content = read_file_content(
                file_location=file_location, file_name=file.filename
            )
            selected_sentences = [tmp["text"] for tmp in file_content[:TOP_K_SENTENCES]]
            selected_meta_data = [tmp["meta"] for tmp in file_content[:TOP_K_SENTENCES]]
            current_cognitive_kernel.update_knowledge_engine(
                db_name=db_name,
                db_path=db_location,
                sentences=selected_sentences,
                metadata=selected_meta_data,
            )
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.post("/upload_character_info")
async def upload_avatar(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"{UPLOAD_DIR}/{unique_filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_url = f"/api/uploads/{unique_filename}"
        return JSONResponse(content={"url": file_url})
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.post("/delete_agent/")
async def delete_agent(username: str = Form(...), agent_name: str = Form(...)):
    agent_path = f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_name}"
    if os.path.exists(agent_path):
        shutil.rmtree(agent_path)
        current_cognitive_kernel.delete_character(f"{username}_{agent_name}")
        return {"info": "success"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent does not exist.")

@app.post("/create_agent/")
async def create_agent(
    username: str = Form(...),
    avatarURL: Optional[str] = Form(None),
    backgroundURL: Optional[str] = Form(None),
    agent_name: str = Form(...),
    agent_id: str = Form(...),
    description: Optional[str] = Form(None),
):
    try:
        agent_path = f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}"
        if os.path.exists(agent_path):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Agent already exists.",
            )
        existing_agent_count = sum(
            1 for tmp_file in os.listdir(CUSTOMIZED_CHARACTER_POOL_PATH)
            if username == tmp_file.split("_")[0]
        )
        if existing_agent_count >= MAX_CUSTOMIZED_CHARACTER:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Exceeds the maximum number of customized characters: {MAX_CUSTOMIZED_CHARACTER}",
            )
        os.makedirs(f"{agent_path}/functions", exist_ok=True)
        if backgroundURL and len(backgroundURL) > 0:
            real_background_path = backgroundURL.replace("/api/uploads", UPLOAD_DIR)
            db_location = f"{GLOBAL_DB_LOCATIONS}/{username}_{agent_id}_background.db"
            db_name = f"{username}_{agent_id}"
            file_content = read_file_content(
                file_location=real_background_path,
                file_name=f"{username}_{agent_id}_background.txt",
            )
            selected_sentences = [tmp["text"] for tmp in file_content[:TOP_K_SENTENCES]]
            selected_meta_data = [tmp["meta"] for tmp in file_content[:TOP_K_SENTENCES]]
            current_cognitive_kernel.update_knowledge_engine(
                db_name=db_name,
                db_path=db_location,
                sentences=selected_sentences,
                metadata=selected_meta_data,
            )
        avatar_location = f"{agent_path}/avatar.png"
        if avatarURL and len(avatarURL) > 0:
            real_avatar_path = avatarURL.replace("/api/uploads", UPLOAD_DIR)
            shutil.copy(real_avatar_path, avatar_location)
        else:
            shutil.copy("/app/resources/default_avatar.png", avatar_location)
        agent_info = {
            "name": f"{username}_{agent_id}",
            "id": agent_id,
            "shown_title": agent_name,
            "description": description or "Your Customized Character",
            "visible_users": [username],
            "head_system_prompt": description,
            "tail_system_prompt": (
                "Please play your role well, do not violate or reveal your character. "
                "When the player behaves maliciously or provokes you many times, please politely refuse and express anger"
            ),
            "global_db_info": {
                f"{username}_{agent_id}": f"{username}_{agent_id}_background"
            },
            "system_prompt_sequence": [
                {
                    "name": "available_functions",
                    "pre_defined": True,
                    "step_type": "dynamic",
                    "head": "",
                    "content": [],
                    "CK_status_key": "",
                },
                {
                    "name": "character_dbs",
                    "pre_defined": False,
                    "step_type": "static",
                    "head": "可用的db名称及对应描述如下:",
                    "content": [f"{username}_{agent_id}"],
                    "CK_status_key": "uploaded_files",
                }
            ]
        }
        shutil.copytree(
            f"{CHARACTER_POOL_PATH}/cognitiveKernel/functions/CallMemoryKernel",
            f"{agent_path}/functions/CallMemoryKernel",
        )
        with open(f"{agent_path}/info.json", "w") as f:
            json.dump(agent_info, f)
        current_cognitive_kernel.update_character(f"{username}_{agent_id}", "customized")
        return {"info": "success"}
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.post("/get_all_characters")
async def get_all_characters(username: str):
    return JSONResponse(
        content=current_cognitive_kernel.get_all_characters(username=username),
        media_type="application/json"
    )

# --- Main ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=NUM_WORKERS,
    )
