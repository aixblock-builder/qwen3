import os
from typing import Any, Dict, Optional, List, Union
import requests
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel
from starlette.routing import Mount

from model import MyModel, mcp


# Models for request validation
class InstallServiceRequest(BaseModel):
    git: str


class ServiceInfoRequest(BaseModel):
    directory: str
    port_map: Optional[int] = None


class StopServiceRequest(BaseModel):
    port_map: int
    directory: Optional[str] = None


class DashboardRequest(BaseModel):
    directory: str


app = FastAPI(title="MyModel API", openapi_url="/swagger.json", docs_url="/swagger")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus and Loki configuration
LOKI_URL = os.getenv("LOKI_URL", "http://207.246.109.178:3100")
JOB_NAME = os.getenv("JOB_NAME", "test-fastapi")
PUSH_GATEWAY_URL = os.getenv("PUSH_GATEWAY_URL", "http://207.246.109.178:9091")
JOB_INTERVAL = int(os.getenv("JOB_INTERVAL", 60))

model = MyModel()


class ActionRequest(BaseModel):
    command: str
    params: Dict[str, Any]
    doc_file_urls: Optional[Union[str, List[str]]] = None


@app.post("/action")
async def action(request: ActionRequest = Body(...)):
    try:
        parsed_params = request.params

        # Normalize URL list
        doc_file_urls = request.doc_file_urls
        if isinstance(doc_file_urls, str):
            doc_file_urls = [doc_file_urls]

        if doc_file_urls:
            # Convert URLs to list of temp file paths
            file_paths = fetch_file_paths_from_urls_sync(doc_file_urls)
            parsed_params["doc_files"] = file_paths
            parsed_params["docchat"] = True

        result = model.action(request.command, **parsed_params)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_file_paths_from_urls_sync(urls: List[str], save_dir: str = "downloads") -> List[str]:
    file_paths = []

    # Tạo thư mục nếu chưa tồn tại
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            filename = url.split("/")[-1] or "file"
            suffix = os.path.splitext(filename)[-1] or ".pdf"

            # Đảm bảo tên file không trùng lặp
            save_path = Path(save_dir) / filename
            counter = 1
            while save_path.exists():
                save_path = Path(save_dir) / f"{Path(filename).stem}_{counter}{suffix}"
                counter += 1

            # Ghi nội dung vào file
            with open(save_path, "wb") as f:
                f.write(response.content)

            file_paths.append(str(save_path))

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            continue

    return file_paths

@app.get("/")
async def model_endpoint(data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model(**(data or {}))
        if "share_url" in result:
            return RedirectResponse(url=result["share_url"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model")
async def model_endpoint(data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model(**(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model-trial")
async def model_trial(project: str, data: Optional[Dict[str, Any]] = None):
    try:
        result = model.model_trial(project, **(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
async def download(project: str, data: Optional[Dict[str, Any]] = None):
    try:
        result = model.download(project, **(data or {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/downloads")
async def download_file(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="File name is required")

    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path, filename=os.path.basename(full_path))


sse = SseServerTransport("/messages/")

app.router.routes.append(Mount("/messages", app=sse.handle_post_message))


# Add documentation for the /messages endpoint
@app.get("/messages", tags=["MCP"], include_in_schema=True)
def messages_docs():
    pass


@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server

    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    # Use sse.connect_sse to establish an SSE connection with the MCP server
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        # Run the MCP server with the established streams
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


# @app.on_event("shutdown")
# async def shutdown_event():
#     scheduler.shutdown()

if __name__ == "__main__":
    import socket
    import ssl
    import sys
    import subprocess

    import uvicorn

    def find_available_port(start_port=3000, max_port=5000):
        for port in range(start_port, max_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available ports found")

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="ssl/cert.pem", keyfile="ssl/key.pem")

    available_port = find_available_port()
    print(f"Starting server on port {available_port}")

    # uvicorn.run(
    #     app,
    #     host="0.0.0.0",
    #     port=available_port,
    #     ssl_keyfile="ssl/key.pem",
    #     ssl_certfile="ssl/cert.pem",
    # )
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",  # đổi "main" nếu file của bạn tên khác
        "--host", "0.0.0.0",
        "--port", str(available_port),
        "--workers", "1",
        "--ssl-keyfile", "ssl/key.pem",
        "--ssl-certfile", "ssl/cert.pem"
    ]

    subprocess.run(cmd)
