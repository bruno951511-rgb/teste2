from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import subprocess
import asyncio
import aiofiles
import re
import base64
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import google.generativeai as genai

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'terminal_ai')]

# Sistema de múltiplas chaves Gemini
GEMINI_KEYS = os.environ.get('GEMINI_KEYS', '').split(',')
GEMINI_KEYS = [key.strip() for key in GEMINI_KEYS if key.strip()]

if not GEMINI_KEYS:
    single_key = os.environ.get('GEMINI_API_KEY', '')
    if single_key:
        GEMINI_KEYS = [single_key]

print(f"🔑 Carregadas {len(GEMINI_KEYS)} chaves Gemini")

current_key_index = 0
key_usage_count = {}

def get_next_gemini_model():
    global current_key_index
    if not GEMINI_KEYS:
        raise Exception("Nenhuma chave Gemini configurada!")
    
    key = GEMINI_KEYS[current_key_index]
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    if key not in key_usage_count:
        key_usage_count[key] = 0
    key_usage_count[key] += 1
    
    masked_key = f"{key[:10]}...{key[-4:]}"
    print(f"🔑 Usando chave {current_key_index + 1}/{len(GEMINI_KEYS)} ({masked_key}) - Uso #{key_usage_count[key]}")
    
    return model, current_key_index

def rotate_to_next_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(GEMINI_KEYS)
    print(f"🔄 Rotacionando para chave {current_key_index + 1}/{len(GEMINI_KEYS)}")

# FastAPI app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Diretórios
SANDBOX_DIR = Path(ROOT_DIR / "sandbox")
SAVE_DIR = Path(ROOT_DIR / "save")
CHATS_DIR = Path(ROOT_DIR / "chats")
UPLOADS_DIR = Path(ROOT_DIR / "uploads")

for dir_path in [SANDBOX_DIR, SAVE_DIR, CHATS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

MEMORY_FILE = SAVE_DIR / "save.txt"
if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text("")

# Models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    command: Optional[str] = None
    output: Optional[str] = None
    files_created: Optional[List[str]] = None

class Chat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CreateChatRequest(BaseModel):
    name: str

class SendMessageRequest(BaseModel):
    content: str
    chat_id: str
    image: Optional[str] = None

class MemoryEntry(BaseModel):
    question: str
    thought: str
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
async def load_memory() -> str:
    if MEMORY_FILE.exists():
        async with aiofiles.open(MEMORY_FILE, 'r') as f:
            return await f.read()
    return ""

async def save_to_memory(entry: MemoryEntry):
    memory_text = f"""\n{'='*50}\n📅 {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n🗣️ USUÁRIO: {entry.question}\n💭 PENSAMENTO: {entry.thought}\n🤖 RESPOSTA: {entry.response}\n{'='*50}\n"""
    async with aiofiles.open(MEMORY_FILE, 'a') as f:
        await f.write(memory_text)

async def execute_command(cmd: str, timeout: int = 30) -> tuple[str, int, List[str]]:
    """Executa comando e retorna output, exit_code e arquivos criados"""
    files_before = set(SANDBOX_DIR.iterdir())
    
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SANDBOX_DIR),
            env={**os.environ, 'HOME': str(SANDBOX_DIR)}
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            output = stdout.decode('utf-8', errors='ignore')
            if stderr:
                output += "\n" + stderr.decode('utf-8', errors='ignore')
            
            # Verificar arquivos criados
            files_after = set(SANDBOX_DIR.iterdir())
            new_files = [str(f.name) for f in (files_after - files_before)]
            
            return output.strip() if output.strip() else "✓ Comando executado com sucesso", process.returncode or 0, new_files
        except asyncio.TimeoutError:
            process.kill()
            return f"⏱️ Timeout: Comando demorou mais de {timeout}s", 1, []
    except Exception as e:
        return f"❌ Erro: {str(e)}", 1, []

async def ask_gemini(user_message: str, memory: str, context: str = "", image_data: Optional[str] = None) -> dict:
    """Consulta Gemini com pensamento profundo e suporte a imagens"""
    
    system_prompt = f"""Você é um assistente de terminal Linux AVANÇADO focado em PROGRAMAÇÃO.

📁 PASTA ATUAL: sandbox/ (você PODE criar arquivos aqui!)
🔧 AMBIENTE: Linux Ubuntu

REGRAS IMPORTANTES:
1. Quando o usuário pedir para CRIAR um arquivo, use comandos que REALMENTE criem:
   - Para arquivos simples: echo "conteúdo" > arquivo.txt
   - Para arquivos HTML: cat > index.html << 'EOF' e o conteúdo e EOF
   - Para Python: cat > script.py << 'EOF' e o código e EOF
   
2. SEMPRE retorne JSON válido no formato:
   {{
       "type": "command" ou "chat",
       "thought": "seu raciocínio detalhado",
       "command": "comando a executar" (se type=command),
       "response": "resposta em texto" (se type=chat),
       "explanation": "explicação" (se type=command)
   }}

3. NUNCA retorne o JSON como texto puro na resposta!

4. Para criar arquivos multi-linha use heredoc:
   cat > arquivo.html << 'EOF'
   <conteúdo aqui>
   EOF

MEMÓRIA:
{memory[-2000:] if memory else "Vazia"}

CONTEXTO:
{context[-1000:] if context else "Início"}

COMANDOS DISPONÍVEIS:
ls, cat, echo, touch, mkdir, rm, cp, mv, nano, vim, python3, node, npm, gcc, make, git, curl, wget
"""

    max_retries = len(GEMINI_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            model, key_idx = get_next_gemini_model()
            
            # Preparar conteúdo (com imagem se houver)
            content_parts = [f"{system_prompt}\n\nUsuário: {user_message}"]
            
            if image_data:
                # Adicionar imagem para análise
                content_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': image_data
                })
            
            response = model.generate_content(content_parts)
            response_text = response.text.strip()
            
            # Extrair JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    # Garantir que não retorne JSON como texto
                    if parsed.get('type') == 'chat' and 'response' in parsed:
                        return parsed
                    elif parsed.get('type') == 'command':
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            # Se não conseguiu parsear, retornar como chat
            return {
                "type": "chat",
                "thought": "Processando resposta",
                "response": response_text
            }
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ Chave {key_idx + 1} esgotou! Rotacionando...")
                rotate_to_next_key()
                retry_count += 1
                
                if retry_count < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {
                        "type": "error",
                        "thought": "Todas as chaves esgotaram",
                        "response": f"❌ Todas as {len(GEMINI_KEYS)} chaves esgotaram a quota. Aguarde 24h."
                    }
            else:
                return {
                    "type": "error",
                    "thought": "Erro ao processar",
                    "response": f"Erro: {error_msg}"
                }
    
    return {
        "type": "error",
        "thought": "Máximo de tentativas excedido",
        "response": "Não foi possível processar após múltiplas tentativas."
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Terminal AI V2 - Advanced Edition"}

@api_router.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0",
        "gemini_keys": len(GEMINI_KEYS),
        "current_key": current_key_index + 1,
        "key_usage": key_usage_count
    }

@api_router.post("/chats", response_model=Chat)
async def create_chat(request: CreateChatRequest):
    chat = Chat(name=request.name)
    chat_dict = chat.model_dump()
    chat_dict['created_at'] = chat_dict['created_at'].isoformat()
    chat_dict['updated_at'] = chat_dict['updated_at'].isoformat()
    await db.chats.insert_one(chat_dict)
    return chat

@api_router.get("/chats", response_model=List[Chat])
async def get_chats():
    chats = await db.chats.find({}, {"_id": 0}).sort("updated_at", -1).to_list(100)
    for chat in chats:
        if isinstance(chat.get('created_at'), str):
            chat['created_at'] = datetime.fromisoformat(chat['created_at'])
        if isinstance(chat.get('updated_at'), str):
            chat['updated_at'] = datetime.fromisoformat(chat['updated_at'])
        for msg in chat.get('messages', []):
            if isinstance(msg.get('timestamp'), str):
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    return chats

@api_router.get("/chats/{chat_id}", response_model=Chat)
async def get_chat(chat_id: str):
    chat = await db.chats.find_one({"id": chat_id}, {"_id": 0})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if isinstance(chat.get('created_at'), str):
        chat['created_at'] = datetime.fromisoformat(chat['created_at'])
    if isinstance(chat.get('updated_at'), str):
        chat['updated_at'] = datetime.fromisoformat(chat['updated_at'])
    for msg in chat.get('messages', []):
        if isinstance(msg.get('timestamp'), str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    return chat

@api_router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    result = await db.chats.delete_one({"id": chat_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}

@api_router.post("/message")
async def send_message(request: SendMessageRequest):
    memory = await load_memory()
    chat = await db.chats.find_one({"id": request.chat_id}, {"_id": 0})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    context = ""
    for msg in chat.get('messages', [])[-5:]:
        context += f"{msg['role']}: {msg['content']}\n"
    
    user_message = Message(role="user", content=request.content)
    
    # Processar imagem se houver
    image_data = None
    if request.image:
        image_data = request.image
    
    gemini_response = await ask_gemini(request.content, memory, context, image_data)
    assistant_message = Message(role="assistant", content="")
    command_output = None
    files_created = []
    
    if gemini_response.get("type") == "command":
        cmd = gemini_response.get("command", "")
        thought = gemini_response.get("thought", "")
        explanation = gemini_response.get("explanation", "")
        
        output, exit_code, new_files = await execute_command(cmd)
        files_created = new_files
        
        # Montar resposta SEM mostrar JSON
        response_parts = []
        if thought:
            response_parts.append(f"💭 {thought}")
        response_parts.append(f"\n🔧 Executei: `{cmd}`")
        if explanation:
            response_parts.append(f"\n📝 {explanation}")
        if new_files:
            response_parts.append(f"\n📁 Arquivos criados: {', '.join(new_files)}")
        
        assistant_message.content = "\n".join(response_parts)
        assistant_message.command = cmd
        assistant_message.output = output
        assistant_message.files_created = files_created
        command_output = output
        
        await save_to_memory(MemoryEntry(
            question=request.content,
            thought=thought,
            response=f"Executei: {cmd}\nSaída: {output[:500]}"
        ))
        
    elif gemini_response.get("type") == "chat":
        thought = gemini_response.get("thought", "")
        response = gemini_response.get("response", "")
        
        # Retornar APENAS a resposta, SEM o JSON
        assistant_message.content = response
        
        await save_to_memory(MemoryEntry(
            question=request.content,
            thought=thought,
            response=response[:500]
        ))
    else:
        assistant_message.content = gemini_response.get("response", "Erro desconhecido")
    
    user_msg_dict = user_message.model_dump()
    user_msg_dict['timestamp'] = user_msg_dict['timestamp'].isoformat()
    
    assistant_msg_dict = assistant_message.model_dump()
    assistant_msg_dict['timestamp'] = assistant_msg_dict['timestamp'].isoformat()
    
    await db.chats.update_one(
        {"id": request.chat_id},
        {
            "$push": {"messages": {"$each": [user_msg_dict, assistant_msg_dict]}},
            "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}
        }
    )
    
    return {
        "user_message": user_message,
        "assistant_message": assistant_message,
        "command_output": command_output,
        "files_created": files_created
    }

@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...), chat_id: str = Query(...)):
    file_path = SANDBOX_DIR / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    return {
        "message": f"Arquivo '{file.filename}' enviado para sandbox/",
        "path": str(file_path),
        "size": len(content)
    }

@api_router.get("/files")
async def list_files():
    files = []
    for item in SANDBOX_DIR.iterdir():
        files.append({
            "name": item.name,
            "is_dir": item.is_dir(),
            "size": item.stat().st_size if item.is_file() else 0,
            "path": str(item)
        })
    return {"files": files}

@api_router.get("/files/{filename}")
async def download_file(filename: str):
    file_path = SANDBOX_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(file_path, filename=filename)

@api_router.get("/memory")
async def get_memory():
    memory = await load_memory()
    return {"memory": memory}

@api_router.delete("/memory")
async def clear_memory():
    async with aiofiles.open(MEMORY_FILE, 'w') as f:
        await f.write("")
    return {"message": "Memória limpa"}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
