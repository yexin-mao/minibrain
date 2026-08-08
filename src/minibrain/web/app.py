"""FastAPI 应用：既出 HTML 页面，也出 htmx 片段。

这一层的唯一规矩：只做解析、鉴权、调 gateway、渲染。
任何一行业务逻辑都不许写在这里 —— 一旦开了"就这一处先放这儿"的口子，
它会长成一个上千行、几十个分支的路由文件，然后再也搬不回去。

路由全部是 def 而不是 async def：FastAPI 会自动丢进线程池。
MVP 没有并发压力，async 只会把你拖进事件循环和阻塞调用泄漏的问题里。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import gateway, identity
from ..agent.graph import answer as run_agent
from ..config import get_config
from ..contracts import ModuleError, UserContext
from ..db import apply_schema

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]
SESSION_COOKIE = "minibrain_session"
# ★ 会话 id 和登录 token 是**两个不同的东西**，刻意分开：
#   - 登录 token 决定"你是谁"，泄露了等于被盗号
#   - 会话 id 只决定"这轮对话的历史存在哪个 thread 里"
#   合用一个的话，退出再登录会拿到新 token，对话历史就断了；
#   反过来，想开一个新对话就得重新登录。语义不同，生命周期也不同。
CHAT_COOKIE = "minibrain_chat"

app = FastAPI(title="minibrain", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
def startup() -> None:
    """schema.sql 整份跑一遍。幂等，没上线之前不需要迁移框架。"""
    apply_schema((ROOT_DIR / "schema.sql").read_text(encoding="utf-8"))


def current_user(request: Request) -> UserContext | None:
    return identity.resolve_session(request.cookies.get(SESSION_COOKIE))


@app.exception_handler(ModuleError)
def module_error_handler(request: Request, exc: ModuleError):
    """模块错误统一映射。模块抛稳定错误，web 层只负责翻译成 HTTP。"""
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            f'<div class="error">出错了（{exc.code}）：{exc.message}</div>', status_code=200
        )
    return JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)


# ---------------------------------------------------------------- 登录

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        token = identity.authenticate(username, password)
    except ModuleError as exc:
        return templates.TemplateResponse(
            request, "login.html", {"error": exc.message}, status_code=401
        )

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="lax",
        max_age=get_config().session_ttl_hours * 3600,
    )
    return response


@app.post("/logout")
def logout(request: Request):
    identity.revoke_session(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    # 登出也断开对话上下文：换个人用同一台电脑，不该看到上一个人的对话
    response.delete_cookie(CHAT_COOKIE)
    return response


# ---------------------------------------------------------------- 主页

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "modules": gateway.list_modules(),
            "documents": gateway.call("vector-rag", "list_documents", user),
            "datasets": gateway.call("table-rag", "list_datasets", user),
        },
    )


@app.get("/fragments/knowledge", response_class=HTMLResponse)
def knowledge_fragment(request: Request):
    """htmx 轮询这个片段来刷新处理状态。"""
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    return templates.TemplateResponse(
        request,
        "_knowledge.html",
        {
            "documents": gateway.call("vector-rag", "list_documents", user),
            "datasets": gateway.call("table-rag", "list_datasets", user),
        },
    )


# ---------------------------------------------------------------- 上传

@app.post("/upload", response_class=HTMLResponse)
def upload(
    request: Request,
    background: BackgroundTasks,
    module: str = Form(...),
    file: UploadFile = Form(...),
):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    raw = file.file.read()
    filename = file.filename or "untitled"

    if module == "vector-rag":
        # ★ 传 raw 字节，不在这里 decode。
        #   原来这里是 raw.decode("utf-8", "replace") —— 传 PDF 进来会被腐化成
        #   一片替换字符，然后照常入库、状态标成 ready，用户永远检索不到。
        #   类型识别和解析统一在 core.upload_bytes 里做，失败会落 failed 并说明原因。
        entity_id = gateway.call("vector-rag", "upload_bytes", user, None, filename, raw)
    elif module == "table-rag":
        entity_id = gateway.call("table-rag", "upload_csv", user, None, filename, raw)
    else:
        raise ModuleError(f"未知模块 {module}", code="unknown_module", status=404)

    # 立刻返回，处理放后台。前端靠 htmx 轮询状态表。
    background.add_task(gateway.process, module, entity_id)

    return templates.TemplateResponse(
        request,
        "_knowledge.html",
        {
            "documents": gateway.call("vector-rag", "list_documents", user),
            "datasets": gateway.call("table-rag", "list_datasets", user),
        },
    )


# ---------------------------------------------------------------- 问答

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    """带会话的问答。历史由 LangGraph 的 PostgresSaver 管，我们只传一个 id。

    ★ 这里**不**自己拼历史。第一版的诱惑是「把上几轮问答塞进 prompt」，
      那样要自己解决三件事：存哪、超长了怎么裁、并发写同一个会话怎么办。
      checkpointer 就是为这三件事存在的，我们只负责给它一个 thread_id。
    """
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    chat_id = request.cookies.get(CHAT_COOKIE)
    fresh = not chat_id
    if fresh:
        # ★ 会话 id 必须**服务端生成**，不能接受客户端传。
        #   接受的话，改一下 cookie 就能读到别人的对话历史 ——
        #   checkpointer 只认 thread_id，它不知道谁该看哪个 thread。
        chat_id = f"{user.user_id}:{uuid.uuid4().hex}"

    result = run_agent(user, question.strip(), session_id=chat_id)
    response = templates.TemplateResponse(
        request, "_answer.html", {"question": question, "result": result}
    )
    if fresh:
        response.set_cookie(CHAT_COOKIE, chat_id, httponly=True, samesite="lax",
                            max_age=get_config().session_ttl_hours * 3600)
    return response


@app.post("/chat/new")
def new_chat(request: Request):
    """开一个新对话：把会话 cookie 删掉，下次提问会生成新的 thread_id。

    ★ 旧的 checkpoint **不删**。历史留在库里，将来要做「对话列表」
      直接就有数据；现在只是不再往那个 thread 上追加。
    """
    response = HTMLResponse(
        '<div class="hint">已开始新对话，之前的上下文不再带入。</div>')
    response.delete_cookie(CHAT_COOKIE)
    return response


# ---------------------------------------------------------------- 运维

@app.get("/health")
def health():
    cfg = get_config()
    return {
        "status": "ok",
        "modules": gateway.health(),
        "embedding_configured": cfg.embedding_configured,
        "agent_configured": cfg.agent_configured,
        "embedding_model": cfg.embedding_model,
        "embedding_dimensions": cfg.embedding_dimensions,
    }
