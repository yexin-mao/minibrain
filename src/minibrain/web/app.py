"""FastAPI 应用：既出 HTML 页面，也出 htmx 片段。

这一层的唯一规矩：只做解析、鉴权、调 gateway、渲染。
任何一行业务逻辑都不许写在这里 —— 一旦开了"就这一处先放这儿"的口子，
它会长成一个上千行、几十个分支的路由文件，然后再也搬不回去。

路由全部是 def 而不是 async def：FastAPI 会自动丢进线程池。
MVP 没有并发压力，async 只会把你拖进事件循环和阻塞调用泄漏的问题里。
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import gateway, identity
from ..agent.graph import answer as run_agent
from ..agent.session_control import valid_chat_id
from ..config import get_config
from ..contracts import ModuleError, UserContext
from ..demo import seed_demo
from ..db import apply_schema
from ..observability import core as observability

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


def _knowledge_context(user: UserContext) -> dict:
    """知识管理片段的唯一上下文组装点，避免各路由漏传 source 或权限状态。"""
    return {
        "user": user,
        "documents": gateway.call("vector-rag", "list_documents", user),
        "datasets": gateway.call("table-rag", "list_datasets", user),
        "vector_sources": gateway.call("vector-rag", "list_sources", user),
        "table_sources": gateway.call("table-rag", "list_sources", user),
    }


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
            **_knowledge_context(user), "modules": gateway.list_modules(),
            "request_id": str(uuid.uuid4()),
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
        _knowledge_context(user),
    )


# ---------------------------------------------------------------- Wiki

@app.get("/wiki", response_class=HTMLResponse)
def wiki_index(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "wiki.html",
        {
            "user": user,
            "pages": gateway.call("vector-rag", "list_wiki_pages", user),
            "topics": gateway.call("vector-rag", "list_wiki_topics", user),
            "health": gateway.call("vector-rag", "get_wiki_health", user),
            "events": gateway.call("vector-rag", "list_wiki_events", user, 12),
            "documents": gateway.call("vector-rag", "list_documents", user),
        },
    )


@app.post("/wiki/build")
def wiki_build(request: Request, document_id: str = Form(...)):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    page = gateway.call("vector-rag", "build_wiki_page", user, document_id)
    return RedirectResponse(f"/wiki/{page['id']}", status_code=303)


@app.post("/wiki/ask", response_class=HTMLResponse)
def wiki_ask(request: Request, question: str = Form(...)):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    result = gateway.call("vector-rag", "ask_wiki", user, question)
    return templates.TemplateResponse(
        request,
        "_wiki_answer.html",
        {"question": question, "result": result},
    )


@app.post("/wiki/lint")
def wiki_lint(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    gateway.call("vector-rag", "run_wiki_lint", user)
    return RedirectResponse("/wiki", status_code=303)


@app.get("/wiki/topic/{topic_id}", response_class=HTMLResponse)
def wiki_topic_detail(request: Request, topic_id: str):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "wiki_topic_detail.html",
        {"user": user, "topic": gateway.call("vector-rag", "get_wiki_topic", user, topic_id)},
    )


@app.get("/wiki/source/{document_id}/{version}", response_class=HTMLResponse)
def wiki_source_revision(request: Request, document_id: str, version: int):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "wiki_source_revision.html",
        {
            "user": user,
            "revision": gateway.call(
                "vector-rag", "get_document_revision", user, document_id, version
            ),
        },
    )


@app.get("/wiki/{page_id}", response_class=HTMLResponse)
def wiki_detail(request: Request, page_id: str):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "wiki_detail.html",
        {"user": user, "page": gateway.call("vector-rag", "get_wiki_page", user, page_id)},
    )


# ---------------------------------------------------------------- 上传

@app.post("/upload", response_class=HTMLResponse)
def upload(
    request: Request,
    module: str = Form(...),
    source_id: str = Form(""),
    file: UploadFile = Form(...),
):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    # 只多读 1 字节用于判定超限，避免先把任意大文件完整装进 Web 进程内存。
    limit = get_config().upload_max_bytes
    raw = file.file.read(limit + 1)
    filename = file.filename or "untitled"

    # XLSX 的 OOXML 结构不可歧义：即使页面还选着文档链路，也自动送 table-rag。
    # 两模块的 source 在不同 schema，自动切换时不能复用文档 source_id。
    detected_xlsx = gateway.call(
        "table-rag", "detect_upload_kind", user, filename, raw) == "xlsx"
    if detected_xlsx and module != "table-rag":
        module, source_id = "table-rag", ""

    if module == "vector-rag":
        # ★ 传 raw 字节，不在这里 decode。
        #   原来这里是 raw.decode("utf-8", "replace") —— 传 PDF 进来会被腐化成
        #   一片替换字符，然后照常入库、状态标成 ready，用户永远检索不到。
        #   类型识别和解析统一在 core.upload_bytes 里做，失败会落 failed 并说明原因。
        gateway.call(
            "vector-rag", "upload_bytes", user, source_id or None, filename, raw)
    elif module == "table-rag":
        if len(raw) > limit:
            raise ModuleError(
                f"{filename} 超过上传上限 {limit / (1024 * 1024):g} MB",
                code="file_too_large", status=413)
        gateway.call(
            "table-rag", "upload_bytes", user, source_id or None, filename, raw)
    else:
        raise ModuleError(f"未知模块 {module}", code="unknown_module", status=404)

    return templates.TemplateResponse(
        request,
        "_knowledge.html",
        _knowledge_context(user),
    )


@app.post("/knowledge/source/create", response_class=HTMLResponse)
def create_source(request: Request, module: str = Form(...), name: str = Form(...),
                  visibility: str = Form("private")):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    gateway.call(module, "create_source", user, name, visibility)
    return templates.TemplateResponse(request, "_knowledge.html", _knowledge_context(user))


@app.post("/knowledge/demo/load", response_class=HTMLResponse)
def load_demo_knowledge(request: Request):
    """幂等登记内置示例；worker 继续完成解析和索引。"""
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    seed_demo(user)
    return templates.TemplateResponse(request, "_knowledge.html", _knowledge_context(user))


@app.post("/knowledge/source/delete", response_class=HTMLResponse)
def delete_source(request: Request, module: str = Form(...),
                  source_id: str = Form(...)):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    gateway.call(module, "delete_source", user, source_id)
    return templates.TemplateResponse(request, "_knowledge.html", _knowledge_context(user))


@app.post("/knowledge/entity/action", response_class=HTMLResponse)
def entity_action(request: Request, module: str = Form(...), entity_id: str = Form(...),
                  action: str = Form(...)):
    """文档/数据集生命周期入口；允许的动作使用固定映射，不反射用户输入。"""
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    methods = {
        ("vector-rag", "delete"): "delete_document",
        ("vector-rag", "retry"): "retry_document",
        ("vector-rag", "reindex"): "reindex_document",
        ("table-rag", "delete"): "delete_dataset",
        ("table-rag", "retry"): "retry_dataset",
        ("table-rag", "reindex"): "reprocess_dataset",
    }
    method = methods.get((module, action))
    if method is None:
        raise ModuleError("不支持的知识库操作", code="invalid_action", status=400)
    gateway.call(module, method, user, entity_id)
    return templates.TemplateResponse(request, "_knowledge.html", _knowledge_context(user))


# ---------------------------------------------------------------- 问答

def _chat_id_for(user: UserContext, request: Request) -> tuple[str, bool]:
    """Resolve a user-owned conversation id; never trust a client-supplied new id."""
    chat_id = request.cookies.get(CHAT_COOKIE)
    fresh = not valid_chat_id(user, chat_id)
    if fresh:
        chat_id = f"{user.user_id}:{uuid.uuid4().hex}"
    return chat_id, fresh


def _set_chat_cookie(response, chat_id: str, fresh: bool) -> None:
    if fresh:
        response.set_cookie(
            CHAT_COOKIE, chat_id, httponly=True, samesite="lax",
            max_age=get_config().session_ttl_hours * 3600,
        )


def _sse(event: str, payload: dict) -> str:
    """Encode one SSE event. JSON encoding keeps embedded HTML/newlines frame-safe."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...),
        request_id: str | None = Form(None)):
    """带会话的问答。历史由 LangGraph 的 PostgresSaver 管，我们只传一个 id。

    ★ 这里**不**自己拼历史。checkpointer 负责持久化，HistoryWindowMiddleware
      负责裁剪；checkpointer 本身不保证并发，所以 graph.answer 另用数据库锁
      串行化 thread。Web 只负责产生并校验 thread_id / request_id。
    """
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')

    chat_id, fresh = _chat_id_for(user, request)

    result = run_agent(
        user, question.strip(), session_id=chat_id,
        request_id=request_id or str(uuid.uuid4()),
    )
    response = templates.TemplateResponse(
        request, "_answer.html", {"question": question, "result": result}
    )
    _set_chat_cookie(response, chat_id, fresh)
    return response


@app.post("/ask/stream")
def ask_stream(request: Request, question: str = Form(...),
               request_id: str | None = Form(None)):
    """SSE transport for a verified RAG answer.

    The agent currently produces a structured answer atomically.  Streaming its raw
    model tokens would bypass the grounding/citation gate, so this endpoint streams
    lifecycle + heartbeat events and only publishes answer HTML after validation.
    """
    user = current_user(request)
    if user is None:
        return JSONResponse(
            {"error": "unauthenticated", "message": "会话已过期，请重新登录。"},
            status_code=401,
        )

    normalized_question = question.strip()
    if not normalized_question:
        return JSONResponse(
            {"error": "empty_question", "message": "问题不能为空。"}, status_code=400)

    chat_id, fresh = _chat_id_for(user, request)
    normalized_request_id = request_id or str(uuid.uuid4())

    def events():
        outcome: queue.Queue = queue.Queue(maxsize=1)

        def work() -> None:
            try:
                result = run_agent(
                    user, normalized_question, session_id=chat_id,
                    request_id=normalized_request_id,
                )
                outcome.put(("result", result))
            except ModuleError as exc:
                outcome.put(("module_error", exc))
            except Exception:
                # 不把 provider、SQL 或栈信息泄漏到浏览器；详细错误已由 run trace 记录。
                outcome.put(("error", None))

        yield _sse("started", {
            "request_id": normalized_request_id,
            "message": "正在检索并校验证据…",
        })
        threading.Thread(target=work, name="minibrain-sse-answer", daemon=True).start()

        while True:
            try:
                kind, value = outcome.get(timeout=10)
            except queue.Empty:
                yield _sse("heartbeat", {"message": "仍在处理中…"})
                continue

            if kind == "result":
                html = templates.get_template("_answer.html").render(
                    request=request, question=question, result=value)
                yield _sse("completed", {
                    "request_id": normalized_request_id, "html": html})
            elif kind == "module_error":
                yield _sse("failed", {
                    "request_id": normalized_request_id,
                    "error": value.code, "message": value.message,
                })
            else:
                yield _sse("failed", {
                    "request_id": normalized_request_id,
                    "error": "answer_failed", "message": "问答失败，请稍后重试。",
                })
            break

    response = StreamingResponse(
        events(), media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
    _set_chat_cookie(response, chat_id, fresh)
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


# ---------------------------------------------------------------- 检索实验室

def _retrieval_debug_context(result=None, **values) -> dict:
    return {
        "result": result,
        "query": values.get("query", ""),
        "mode": values.get("mode", "hybrid"),
        "top_k": values.get("top_k", 5),
        "candidate_pool": values.get("candidate_pool", 20),
        "use_mmr": values.get("use_mmr", False),
        "mmr_lambda": values.get("mmr_lambda", 0.7),
        "use_rerank": values.get("use_rerank", False),
    }


@app.get("/retrieval-debug", response_class=HTMLResponse)
def retrieval_debug_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "retrieval_debug.html",
        {"user": user, **_retrieval_debug_context()},
    )


@app.post("/retrieval-debug", response_class=HTMLResponse)
def retrieval_debug_run(
    request: Request,
    query: str = Form(...),
    mode: str = Form("hybrid"),
    top_k: int = Form(5),
    candidate_pool: int = Form(20),
    use_mmr: bool = Form(False),
    mmr_lambda: float = Form(0.7),
    use_rerank: bool = Form(False),
):
    """直接执行文档检索并展示阶段快照；不调用 Agent，也不把轨迹送给 LLM。"""
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    if not 1 <= top_k <= 20 or not 1 <= candidate_pool <= 100:
        raise ModuleError("top_k 需为 1–20，候选池需为 1–100", code="invalid_debug_limit")

    values = {
        "query": query, "mode": mode, "top_k": top_k,
        "candidate_pool": candidate_pool, "use_mmr": use_mmr,
        "mmr_lambda": mmr_lambda, "use_rerank": use_rerank,
    }
    result = gateway.call(
        "vector-rag", "search", user, query, top_k=top_k, mode=mode,
        candidate_pool=candidate_pool, use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        rerank_pool=candidate_pool if use_rerank else 0,
        explain=True,
    )
    return templates.TemplateResponse(
        request, "_retrieval_debug_result.html",
        _retrieval_debug_context(result, **values),
    )


# ---------------------------------------------------------------- 运行记录

@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "runs.html",
        {"user": user, "runs": observability.list_runs(user)},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(request: Request, run_id: str):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "user": user,
            "run": observability.get_run(user, run_id),
            "feedback": observability.get_feedback(user, run_id),
        },
    )


@app.post("/runs/{run_id}/feedback", response_class=HTMLResponse)
def submit_run_feedback(
    request: Request,
    run_id: str,
    rating: int = Form(...),
    reason: str = Form(""),
    note: str = Form(""),
):
    user = current_user(request)
    if user is None:
        return HTMLResponse('<div class="error">会话已过期，请重新登录。</div>')
    saved = observability.save_feedback(
        user, run_id, rating=rating, reason=reason or None, note=note)
    label = "有帮助" if saved["rating"] == 1 else "待改进"
    return HTMLResponse(
        f'<span class="st-ready">反馈已保存：{label}。可在运行记录中导出回归候选。</span>'
    )


@app.get("/feedback/regression.json")
def export_feedback_regression(request: Request, include_positive: bool = False):
    user = current_user(request)
    if user is None:
        return JSONResponse(
            {"error": "unauthorized", "message": "请先登录"}, status_code=401)
    samples = observability.export_regression_samples(
        user, negative_only=not include_positive)
    response = JSONResponse({
        "schema_version": 1,
        "kind": "minibrain_feedback_regression_candidates",
        "requires_human_review": True,
        "samples": samples,
    })
    response.headers["Content-Disposition"] = (
        'attachment; filename="minibrain-feedback-regression.json"')
    return response


@app.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request, hours: int = 24):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "metrics.html",
        {"user": user, "metrics": observability.metrics_overview(user, hours=hours)},
    )


# ---------------------------------------------------------------- 运维

@app.get("/health")
def health():
    cfg = get_config()
    try:
        from ..modules.vector_rag.index_manifest import manifest_health
        vector_index = manifest_health()
    except Exception as exc:  # schema 尚未升级或数据库不可用时 health 仍返回可诊断状态
        vector_index = {
            "compatible": False,
            "error": getattr(exc, "code", type(exc).__name__),
        }
    return {
        "status": "ok",
        "modules": gateway.health(),
        "embedding_configured": cfg.embedding_configured,
        "agent_configured": cfg.agent_configured,
        "embedding_model": cfg.embedding_model,
        "embedding_dimensions": cfg.embedding_dimensions,
        "vector_index_manifest": vector_index,
    }
