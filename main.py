"""
دستیار پژوهشی هوشمند — پروژه آموزشی Function Calling و Tool Calling
===================================================================

این فایل کل بک‌اند است. جریان کار به‌صورت خطی و ساده نوشته شده تا بتوانی خط‌به‌خط بخوانی:

  ۱) موضوع پژوهش از کاربر گرفته می‌شود.
  ۲) تعریف ابزار search_web (JSON Schema) به مدل معرفی می‌شود.
  ۳) مدل درخواست فراخوانی ابزار (tool_calls) برمی‌گرداند.
  ۴) بک‌اند آرگومان ابزار را می‌خواند و تابع واقعی پایتون (search_web) را اجرا می‌کند.
  ۵) نتایج جست‌وجو با Compaction فشرده و سپس با قالب role="tool" به مدل برگردانده می‌شوند.
  ۶) مدل مقاله فارسی را تولید می‌کند.
  ۷) مقاله در پوشه research_notes با UTF-8 ذخیره و مسیر آن برگردانده می‌شود.

نکته آموزشی مهم: مدل هرگز خودش تابع پایتون را اجرا نمی‌کند؛ فقط «درخواست» اجرا می‌دهد
و اجرای واقعی کار بک‌اند است.
"""

import asyncio
import html
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, Field

# کتابخانهی HTTP زیربنای کلاینت openai. نام این ماژول در همهی محیطها یکی نیست،
# پس هر دو حالت پوشش داده میشود (برای تنظیم دقیق «مهلت اتصال» لازم است).
try:
    import httpx as _httpx
except ImportError:  # pragma: no cover - فقط در محیطهایی که نام ماژول فرق دارد
    import httpx2 as _httpx

# ════════════════════════════════════════════════════════════════════════════
# ۱) تنظیمات — همه‌چیز از فایل .env خوانده می‌شود (هیچ کلیدی داخل کد نیست)
# ════════════════════════════════════════════════════════════════════════════

# load_dotenv() فایل .env را از پوشه همین فایل و اگر نبود از پوشه‌های بالاتر می‌خواند.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dastyar")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
NOTES_DIR = (BASE_DIR / ((os.getenv("RESEARCH_NOTES_DIR") or "research_notes").strip())).resolve()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-4o-mini"

# سرویس جست‌وجو: wikipedia (بدون کلید) یا tavily (نیازمند SEARCH_API_KEY)
SEARCH_PROVIDER = (os.getenv("SEARCH_PROVIDER") or "wikipedia").strip().lower()
SEARCH_API_KEY = (os.getenv("SEARCH_API_KEY") or "").strip()


def _env_int(name: str, default: int) -> int:
    """خواندن یک تنظیم عددی از .env به‌صورت امن."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("مقدار %s نامعتبر است (%r)؛ مقدار پیش‌فرض %s استفاده می‌شود.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """خواندن یک تنظیم اعشاری از .env به‌صورت امن."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("مقدار %s نامعتبر است (%r)؛ مقدار پیش‌فرض %s استفاده می‌شود.", name, raw, default)
        return default


SEARCH_MAX_RESULTS = _env_int("SEARCH_MAX_RESULTS", 5)
SEARCH_TIMEOUT = _env_float("SEARCH_TIMEOUT", 25.0)

# ---------- مهلتهای تماس با مدل ----------
# نکته مهم: کلاینت openai بهصورت پیشفرض فقط ۵ ثانیه برای «برقراری اتصال» مهلت
# میدهد. روی شبکههای کند (یا پشت پروکسی/VPN) همین مهلت کوتاه باعث خطای
# «ارتباط با مدل برقرار نشد» میشود، در حالی که تنظیمات .env کاملاً درست است.
# پس مهلت اتصال را جدا و قابلتنظیم میکنیم.
OPENAI_CONNECT_TIMEOUT = _env_float("OPENAI_CONNECT_TIMEOUT", 30.0)
OPENAI_READ_TIMEOUT = _env_float("OPENAI_READ_TIMEOUT", 300.0)
OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 2)

USER_AGENT = "research-assistant-demo/1.0 (educational project)"

app = FastAPI(
    title="دستیار پژوهشی هوشمند",
    description="پروژه آموزشی Function Calling / Tool Calling با FastAPI و مدل سازگار با OpenAI",
    version="1.0.0",
)


class ResearchError(Exception):
    """خطای قابل‌نمایش به کاربر.

    پیام این خطا مستقیماً در رابط کاربری نشان داده می‌شود، پس نباید شامل
    کلید API یا جزئیات حساس باشد؛ جزئیات کامل فقط در لاگ سرور ثبت می‌شود.
    """


# ════════════════════════════════════════════════════════════════════════════
# ۲) تعریف ابزار برای مدل (اینجا «Tool Definition» را می‌سازیم)
# ════════════════════════════════════════════════════════════════════════════
# این ساختار JSON Schema است و دقیقاً همان چیزی است که برای مدل فرستاده می‌شود.
# مدل هیچ‌وقت خود تابع پایتون را نمی‌بیند؛ فقط نام، توضیح و ورودی‌های آن را می‌بیند.
SEARCH_WEB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "جست‌وجوی اینترنتی برای جمع‌آوری اطلاعات، منبع و لینک درباره یک موضوع. "
            "برای نوشتن مقاله پژوهشی، اول این ابزار را صدا بزن."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "عبارت جست‌وجو به زبان فارسی یا انگلیسی، مثلاً: کاربرد هوش مصنوعی در پزشکی",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """تو یک دستیار پژوهشی فارسی‌زبان هستی.

قواعد کار:
۱) اگر برای پاسخ به اطلاعات بیرونی نیاز داری، حتماً از ابزار search_web استفاده کن
   و از خودت اطلاعات نساز.
۲) بعد از دریافت نتایج ابزار، یک مقاله فارسی با این ساختار بنویس:
   عنوان مقاله (در خط اول با یک علامت #)، مقدمه، بخش‌های اصلی، جمع‌بندی، منابع.
۳) منابع را فقط از نتایج واقعی جست‌وجو (همان عنوان‌ها و لینک‌هایی که در ابزار آمده) بنویس؛
   هیچ لینک یا منبع ساختگی نساز.
۴) اگر اطلاعات جمع‌آوری‌شده کافی نیست، همین محدودیت را صادقانه در مقاله بیان کن.
۵) خروجی نهایی فقط به قالب Markdown باشد و از عنوان‌های سطح دو (##) برای بخش‌ها استفاده کن.
"""

# ════════════════════════════════════════════════════════════════════════════
# ۳) تابع واقعی جست‌وجو (همان تابعی که مدل درخواست اجرای آن را می‌دهد)
# ════════════════════════════════════════════════════════════════════════════
# این تابع هیچ داده ساختگی برنمی‌گرداند: یا نتیجه واقعی می‌دهد یا خطای واضح می‌دهد.


def search_web(query: str) -> list[dict[str, str]]:
    """جست‌وجوی واقعی اینترنت و برگرداندن لیستی از {title, url, snippet}.

    سرویس جست‌وجو با متغیر SEARCH_PROVIDER در فایل .env انتخاب می‌شود:
      - wikipedia : جست‌وجو در ویکی‌پدیای فارسی با MediaWiki API (بدون کلید)
      - tavily    : جست‌وجوی عمومی وب با Tavily API (نیازمند SEARCH_API_KEY)
    """
    query = (query or "").strip()
    if not query:
        raise ResearchError("عبارت جست‌وجو خالی است؛ مدل آرگومان معتبری نفرستاده است.")

    logger.info("search_web → provider=%s query=%r", SEARCH_PROVIDER, query)

    if SEARCH_PROVIDER in ("wikipedia", "wiki", "fa-wikipedia"):
        return _search_wikipedia(query)
    if SEARCH_PROVIDER == "tavily":
        return _search_tavily(query)

    raise ResearchError(
        "سرویس جست‌وجوی انتخاب‌شده پشتیبانی نمی‌شود. مقدار SEARCH_PROVIDER را "
        "روی wikipedia یا tavily بگذار."
    )


def _strip_html_tags(text: str) -> str:
    """حذف تگ‌های HTML از خلاصه نتایج (ویکی‌پدیا کلمه‌های کلیدی را تگ‌گذاری می‌کند)."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(text or ""))).strip()


def _search_wikipedia(query: str) -> list[dict[str, str]]:
    """جست‌وجو در ویکی‌پدیای فارسی با MediaWiki API (بدون نیاز به کلید)."""
    try:
        response = requests.get(
            "https://fa.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": SEARCH_MAX_RESULTS,
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.exception("خطا در اتصال به ویکی‌پدیا")
        raise ResearchError(
            "ارتباط با سرویس جست‌وجو (ویکی‌پدیا) برقرار نشد. اتصال اینترنت را بررسی کن."
        ) from exc
    except ValueError as exc:  # پاسخ JSON نامعتبر
        logger.exception("پاسخ نامعتبر از ویکی‌پدیا")
        raise ResearchError("پاسخ سرویس جست‌وجو نامعتبر بود.") from exc

    results: list[dict[str, str]] = []
    for hit in data.get("query", {}).get("search", []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        results.append(
            {
                "title": title,
                "url": "https://fa.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "snippet": _strip_html_tags(hit.get("snippet", ""))[:400],
            }
        )

    if not results:
        logger.warning("ویکی‌پدیا برای %r نتیجه‌ای نداشت", query)
    return results


def _search_tavily(query: str) -> list[dict[str, str]]:
    """جست‌وجوی عمومی وب با Tavily API (نیازمند SEARCH_API_KEY در فایل .env)."""
    if not SEARCH_API_KEY:
        raise ResearchError(
            "سرویس جست‌وجوی Tavily انتخاب شده اما SEARCH_API_KEY تنظیم نشده است. "
            "کلید رایگان را از tavily.com بگیر و در فایل .env بگذار."
        )

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": SEARCH_API_KEY,
                "query": query,
                "max_results": SEARCH_MAX_RESULTS,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.exception("خطا در اتصال به Tavily")
        raise ResearchError(
            "ارتباط با سرویس جست‌وجوی Tavily برقرار نشد. کلید API و اتصال شبکه را بررسی کن."
        ) from exc
    except ValueError as exc:
        logger.exception("پاسخ نامعتبر از Tavily")
        raise ResearchError("پاسخ سرویس جست‌وجو نامعتبر بود.") from exc

    results: list[dict[str, str]] = []
    for item in data.get("results", []):
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": (item.get("title") or url).strip(),
                "url": url,
                "snippet": _strip_html_tags(item.get("content", ""))[:400],
            }
        )

    if not results:
        logger.warning("Tavily برای %r نتیجه‌ای نداشت", query)
    return results

# ════════════════════════════════════════════════════════════════════════════
# ۴) ابزارهای کمکی و ذخیره مقاله
# ════════════════════════════════════════════════════════════════════════════

# عنوان مراحل — همین شش مرحله در رابط کاربری هم نمایش داده می‌شود
STEP_TITLES = {
    "topic": "دریافت موضوع پژوهش",
    "decide": "تصمیم مدل برای استفاده از ابزار جست‌وجو",
    "search": "جست‌وجوی اینترنت",
    "send": "ارسال نتایج به مدل",
    "article": "تولید مقاله",
    "save": "ذخیره مقاله روی لپ‌تاپ",
}

RUNNING, DONE, SKIPPED, ERROR = "running", "done", "skipped", "error"


def _step(key: str, status: str, detail: str = "") -> dict[str, str]:
    """ساختن یک رویداد «مرحله» که به فرانت‌اند فرستاده می‌شود."""
    return {"type": "step", "key": key, "title": STEP_TITLES[key], "status": status, "detail": detail}


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """خواندن آرگومان‌های ابزار که مدل فرستاده است (رشته JSON → دیکشنری).

    اگر مدل JSON نامعتبر بفرستد، خطا را لاگ می‌کنیم و دیکشنری خالی برمی‌گردانیم
    تا بک‌اند بتواند با مقدار پیش‌فرض (موضوع پژوهش) ادامه دهد.
    """
    try:
        data = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        logger.warning("آرگومان‌های ابزار JSON نامعتبر بودند: %r", raw_arguments)
        return {}
    return data if isinstance(data, dict) else {}


def compact_search_results(
    results: list[dict[str, str]],
    max_results: int = 3,
    snippet_limit: int = 250,
) -> list[dict[str, str]]:
    """فشرده‌سازی نتایج جست‌وجو پیش از ارسال به مدل (Compaction آموزشی).

    فقط سه نتیجه اول را نگه می‌داریم، عنوان را کوتاه می‌کنیم و متن هر خلاصه
    را به حداکثر ۲۵۰ نویسه می‌رسانیم. عنوان و URL حفظ می‌شوند تا مدل بتواند
    منبع را بشناسد. این کار خلاصه‌سازی هوشمند با مدل نیست؛ کاهش ساده حجم Context است.
    """
    compacted: list[dict[str, str]] = []

    for result in results[:max_results]:
        compacted.append(
            {
                "title": (result.get("title") or "")[:120],
                "url": result.get("url") or "",
                "snippet": (result.get("snippet") or "")[:snippet_limit],
            }
        )

    original_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in results)
    compact_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in compacted)
    logger.info(
        "Compaction: %s نتیجه و %s نویسه → %s نتیجه و %s نویسه",
        len(results), original_chars, len(compacted), compact_chars,
    )
    return compacted


def _unique_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """حذف منابع تکراری بر اساس لینک."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for source in sources:
        url = source.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(source)
    return unique


def _make_summary(article: str, limit: int = 400) -> str:
    """خلاصه کوتاه از متن مقاله (اولین پاراگراف‌های غیرعنوان) برای نمایش در UI."""
    parts: list[str] = []
    for line in article.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        parts.append(line)
        if sum(len(p) for p in parts) >= limit:
            break
    summary = " ".join(parts)
    return (summary[:limit] + "…") if len(summary) > limit else summary


def _sources_markdown(sources: list[dict[str, str]]) -> str:
    """ساختن بخش «منابع» از نتایج واقعی جست‌وجو."""
    lines = ["## منابع (لینک‌های واقعی جست‌وجو)", ""]
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or source.get("url")
        lines.append(f"{index}. [{title}]({source.get('url')})")
        if source.get("snippet"):
            lines.append(f"   - خلاصه: {source['snippet']}")
    lines.append("")
    return "\n".join(lines)


def save_article(topic: str, article: str, sources: list[dict[str, str]]) -> Path:
    """ذخیره مقاله در پوشه research_notes با کدگذاری UTF-8 و برگرداندن مسیر فایل."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = NOTES_DIR / f"research-{stamp}.md"

    body = article.strip()
    # اگر مقاله عنوان سطح یک (#) ندارد، عنوانی از روی موضوع پژوهش اضافه می‌کنیم
    if not re.match(r"^#\s+\S", body):
        body = f"# مقاله پژوهشی: {topic}\n\n{body}"

    footer = (
        "\n---\n"
        f"*موضوع: {topic} | مدل: {OPENAI_MODEL} | سرویس جست‌وجو: {SEARCH_PROVIDER} "
        f"| تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    content = body
    if sources:
        content += "\n\n" + _sources_markdown(sources)
    content += "\n" + footer + "\n"

    try:
        file_path.write_text(content, encoding="utf-8")  # ذخیره با UTF-8 برای متن فارسی
    except OSError as exc:
        logger.exception("خطا در ذخیره فایل مقاله")
        raise ResearchError(
            "ذخیره فایل مقاله ممکن نشد. دسترسی نوشتن در پوشه research_notes را بررسی کن."
        ) from exc

    logger.info("مقاله ذخیره شد: %s", file_path)
    return file_path


def get_client() -> AsyncOpenAI:
    """ساخت کلاینت مدل. base_url اختیاری است و از .env خوانده می‌شود."""
    if not OPENAI_API_KEY:
        raise ResearchError(
            "کلید API مدل تنظیم نشده است. فایل .env را از روی .env.example بساز و "
            "OPENAI_API_KEY را در آن بگذار."
        )
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
        # مهلت اتصال از پیشفرض ۵ ثانیهای openai بیشتر است تا روی شبکههای کند
        # خطای بیمورد «ارتباط برقرار نشد» نگیریم.
        timeout=_httpx.Timeout(OPENAI_READ_TIMEOUT, connect=OPENAI_CONNECT_TIMEOUT),
        max_retries=OPENAI_MAX_RETRIES,
    )


# ════════════════════════════════════════════════════════════════════════════
# ۵) قلب پروژه: اجرای پژوهش با Tool Calling
# ════════════════════════════════════════════════════════════════════════════


def _model_error_message(exc: Exception) -> str:
    """تبدیل خطای فنیِ مدل به پیام فارسیِ دقیق برای کاربر.

    پیشتر برای همهی خطاها یک پیام ثابت نمایش داده میشد و کاربر را به سمت
    تنظیمات .env میفرستاد؛ در حالی که بیشتر اوقات مشکل «شبکه یا سرویس» است و
    تنظیمات .env درست است. اینجا خطاها از هم جدا میشوند.
    """
    service = OPENAI_BASE_URL or "سرویس پیشفرض OpenAI"

    if isinstance(exc, APITimeoutError):
        return (
            "مهلت تماس با سرویس مدل تمام شد. "
            f"دسترسی به {service} را بررسی کن یا OPENAI_CONNECT_TIMEOUT و "
            "OPENAI_READ_TIMEOUT را در فایل .env بیشتر کن."
        )
    if isinstance(exc, APIConnectionError):
        return (
            "ارتباط با سرویس مدل برقرار نشد (خطای شبکه). "
            f"دسترسی به {service} و تنظیمات پروکسی/VPN را بررسی کن."
        )
    if isinstance(exc, AuthenticationError):
        return "کلید API مدل پذیرفته نشد؛ مقدار OPENAI_API_KEY را در فایل .env بررسی کن."
    if isinstance(exc, PermissionDeniedError):
        return "این کلید API اجازهی استفاده از مدل را ندارد؛ حساب سرویس را بررسی کن."
    if isinstance(exc, NotFoundError):
        return (
            f"مدل «{OPENAI_MODEL}» در این سرویس پیدا نشد؛ "
            "مقدار OPENAI_MODEL را در فایل .env بررسی کن."
        )
    if isinstance(exc, RateLimitError):
        return "سرویس مدل فعلاً درخواستها را محدود کرده است؛ چند لحظه بعد دوباره امتحان کن."
    if isinstance(exc, BadRequestError):
        return (
            f"درخواست به مدل نامعتبر بود. احتمالاً مدل «{OPENAI_MODEL}» از tool calling "
            "پشتیبانی نمیکند؛ در فایل .env یک مدل دیگر بگذار."
        )
    if isinstance(exc, APIStatusError):
        return f"سرویس مدل خطای {exc.status_code} برگرداند؛ جزئیات کامل در لاگ سرور است."

    return (
        "ارتباط با مدل هوش مصنوعی برقرار نشد. مقدارهای OPENAI_API_KEY، OPENAI_BASE_URL "
        "و OPENAI_MODEL را در فایل .env بررسی کن."
    )


async def _ask_model(client: AsyncOpenAI, messages: list[dict[str, Any]], with_tools: bool) -> Any:
    """ارسال درخواست به مدل.

    اگر with_tools=True باشد، تعریف ابزار search_web هم فرستاده می‌شود؛
    یعنی مدل «می‌فهمد» چنین ابزاری در اختیار دارد و می‌تواند درخواست اجرایش را بدهد.
    """
    # نکته مهم: وقتی نمیخواهیم ابزاری معرفی شود، کلید «tools» باید کاملاً
    # فرستاده نشود، نه با مقدار None. کتابخانه openai مقدار None را بهصورت
    # «"tools": null» در بدنه JSON میگذارد و برخی سرویسهای سازگار با OpenAI
    # (مثل TokenHarbor) همین null را «درخواست نامعتبر» میدانند و خطای 400 میدهند.
    request_kwargs: dict[str, Any] = {"model": OPENAI_MODEL, "messages": messages}
    if with_tools:
        request_kwargs["tools"] = [SEARCH_WEB_TOOL]  # ← معرفی ابزار به مدل
        request_kwargs["tool_choice"] = "auto"       # ← مدل خودش تصمیم میگیرد

    try:
        return await client.chat.completions.create(**request_kwargs)
    except Exception as exc:  # خطای شبکه، کلید نامعتبر، مدل اشتباه و ...
        logger.exception("خطا در ارتباط با مدل هوش مصنوعی")
        raise ResearchError(_model_error_message(exc)) from exc


async def run_research(topic: str) -> AsyncIterator[dict[str, Any]]:
    """اجرای کدام‌به‌کدام مراحل پژوهش و تولید مقاله.

    این تابع یک «ژنراتور ناهمگام» است: هر مرحله را به‌صورت رویداد (event) بیرون می‌دهد،
    بنابراین رابط کاربری می‌تواند مراحل را همان لحظه‌ای که اجرا می‌شوند ببیند.
    """
    topic = (topic or "").strip()

    # ---------- مرحله ۱: دریافت موضوع پژوهش ----------
    yield _step("topic", RUNNING)
    if not topic:
        yield _step("topic", ERROR, "موضوعی وارد نشده است")
        raise ResearchError("موضوع پژوهش خالی است. یک موضوع واقعی وارد کن.")
    yield _step("topic", DONE, f"موضوع: {topic}")

    client = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"این موضوع را پژوهش کن و مقاله فارسی بنویس: {topic}"},
    ]

    # ---------- مرحله ۲: تصمیم مدل برای استفاده از ابزار ----------
    yield _step("decide", RUNNING, "درخواست به مدل فرستاده شد و ابزار search_web معرفی شد")
    first_response = await _ask_model(client, messages, with_tools=True)
    model_message = first_response.choices[0].message
    tool_calls = list(model_message.tool_calls or [])  # ← اینجا tool_calls را می‌خوانیم

    # پیام مدل (که شامل tool_calls است) باید در تاریخچه مکالمه بماند
    messages.append(model_message)

    if tool_calls:
        names = "، ".join(call.function.name for call in tool_calls)
        yield _step("decide", DONE, f"مدل درخواست اجرای ابزار «{names}» را داد")
    else:
        yield _step("decide", DONE, "مدل نیازی به ابزار ندید و مستقیم پاسخ داد")

    sources: list[dict[str, str]] = []
    article = ""

    if tool_calls:
        # ---------- مراحل ۳ و ۴: جست‌وجو و برگرداندن نتیجه به مدل ----------
        for call in tool_calls:
            # خواندن آرگومان‌های ابزار (رشته JSON که مدل ساخته است)
            arguments = _parse_tool_arguments(call.function.arguments)
            query = str(arguments.get("query") or topic).strip()

            yield _step("search", RUNNING, f"عبارت جست‌وجو: {query}")
            try:
                # اجرای واقعی تابع پایتون در ترد جدا تا حلقه رویداد بلاک نشود
                results = await asyncio.to_thread(search_web, query)
            except ResearchError as exc:
                yield _step("search", ERROR, str(exc))
                raise
            # منابع کامل را برای نمایش و ذخیره مقاله نگه می‌داریم.
            sources.extend(results)

            # Compaction: قبل از ارسال نتیجه به مدل، Context را کوچک می‌کنیم.
            compact_results = compact_search_results(results)
            yield _step(
                "search", DONE,
                f"{len(results)} نتیجه دریافت شد؛ {len(compact_results)} نتیجه فشرده به مدل می‌رود",
            )

            # فقط نسخه فشرده با نقش tool به تاریخچه مکالمه اضافه می‌شود.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        {"query": query, "results": compact_results}, ensure_ascii=False
                    ),
                }
            )

        yield _step("send", DONE, f"نتیجه {len(tool_calls)} فراخوانی ابزار به مدل برگردانده شد")

        # ---------- مرحله ۵: تولید مقاله بر اساس نتایج جست‌وجو ----------
        yield _step("article", RUNNING, "مدل در حال نوشتن مقاله است")
        final_response = await _ask_model(client, messages, with_tools=False)
        article = (final_response.choices[0].message.content or "").strip()
    else:
        yield _step("search", SKIPPED, "مدل تشخیص داد برای این موضوع جست‌وجو لازم نیست")
        yield _step("send", SKIPPED, "نتیجه‌ای برای ارسال نبود")
        yield _step("article", RUNNING, "مدل در حال نوشتن مقاله است")
        article = (model_message.content or "").strip()

    if not article:
        yield _step("article", ERROR, "پاسخ مدل خالی بود")
        raise ResearchError("مدل متن مقاله را تولید نکرد؛ یک بار دیگر امتحان کن.")
    yield _step("article", DONE, f"{len(article)} نویسه نوشته شد")

    # ---------- مرحله ۶: ذخیره مقاله روی لپ‌تاپ ----------
    unique_sources = _unique_sources(sources)
    yield _step("save", RUNNING, f"پوشه ذخیره: {NOTES_DIR}")
    file_path = save_article(topic, article, unique_sources)
    yield _step("save", DONE, str(file_path))

    try:
        saved_content = file_path.read_text(encoding="utf-8")
    except OSError:  # اگر خواندن فایل ناموفق بود، همان متن مقاله را برمی‌گردانیم
        saved_content = article

    # ---------- رویداد پایانی: نتیجه کامل پژوهش ----------
    yield {
        "type": "done",
        "result": {
            "topic": topic,
            "model": OPENAI_MODEL,
            "search_provider": SEARCH_PROVIDER,
            "used_tool": bool(tool_calls),
            "summary": _make_summary(article),
            "article_markdown": saved_content,
            "sources": unique_sources,
            "file_path": str(file_path),
            "file_name": file_path.name,
        },
    }

# ════════════════════════════════════════════════════════════════════════════
# ۶) مدل‌های داده و API ها
# ════════════════════════════════════════════════════════════════════════════


class ResearchRequest(BaseModel):
    """ورودی کاربر: فقط موضوع پژوهش."""

    topic: str = Field(default="", description="موضوع پژوهش", examples=["کاربرد هوش مصنوعی در پزشکی"])


class SourceModel(BaseModel):
    """یک منبع واقعی که از جست‌وجو به دست آمده است."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class ResearchResponse(BaseModel):
    """خروجی نهایی پژوهش."""

    topic: str
    model: str
    search_provider: str
    used_tool: bool
    summary: str
    article_markdown: str
    sources: list[SourceModel]
    file_path: str
    file_name: str


def _sse(event: dict[str, Any]) -> str:
    """قالب استاندارد Server-Sent Events: هر رویداد با یک خط خالی تمام می‌شود."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/health", summary="بررسی تنظیمات و سلامت سرور (برای عیب‌یابی)")
def api_health() -> dict[str, Any]:
    """نشان می‌دهد کدام تنظیمات خوانده شده‌اند — بدون افشای مقدار کلیدها."""
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "model_api_key_set": bool(OPENAI_API_KEY),
        "connect_timeout": OPENAI_CONNECT_TIMEOUT,
        "read_timeout": OPENAI_READ_TIMEOUT,
        "max_retries": OPENAI_MAX_RETRIES,
        "base_url": OPENAI_BASE_URL or "(پیش‌فرض سرویس)",
        "search_provider": SEARCH_PROVIDER,
        "search_api_key_set": bool(SEARCH_API_KEY),
        "notes_dir": str(NOTES_DIR),
        "notes_dir_exists": NOTES_DIR.exists(),
    }


@app.post("/api/research", response_model=ResearchResponse, summary="اجرای کامل پژوهش (نتیجه یکجا)")
async def api_research(request: ResearchRequest) -> dict[str, Any]:
    """نسخه ساده و بدون جریان: مراحل در سرور اجرا می‌شوند و نتیجه کامل برگردانده می‌شود.

    این همان مسیری است که با Swagger یا curl می‌توانی بدون فرانت‌اند آزمایش کنی.
    """
    result: dict[str, Any] | None = None
    try:
        async for event in run_research(request.topic):
            if event["type"] == "done":
                result = event["result"]
    except ResearchError as exc:
        # پیام خطای قابل‌فهم به کاربر؛ جزئیات فنی در لاگ سرور ثبت شده است
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:  # از نظر منطقی نباید رخ دهد، ولی برای اطمینان
        raise HTTPException(status_code=500, detail="پژوهش کامل نشد.")
    return result


@app.get("/api/research/stream", summary="اجرای پژوهش با نمایش زنده مراحل (SSE)")
async def api_research_stream(topic: str = Query("", description="موضوع پژوهش")) -> StreamingResponse:
    """همان منطق بالا، اما مراحل همان لحظه‌ای که اجرا می‌شوند به مرورگر فرستاده می‌شوند."""

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in run_research(topic):
                yield _sse(event)
        except ResearchError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001 — خطای پیش‌بینی‌نشده: جزئیات فقط در لاگ سرور
            logger.exception("خطای پیش‌بینی‌نشده در جریان پژوهش")
            yield _sse({"type": "error", "message": "خطای پیش‌بینی‌نشده در سرور رخ داد. لاگ سرور را ببین."})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# فایل‌های فرانت‌اند (HTML / CSS / JS) از پوشه static سرو می‌شوند
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """نمایش صفحه چت دستیار پژوهشی."""
    return FileResponse(STATIC_DIR / "index.html")


# سازگاری با کش مرورگر: مرورگرهایی که نسخه قدیمی صفحه را در کش دارند، فایل‌های
# استایل و اسکریپت را با مسیر نسبی (بدون /static) درخواست می‌کنند.
# این دو مسیر همان فایل‌ها را برمی‌گردانند تا برنامه در هر حالت کار کند.
@app.get("/style.css", include_in_schema=False)
def style_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "style.css")


@app.get("/script.js", include_in_schema=False)
def script_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "script.js")


if __name__ == "__main__":
    import uvicorn

    # اجرای ساده: python main.py
    # مستندات Swagger: http://127.0.0.1:8000/docs
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

