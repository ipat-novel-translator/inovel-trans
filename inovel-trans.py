#!/usr/bin/env python3
"""
inovel-trans.py - ระบบแปลนิยายจีนเป็นไทยอัจฉริยะ
รองรับ: Gemini 2.5 Flash/Pro + Fallback + .env + metadata
"""

import os
import sys
import re
import json
import time
import shutil
import logging
import datetime
import random
import subprocess
from typing import Dict, List, Any, Tuple
import structlog
import yaml
import argparse

# โหลด .env
from dotenv import load_dotenv
load_dotenv()

# Aho-Corasick (optional)
try:
    import ahocorasick
    AHO_AVAILABLE = True
except ImportError:
    AHO_AVAILABLE = False

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("ERROR: ติดตั้ง google-generativeai: pip install -U google-generativeai")
    sys.exit(1)

# tiktoken (สำหรับประมาณ token)
try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False


# ---------------------------
# 🔒 Custom Exception
# ---------------------------
class SafetyBlockError(Exception):
    """Raised when Gemini blocks content due to safety filters"""
    pass


# ---------------------------
# 🔍 ตรวจสอบโมเดลที่มีอยู่จริง
# ---------------------------
def get_available_models(preferred_models: List[str]) -> List[str]:
    """คืนลิสต์โมเดลที่มีอยู่จริง (ตามลำดับ preferred)"""
    try:
        available = set()
        for m in genai.list_models():
            if 'generateContent' in (m.supported_generation_methods or []):
                name = m.name.replace("models/", "")
                available.add(name)
        structlog.get_logger().debug("Available models", models=sorted(available))
    except Exception as e:
        structlog.get_logger().warning("Cannot fetch model list", error=str(e))
        # ใช้ whitelist แบบปลอดภัย
        available = {
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-pro"
        }

    filtered = [model.replace("models/", "") for model in preferred_models if model.replace("models/", "") in available]
    if not filtered:
        fallback = ["gemini-1.5-flash"]
        structlog.get_logger().error("❌ ไม่มีโมเดลที่ต้องการ! ใช้ fallback", fallback=fallback)
        return fallback
    return filtered


# ---------------------------
# 🔧 Configuration
# ---------------------------
class Config:
    def __init__(self):
        raw_model = os.getenv(
            "TRANSLATION_MODEL",
            "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash,gemini-2.5-pro"
        )
        self.preferred_models = [m.strip() for m in raw_model.split(",")]
        self.available_models = get_available_models(self.preferred_models)
        self.model = self.available_models[0]

        self.token_limit = int(os.getenv("TOKEN_LIMIT", "8000"))
        # ✅ ลดค่า default เป็น 1500
        self.chunk_token_target = int(os.getenv("CHUNK_TOKEN_TARGET", "1500"))
        self.token_buffer = int(os.getenv("TOKEN_BUFFER", "400"))

        self.src_dir = os.getenv("SRC_DIR", "chapters_src")
        self.out_dir = os.getenv("OUT_DIR", "chapters_translated")
        self.log_dir = os.getenv("LOG_DIR", "logs")
        self.backup_dir = os.getenv("BACKUP_DIR", "backups")
        self.glossary_file = os.getenv("GLOSSARY_FILE", "glossary.yaml")
        self.cache_db = os.path.join(self.log_dir, "translation_cache.db")

        for path in [self.src_dir, self.out_dir, self.log_dir, self.backup_dir]:
            os.makedirs(path, exist_ok=True)


config = Config()


# ---------------------------
# Logging
# ---------------------------
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("ip_trans")


# ---------------------------
# 🔌 ตั้งค่า Gemini API
# ---------------------------
# ตรวจสอบว่าใช้ Vertex AI หรือ AI Studio
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # ใช้ Vertex AI (ไม่ต้องตั้ง GEMINI_API_KEY)
    logger.info("ใช้ Vertex AI", credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("ต้องตั้งค่า GEMINI_API_KEY หรือ GOOGLE_APPLICATION_CREDENTIALS ใน .env")
        sys.exit(1)
    genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------
# 💾 Cache System
# ---------------------------
import sqlite3

def init_cache_db():
    with sqlite3.connect(config.cache_db) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                hanzi TEXT PRIMARY KEY,
                thai TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

translation_cache = {}

def load_cache():
    translation_cache.clear()
    try:
        with sqlite3.connect(config.cache_db) as conn:
            cursor = conn.execute("SELECT hanzi, thai FROM translations")
            for row in cursor.fetchall():
                translation_cache[row[0]] = row[1]
        logger.info("Cache โหลดแล้ว", terms=len(translation_cache))
    except Exception as e:
        logger.warning("โหลด cache ล้มเหลว", error=str(e))

def save_cache():
    try:
        with sqlite3.connect(config.cache_db) as conn:
            cursor = conn.cursor()
            for hanzi, thai in translation_cache.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO translations (hanzi, thai, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (hanzi, thai))
            conn.commit()
    except Exception as e:
        logger.warning("บันทึก cache ล้มเหลว", error=str(e))


# ---------------------------
# 🔄 Retry Mechanism
# ---------------------------
def gemini_with_retry(call_func, max_retries=3):
    for i in range(max_retries):
        try:
            return call_func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ["rate limit", "timeout", "quota", "502", "503"]):
                wait = (2 ** i) + random.uniform(0, 1)
                logger.warning("ลองใหม่", attempt=i+1, wait=round(wait,1))
                time.sleep(wait)
            else:
                raise


# ---------------------------
# ✨ ฟังก์ชันแปลที่รับ model_name ได้
# ---------------------------
def translate_with_gemini(
    prompt: str,
    model_name: str,
    max_tokens: int = 2000,
    temperature: float = 0.2
) -> str:
    def make_call():
        model = genai.GenerativeModel(
            model_name,
            system_instruction="คุณเป็นผู้แปลนิยายจีนโบราณ ไม่ต้องวิเคราะห์เนื้อหา แปลตรงๆ ตามต้นฉบับ",
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ]
        )
        return model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40
            )
        )
    
    resp = gemini_with_retry(make_call)
    
    if not resp.candidates:
        raise ValueError(f"ไม่มีคำตอบจากโมเดล. finish_reason: {resp.prompt_feedback}")
    
    candidate = resp.candidates[0]
    if not candidate.content.parts:
        reason = candidate.finish_reason
        reason_name = {1: "STOP", 2: "SAFETY", 3: "RECITATION", 4: "OTHER"}.get(reason, reason)
        if reason == 2:  # SAFETY
            raise SafetyBlockError(f"เนื้อหาถูกบล็อกด้วย safety filter (finish_reason: {reason_name})")
        else:
            raise ValueError(f"คำตอบว่าง. finish_reason: {reason_name} ({reason})")
    
    return candidate.content.parts[0].text.strip()


# ---------------------------
# 🔄 ฟังก์ชัน fallback แบบฉลาด
# ---------------------------
def translate_with_model_fallback(
    prompt: str,
    preferred_models: List[str],
    max_tokens: int = 2000,
    temperature: float = 0.2
) -> Tuple[str, str]:
    """
    ลองแปลด้วยลิสต์โมเดลทีละตัว จนกว่าจะสำเร็จ
    คืนค่า: (ข้อความแปล, ชื่อโมเดลที่ใช้ได้)
    """
    for model in preferred_models:
        try:
            logger.info("ลองแปลด้วยโมเดล", model=model)
            translation = translate_with_gemini(
                prompt=prompt,
                model_name=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return translation, model
        except SafetyBlockError as e:
            logger.warning("โมเดลถูกบล็อกด้วย safety — ลองโมเดลถัดไป", model=model, error=str(e))
            continue  # ลองโมเดลถัดไป!
        except Exception as e:
            logger.error("โมเดลล้มเหลวด้วยเหตุอื่น — หยุดทันที", model=model, error=str(e))
            raise  # ไม่ fallback ถ้าเป็น error อื่น (เช่น network, quota)

    # ถ้ามาถึงตรงนี้ = ทุกโมเดลถูกบล็อกด้วย safety
    raise SafetyBlockError("ทุกโมเดลถูกบล็อกด้วย safety filters — ไม่สามารถแปลได้")


# ---------------------------
# Utilities
# ---------------------------
def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def backup_glossary(path=config.glossary_file):
    if os.path.exists(path):
        ts = now_ts()
        dst = os.path.join(config.backup_dir, f"{os.path.basename(path)}.{ts}.bak")
        shutil.copy(path, dst)
        logger.info("Glossary backed up", src=path, dst=dst)


# ---------------------------
# Load Glossary
# ---------------------------
def load_glossary() -> Dict[str, str]:
    if not os.path.exists(config.glossary_file):
        logger.info("No glossary found", path=config.glossary_file)
        return {}

    with open(config.glossary_file, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
            simple = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    if "th" in v:
                        simple[k] = v["th"]
                elif isinstance(v, str):
                    simple[k] = v
            return simple
        except Exception as e:
            logger.error("Load glossary failed", error=str(e))
            raise


# ---------------------------
# Save Glossary
# ---------------------------
def save_glossary(glossary_dict: Dict[str, str], chapter_num: int = None):
    path = config.glossary_file
    current_data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                current_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read existing glossary", error=str(e))

    added_count = 0
    updated_count = 0

    for hanzi, thai in glossary_dict.items():
        if hanzi in current_data:
            entry = current_data[hanzi]
            if isinstance(entry, dict):
                if entry.get("source_type") == "manual":
                    logger.debug("Skip update (manual override)", term=hanzi, old=entry.get("th"), new=thai)
                    continue
                else:
                    current_data[hanzi]["th"] = thai
                    if chapter_num:
                        current_data[hanzi]["chapter_first_seen"] = min(
                            current_data[hanzi].get("chapter_first_seen", chapter_num),
                            chapter_num
                        )
                    current_data[hanzi]["added_at"] = now_ts()
                    updated_count += 1
            else:
                current_data[hanzi] = {
                    "th": thai,
                    "chapter_first_seen": chapter_num,
                    "added_at": now_ts(),
                    "source_type": "auto",
                    "notes": ""
                }
                updated_count += 1
        else:
            current_data[hanzi] = {
                "th": thai,
                "chapter_first_seen": chapter_num,
                "added_at": now_ts(),
                "source_type": "auto",
                "notes": ""
            }
            added_count += 1

    backup_glossary(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(current_data, f, allow_unicode=True, sort_keys=False, indent=2)

    logger.info("Saved glossary", total=len(current_data), added=added_count, updated=updated_count, path=path)


# ---------------------------
# Aho-Corasick Utils
# ---------------------------
def build_automaton_from_dict(word_dict: dict):
    if not AHO_AVAILABLE:
        return None
    auto = ahocorasick.Automaton()
    for word in word_dict:
        auto.add_word(word, word)
    auto.make_automaton()
    return auto

def find_all_matched_terms(text: str, automaton) -> set:
    if automaton is None:
        return set()
    return {item[1] for item in automaton.iter(text)}


# ---------------------------
# Extract candidates (ใช้ Gemini)
# ---------------------------
def extract_candidates(text: str, existing_glossary: Dict[str, str]) -> List[str]:
    existing_keys = set(existing_glossary.keys())
    truncated_text = text[:4000]

    prompt = (
        "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายจีนโบราณแนวไซ่เซียน "
        "ให้สกัดคำเฉพาะทางใหม่ (ชื่อคน ชื่อสำนัก วิชา ตำแหน่ง ขั้นพลัง) จากข้อความต่อไปนี้ "
        "และส่งกลับเป็น JSON array เท่านั้น เช่น [\"คำ1\", \"คำ2\"] "
        "อย่ารวมคำทั่วไป หรือคำที่อยู่ในรายการที่ให้มา\n\n"
        f"[รายการคำศัพท์ที่มีอยู่แล้ว - อย่าส่งคืน]\n{list(existing_keys)}\n\n"
        f"[ข้อความต้นฉบับ]\n{truncated_text}\n\n"
        "ส่งคืนเฉพาะ JSON array ของคำเฉพาะใหม่:"
    )

    try:
        response_text = translate_with_gemini(
            prompt=prompt,
            model_name=config.model,  # ใช้โมเดล default สำหรับ extract
            max_tokens=300,
            temperature=0.0
        )

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = extract_json_safely(response_text)

        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = list(data.get("glossary", {}).keys()) or data.get("terms", [])
        else:
            candidates = []

        filtered = [
            w for w in candidates
            if isinstance(w, str)
            and len(w) >= 2
            and re.fullmatch(r"[\u4e00-\u9fff]+", w)
            and w not in existing_keys
        ]
        filtered.sort(key=lambda s: -len(s))
        return filtered

    except Exception as e:
        logger.error("สกัดคำศัพท์ล้มเหลว", error=str(e))
        return []


# ---------------------------
# Extract JSON Safely
# ---------------------------
def extract_json_safely(text: str) -> dict:
    stack = []
    start = None
    brackets = {'{': '}', '[': ']'}
    for i, c in enumerate(text):
        if c in '{[':
            if not stack:
                start = i
            stack.append(brackets[c])
        elif c in '}]' and stack:
            if c == stack.pop():
                if not stack and start is not None:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception as e:
                        logger.debug("Failed to parse partial JSON", pos=i, error=str(e))
            else:
                stack.clear()
                start = None
    raise ValueError("No valid JSON found")


# ---------------------------
# Propose translations with cache (ใช้ Gemini)
# ---------------------------
def propose_candidate_translations_json(candidates: List[str]) -> Dict[str,str]:
    results = {}
    unseen = [c for c in candidates if c not in translation_cache]
    for c in candidates:
        if c in translation_cache:
            results[c] = translation_cache[c]

    if not unseen:
        logger.info("ทุกคำศัพท์มีใน cache แล้ว", count=len(candidates))
        return results

    batch_size = 40

    for i in range(0, len(unseen), batch_size):
        batch = unseen[i:i+batch_size]
        terms_list = "\n".join(batch)
        prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายจีนโบราณแนวไซ่เซียนเป็นภาษาไทย "
            "ให้แปลคำศัพท์เฉพาะทางต่อไปนี้เป็นภาษาไทย ตามสไตล์นิยาย "
            "ใช้ภาษาไทยสมัยใหม่ เป็นธรรมชาติ ไม่เป็นทางการเกินไป "
            "ห้ามอธิบาย ห้ามเพิ่มเติม ให้ส่งคืนเฉพาะ JSON รูปแบบ: "
            "{\"glossary\": {\"汉字\":\"ไทย\", ...}}\n\n"
            f"คำศัพท์:\n{terms_list}"
        )

        try:
            response_text = translate_with_gemini(
                prompt=prompt,
                model_name=config.model,  # ใช้โมเดล default
                max_tokens=1500,
                temperature=0.0
            )

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = extract_json_safely(response_text)

            if isinstance(data, dict) and "glossary" in data:
                new_translations = {k.strip(): v.strip() for k,v in data["glossary"].items()}
                results.update(new_translations)
                translation_cache.update(new_translations)
            else:
                logger.warning("ไม่พบ glossary ในคำตอบ", raw=response_text[:200])

        except Exception as e:
            logger.error("เสนอคำแปลล้มเหลว", batch=i//batch_size, error=str(e))

    logger.info("เสนอคำแปล", new=len(results), cached=len(candidates)-len(unseen))
    return results


# ---------------------------
# Smart Chunking (เวอร์ชันแก้ไข — ไม่มีประโยคซ้ำ)
# ---------------------------
def smart_chunk_with_overlap(text: str, target_tokens: int = None) -> List[str]:
    """
    แบ่งข้อความเป็น chunk โดย:
    - ไม่มีการซ้ำประโยค (overlap = 0)
    - ใช้จุดจบประโยคจีน: 。！？；;
    - รักษาความลื่นไหลด้วยการไม่ตัดกลางประโยค
    """
    if target_tokens is None:
        target_tokens = config.chunk_token_target

    if HAVE_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        # แบ่งประโยคด้วย regex ที่ครอบคลุม
        sentences = re.split(r'(?<=[。！？；;])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text[:2000]]

        chunks = []
        current_chunk = ""

        for sent in sentences:
            # ตรวจสอบ token ถ้าเพิ่มประโยคนี้เข้าไป
            temp_chunk = current_chunk + sent
            token_count = len(enc.encode(temp_chunk))

            if token_count > target_tokens and current_chunk:
                # ปิด chunk นี้ (ไม่ซ้อนทับ!)
                chunks.append(current_chunk.strip())
                current_chunk = sent  # เริ่ม chunk ใหม่ด้วยประโยคนี้
            else:
                current_chunk += sent

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Safety net: ถ้ายังมี chunk ใหญ่เกิน → แบ่งด้วยบรรทัด
        final_chunks = []
        for chunk in chunks:
            if count_tokens(chunk) > target_tokens + 500:
                lines = chunk.split('\n')
                sub_chunk = ""
                for line in lines:
                    if count_tokens(sub_chunk + line) < target_tokens + 800:
                        sub_chunk += line + "\n"
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = line + "\n"
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks if final_chunks else [text[:2000]]

    # Fallback
    return [text[i:i+1500] for i in range(0, len(text), 1500)]


# ---------------------------
# Count tokens
# ---------------------------
def count_tokens(text: str) -> int:
    if HAVE_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text)//3)


# ---------------------------
# Translate chunk using Gemini (เวอร์ชันใหม่)
# ---------------------------
def translate_chunk_using_glossary(chunk_text: str, full_glossary: Dict[str, str], automaton=None) -> Tuple[str, str]:
    if AHO_AVAILABLE and automaton:
        matched_terms = find_all_matched_terms(chunk_text, automaton)
    else:
        matched_terms = {k for k in full_glossary if k in chunk_text}

    relevant_glossary = {k: full_glossary[k] for k in matched_terms}
    glossary_lines = "\n".join(f"{k} | {v}" for k, v in relevant_glossary.items()) if relevant_glossary else "(ไม่มีคำเฉพาะ)"

    prompt = (
        "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายแฟนตาซีจีนโบราณเป็นภาษาไทย "
        "ใช้ภาษาไทยสมัยใหม่ เป็นธรรมชาติ เหมาะกับนิยายทั่วไป "
        "แปลข้อความต่อไปนี้จากจีนเป็นไทยอย่างลื่นไหล ไม่ต้องอธิบายเพิ่ม "
        "ให้ใช้คำแปลตาม glossary อย่างเคร่งครัด\n\n"
        f"[Glossary]\n{glossary_lines}\n\n"
        f"[ต้นฉบับ]\n{chunk_text}\n\n"
        "แปลเป็น:"
    )

    try:
        translation, used_model = translate_with_model_fallback(
            prompt=prompt,
            preferred_models=config.available_models,
            max_tokens=2000,
            temperature=0.2
        )
        return translation, used_model

    except Exception as e:
        logger.error("แปล chunk ล้มเหลว", error=str(e), chunk_preview=chunk_text[:100])
        raise


# ---------------------------
# ฟังก์ชัน: ดึง commit hash ปัจจุบันของ glossary
# ---------------------------
def get_current_glossary_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%H", "--", config.glossary_file],
            capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------
# Save new terms file
# ---------------------------
def save_new_terms_file(chapter_name: str, new_terms: Dict[str,str]):
    fn = os.path.join(config.log_dir, "new_terms", f"new_terms_{chapter_name}_{now_ts()}.txt")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w", encoding="utf-8") as f:
        for k,v in new_terms.items():
            f.write(f"{k} | {v}\n")
    logger.info("Saved new terms", file=fn, count=len(new_terms))


# ---------------------------
# Main function
# ---------------------------
def process_chapter_file(chapter_path: str):
    chapter_name = os.path.splitext(os.path.basename(chapter_path))[0]
    chapter_num = int(re.search(r"(\d+)", chapter_name).group(1)) if re.search(r"(\d+)", chapter_name) else None

    logger.info("Processing chapter", chapter=chapter_name, chapter_num=chapter_num)

    dst_src = os.path.join(config.src_dir, os.path.basename(chapter_path))
    if not os.path.exists(dst_src):
        logger.error("Source file not found", path=dst_src)
        raise FileNotFoundError(f"Not found: {dst_src}")

    with open(dst_src, "r", encoding="utf-8") as f:
        src_text = f.read().strip()

    t0 = time.time()
    glossary = load_glossary()
    automaton = build_automaton_from_dict(glossary) if AHO_AVAILABLE else None

    if automaton:
        matched_keys = find_all_matched_terms(src_text, automaton)
        matches = {k: glossary[k] for k in matched_keys}
    else:
        matches = {k: v for k, v in glossary.items() if k in src_text}

    logger.info("Found existing terms", count=len(matches))

    candidates = extract_candidates(src_text, glossary)
    logger.info("Extracted candidates", count=len(candidates))

    new_translations = propose_candidate_translations_json(candidates)
    logger.info("New translations", count=len(new_translations))

    per_chapter_glossary = {**matches, **new_translations}

    if new_translations:
        save_new_terms_file(chapter_name, new_translations)
        save_glossary(new_translations, chapter_num=chapter_num)

    chunks = smart_chunk_with_overlap(src_text)
    logger.info("Chunking complete", chunks=len(chunks))

    translations = []
    used_models = []
    try:
        from tqdm import tqdm
        chunks_iter = tqdm(chunks, desc=f"แปล {chapter_name}", unit="chunk")
    except ImportError:
        chunks_iter = chunks

    for i, ch in enumerate(chunks_iter):
        logger.debug("แปล chunk", index=i+1, tokens=count_tokens(ch))
        tr, model_used = translate_chunk_using_glossary(ch, full_glossary=per_chapter_glossary, automaton=automaton)
        translations.append(tr)
        used_models.append(model_used)

    def smart_join_chunks(chunks: List[str]) -> str:
        """เชื่อม chunk ให้ลื่น โดยไม่ขึ้นบรรทัดใหม่"""
        if not chunks:
            return ""
    
        result = chunks[0]
        for chunk in chunks[1:]:
            # ถ้า chunk ก่อนหน้าจบด้วยเครื่องหมายประโยค → ขึ้นบรรทัดใหม่
            if result and result[-1] in "。！？；;…":
                result += "\n" + chunk
            else:
                # ถ้าไม่ใช่ → เชื่อมด้วยเว้นวรรค
                result += " " + chunk
        return result

    full_translation = smart_join_chunks(translations)

    out_path = os.path.join(config.out_dir, f"{chapter_name}_translated.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_translation)

    # ✅ บันทึก metadata
    meta_dir = os.path.join(config.out_dir, ".meta")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{chapter_name}.json")

    glossary_commit = get_current_glossary_commit()

    log_record = {
        "timestamp": now_ts(),
        "chapter": chapter_name,
        "source": dst_src,
        "output": out_path,
        "glossary_matches": len(matches),
        "candidates_extracted": len(candidates),
        "new_translations_added": len(new_translations),
        "chunks": len(chunks),
        "duration_sec": round(time.time() - t0, 2),
        "last_glossary_commit": glossary_commit,
        "model_used": used_models[0] if len(set(used_models)) == 1 else used_models,  # ถ้าใช้หลายโมเดล บันทึกทั้งลิสต์
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("แปลบทสำเร็จ", chapter=chapter_name, duration=log_record["duration_sec"], output=out_path)
    return out_path


# ---------------------------
# 🔹 CLI with argparse (เวอร์ชัน batch)
# ---------------------------
def get_files_from_range(start: int, end: int, prefix: str = "sa", src_dir: str = "chapters_src") -> list:
    """สร้างรายการไฟล์จากช่วงบท"""
    files = []
    for i in range(start, end + 1):
        filename = f"{prefix}_{i:04d}.txt"
        filepath = os.path.join(src_dir, filename)
        if os.path.exists(filepath):
            files.append(filepath)
        else:
            logger.warning("ไม่พบไฟล์", path=filepath)
    return files

def is_already_translated(chapter_path: str, out_dir: str) -> bool:
    """ตรวจสอบว่าแปลแล้วหรือยัง"""
    chapter_name = os.path.splitext(os.path.basename(chapter_path))[0]
    output_file = os.path.join(out_dir, f"{chapter_name}_translated.txt")
    return os.path.exists(output_file)

def main():
    parser = argparse.ArgumentParser(
        description="ระบบแปลนิยายจีนเป็นไทยอัจฉริยะ - รองรับ batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # แปลบทเดียว
  python inovel-trans.py chapters_src/sa_0001.txt

  # แปลหลายบท
  python inovel-trans.py --range 1 10
  python inovel-trans.py --range 1 50 --prefix novel

  # แปลแบบข้ามบทที่แปลแล้ว
  python inovel-trans.py --range 1 100 --skip-existing

  # ระบุไฟล์เอง
  python inovel-trans.py --files chapters_src/sa_0001.txt chapters_src/sa_0005.txt
        """
    )
    # โหมดเดียว (ยังรองรับ)
    parser.add_argument("chapter", nargs="?", help="ไฟล์บทเดียว (เช่น chapters_src/sa_0001.txt)")
    
    # โหมด batch
    parser.add_argument("--files", nargs="+", help="ระบุหลายไฟล์เอง")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="ช่วงบท เช่น --range 1 100")
    parser.add_argument("--prefix", default="sa", help="prefix ชื่อไฟล์ (default: sa)")
    
    # ตัวเลือกเพิ่มเติม
    parser.add_argument("--skip-existing", action="store_true", help="ข้ามบทที่แปลแล้ว")
    parser.add_argument("--quiet", action="store_true", help="ปิดการแสดงผลระหว่างทำงาน (ยกเว้น error)")
    
    # config override
    parser.add_argument("--model", help="โมเดล Gemini (คั่นด้วย comma)", default=None)
    parser.add_argument("--src-dir", help="ไดเรกทอรีต้นทาง", default=None)
    parser.add_argument("--out-dir", help="ไดเรกทอรีปลายทาง", default=None)
    parser.add_argument("--log-dir", help="ไดเรกทอรี logs", default=None)

    args = parser.parse_args()

    # อัปเดต config
    if args.model:
        config.preferred_models = [m.strip() for m in args.model.split(",")]
        config.available_models = get_available_models(config.preferred_models)
        config.model = config.available_models[0]

    if args.src_dir: config.src_dir = args.src_dir
    if args.out_dir: config.out_dir = args.out_dir
    if args.log_dir: config.log_dir = args.log_dir

    # เริ่มต้น cache
    init_cache_db()
    load_cache()

    # สร้างรายการไฟล์
    if args.range:
        files = get_files_from_range(args.range[0], args.range[1], args.prefix, config.src_dir)
    elif args.files:
        files = []
        for f in args.files:
            if os.path.exists(f):
                files.append(f)
            else:
                logger.warning("ไม่พบไฟล์", path=f)
    elif args.chapter:
        if os.path.exists(args.chapter):
            files = [args.chapter]
        else:
            logger.error("ไม่พบไฟล์", path=args.chapter)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    if not files:
        logger.error("ไม่มีไฟล์ให้แปล")
        sys.exit(1)

    # กรองไฟล์ที่แปลแล้ว (ถ้าใช้ --skip-existing)
    if args.skip_existing:
        original_count = len(files)
        files = [f for f in files if not is_already_translated(f, config.out_dir)]
        skipped = original_count - len(files)
        if skipped > 0 and not args.quiet:
            logger.info("ข้ามบทที่แปลแล้ว", count=skipped)

    if not files:
        logger.info("ไม่มีบทใหม่ให้แปล")
        sys.exit(0)

    # เริ่ม batch
    total = len(files)
    success_count = 0
    error_count = 0
    error_files = []
    start_time = time.time()

    # --- เพิ่ม tqdm progress bar ---
    try:
        from tqdm import tqdm
        chapter_iter = tqdm(files, desc="แปลหลายบท", unit="บท", disable=args.quiet)
    except ImportError:
        chapter_iter = files
        if not args.quiet:
            logger.info("ไม่พบ tqdm — แสดงผลแบบธรรมดา")

    for chapter_path in chapter_iter:
        chapter_name = os.path.splitext(os.path.basename(chapter_path))[0]
        if not args.quiet and 'tqdm' not in sys.modules:
            logger.info(f"กำลังแปล: {chapter_name}")

        try:
            process_chapter_file(chapter_path)
            success_count += 1
        except Exception as e:
            error_count += 1
            error_files.append(chapter_path)
            logger.error("แปลบทล้มเหลว", chapter=chapter_name, error=str(e))

        # Rate limiting: หน่วง 1 วินาที/บท (เว้นบทสุดท้าย)
        if chapter_path != files[-1]:  # ไม่หน่วงหลังบทสุดท้าย
            time.sleep(1.0)

    # สรุปผล
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info("สรุปผลการแปล batch")
    logger.info("แปลสำเร็จ", count=success_count)
    logger.info("ล้มเหลว", count=error_count)
    if error_files:
        failed_names = [os.path.splitext(os.path.basename(f))[0] for f in error_files]
        logger.info("บทที่ล้มเหลว", chapters=failed_names)
    logger.info("ใช้เวลา", seconds=round(elapsed, 1))
    logger.info("="*50)

    # บันทึก cache ก่อนออกจากโปรแกรม
    save_cache()

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()