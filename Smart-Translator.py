#!/usr/bin/env python3
"""
trans1.py - แปลนิยายจีนเป็นไทย (รองรับ glossary.yaml + manual override)
ปรับปรุง: ใช้ argparse + SQLite cache + Git-based versioning + GEMINI API
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
from typing import Dict, List, Any
import structlog
import yaml
import argparse
from dotenv import load_dotenv
load_dotenv()  # โหลดค่าจาก .env เข้าสู่ environment

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
    GEMINI_AVAILABLE = False
    print("ERROR: ติดตั้ง google-generativeai ด้วยคำสั่ง: pip install -U google-generativeai")
    sys.exit(1)

# tiktoken for token counting (ใช้ประมาณการเท่านั้น — Gemini ใช้ token ต่างจาก OpenAI)
try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False


# ---------------------------
# 🔧 Configuration
# ---------------------------
class Config:
    def __init__(self):
        self.model = os.getenv("TRANSLATION_MODEL", "gemini-1.5-flash")  # เปลี่ยนเป็น gemini
        self.token_limit = int(os.getenv("TOKEN_LIMIT", "8000"))
        self.chunk_token_target = int(os.getenv("CHUNK_TOKEN_TARGET", "1800"))
        self.token_buffer = int(os.getenv("TOKEN_BUFFER", "400"))

        self.src_dir = os.getenv("SRC_DIR", "chapters_src")
        self.out_dir = os.getenv("OUT_DIR", "chapters_translated")
        self.log_dir = os.getenv("LOG_DIR", "logs")
        self.backup_dir = os.getenv("BACKUP_DIR", "backups")
        self.glossary_file = os.getenv("GLOSSARY_FILE", "glossary.yaml")
        self.cache_db = os.path.join(self.log_dir, "translation_cache.db")

        # สร้างไดเรกทอรี
        for path in [self.src_dir, self.out_dir, self.log_dir, self.backup_dir]:
            os.makedirs(path, exist_ok=True)

config = Config()


# ---------------------------
# Structured Logging
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
logger = structlog.get_logger("novel_translate")


# ---------------------------
# 🔽 Global Cache Variable
# ---------------------------
translation_cache = {}


# ---------------------------
# 🔽 SQLite Cache System
# ---------------------------
import sqlite3

def init_cache_db():
    """สร้างตาราง cache หากยังไม่มี"""
    with sqlite3.connect(config.cache_db) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                hanzi TEXT PRIMARY KEY,
                thai TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def load_cache():
    """โหลด cache จาก SQLite"""
    translation_cache.clear()
    try:
        with sqlite3.connect(config.cache_db) as conn:
            cursor = conn.execute("SELECT hanzi, thai FROM translations")
            for row in cursor.fetchall():
                translation_cache[row[0]] = row[1]
        logger.info("Cache loaded from SQLite", terms=len(translation_cache))
    except Exception as e:
        logger.warning("Failed to load cache from SQLite", error=str(e))

def save_cache():
    """บันทึก cache ลง SQLite"""
    try:
        with sqlite3.connect(config.cache_db) as conn:
            cursor = conn.cursor()
            for hanzi, thai in translation_cache.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO translations (hanzi, thai, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (hanzi, thai))
            conn.commit()
        logger.debug("Cache saved to SQLite", file=config.cache_db)
    except Exception as e:
        logger.warning("Failed to save cache to SQLite", error=str(e))


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
# Save Glossary (respect manual)
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
# Gemini Client Setup
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set", action="exit")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------
# Retry Mechanism for Gemini
# ---------------------------
def gemini_with_retry(call_func: callable, max_retries: int = 3) -> Any:
    for i in range(max_retries):
        try:
            return call_func()
        except Exception as e:
            if i == max_retries - 1:
                logger.error("Gemini request failed after retries", attempts=max_retries, error=str(e))
                raise

            err_msg = str(e).lower()
            retryable = any(keyword in err_msg for keyword in ["rate limit", "timeout", "connection", "502", "503", "quota"])
            if retryable:
                wait = (2 ** i) + random.uniform(0, 1)
                logger.warning("Retrying Gemini", attempt=i+1, wait_seconds=round(wait, 1), reason=str(e))
                time.sleep(wait)
            else:
                logger.error("Non-retryable error", error=str(e))
                raise


# ---------------------------
# Extract candidates (ใช้ Gemini)
# ---------------------------
def extract_candidates(text: str, existing_glossary: Dict[str,str]) -> List[str]:
    system_prompt = (
        "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายจีนโบราณแนวไซ่เซียนเป็นภาษาไทย "
        "ให้สกัดคำเฉพาะทางใหม่ (ชื่อคน ชื่อสำนัก วิชา ตำแหน่ง ขั้นพลัง) จากข้อความต่อไปนี้ "
        "และส่งกลับเป็น JSON array เท่านั้น เช่น [\"คำ1\", \"คำ2\"] "
        "อย่ารวมคำทั่วไป หรือคำที่อยู่ในรายการที่ให้มา"
    )

    existing_keys = set(existing_glossary.keys())
    truncated_text = text[:4000]

    user_prompt = (
        f"[รายการคำศัพท์ที่มีอยู่แล้ว - อย่าส่งคืน]\n{list(existing_keys)}\n\n"
        f"[ข้อความต้นฉบับ]\n{truncated_text}\n\n"
        "ส่งคืนเฉพาะ JSON array ของคำเฉพาะใหม่:"
    )

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        def make_call():
            model = genai.GenerativeModel(config.model)
            return model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                    response_mime_type="application/json"
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )

        resp = gemini_with_retry(make_call)
        content = resp.text.strip()

        # ลอง parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # ถ้าไม่สำเร็จ ลอง extract แบบเดิม
            data = extract_json_safely(content)

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
        logger.error("Candidate extraction failed", error=str(e))
        return []


# ---------------------------
# Extract JSON Safely (เหมือนเดิม)
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
        logger.info("All candidates are cached", count=len(candidates))
        return results

    batch_size = 40

    for i in range(0, len(unseen), batch_size):
        batch = unseen[i:i+batch_size]
        terms_list = "\n".join(batch)
        prompt = (
            "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายจีนโบราณแนวไซ่เซียนเป็นภาษาไทย "
            "ให้แปลคำศัพท์เฉพาะทางต่อไปนี้เป็นภาษาไทย ตามสไตล์นิยาย "
            "ใช้ภาษาไทยสมัยใหม่ เป็นธรรมชาติ ไม่เป็นทางการเกินไป "
            "ห้ามอธิบาย ห้ามเพิ่มเติม ให้ส่งคืนเฉพาะ JSON รูปแบบ: {\"glossary\": {\"汉字\":\"ไทย\", ...}}\n\n"
            f"คำศัพท์:\n{terms_list}"
        )

        try:
            def make_call():
                model = genai.GenerativeModel(config.model)
                return model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1500,
                        response_mime_type="application/json"
                    ),
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )

            resp = gemini_with_retry(make_call)
            content = resp.text.strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = extract_json_safely(content)

            if isinstance(data, dict) and "glossary" in data:
                new_translations = {k.strip(): v.strip() for k,v in data["glossary"].items()}
                results.update(new_translations)
                translation_cache.update(new_translations)
            else:
                logger.warning("No glossary in response", raw=content[:200])

        except Exception as e:
            logger.error("Translation proposal failed", batch=i//batch_size, error=str(e))

    logger.info("Proposed translations", new=len(results), cached=len(candidates)-len(unseen))
    return results


# ---------------------------
# Smart Chunking (เหมือนเดิม)
# ---------------------------
def smart_chunk_with_overlap(text: str, target_tokens: int = None, overlap_sentences: int = 1) -> List[str]:
    if target_tokens is None:
        target_tokens = config.chunk_token_target

    if HAVE_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")  # ใช้เป็น proxy สำหรับประมาณการ
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        sentences = re.split(r'(?<=[。！？;；!?])\s*', text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if not sent.strip():
                continue
            temp_chunk = current_chunk + sent
            token_count = len(enc.encode(temp_chunk))

            if token_count > target_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                last_sents = re.split(r'(?<=[。！？;；!?])\s*', current_chunk)[-overlap_sentences:]
                context = " ".join(ls.strip() for ls in last_sents if ls.strip())
                current_chunk = context + " " + sent
            else:
                current_chunk = temp_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    return [text[i:i+2000] for i in range(0, len(text), 2000)]


# ---------------------------
# Count tokens (ใช้ tiktoken เป็น proxy)
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
# Translate chunk using Gemini
# ---------------------------
def translate_chunk_using_glossary(chunk_text: str, full_glossary: Dict[str, str], automaton=None) -> str:
    if AHO_AVAILABLE and automaton:
        matched_terms = find_all_matched_terms(chunk_text, automaton)
    else:
        matched_terms = {k for k in full_glossary if k in chunk_text}

    relevant_glossary = {k: full_glossary[k] for k in matched_terms}
    glossary_lines = "\n".join(f"{k} | {v}" for k, v in relevant_glossary.items()) if relevant_glossary else "(ไม่มีคำเฉพาะ)"

    prompt = (
        "คุณเป็นผู้เชี่ยวชาญด้านการแปลนิยายจีนโบราณแนวไซ่เซียนเป็นภาษาไทย "
        "ใช้ภาษาไทยสมัยใหม่ เป็นธรรมชาติ เหมาะกับนิยาย "
        "แปลข้อความต่อไปนี้จากจีนเป็นไทยอย่างลื่นไหล ไม่ต้องอธิบายเพิ่ม "
        "ให้ใช้คำแปลตาม glossary อย่างเคร่งครัด\n\n"
        f"[Glossary]\n{glossary_lines}\n\n"
        f"[ต้นฉบับ]\n{chunk_text}\n\n"
        "แปลเป็น:"
    )

    try:
        def make_call():
            model = genai.GenerativeModel(config.model)
            return model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )

        resp = gemini_with_retry(make_call)
        return resp.text.strip()

    except Exception as e:
        logger.error("Chunk translation failed", error=str(e))
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
    try:
        from tqdm import tqdm
        chunks_iter = tqdm(chunks, desc=f"Translating {chapter_name}", unit="chunk")
    except ImportError:
        chunks_iter = chunks

    for i, ch in enumerate(chunks_iter):
        logger.debug("Translating chunk", index=i+1, tokens=count_tokens(ch))
        tr = translate_chunk_using_glossary(ch, full_glossary=per_chapter_glossary, automaton=automaton)
        translations.append(tr)

    full_translation = "\n".join(translations)
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
        "model": config.model  # บันทึกว่าใช้โมเดลไหน
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("Chapter processed", chapter=chapter_name, duration=log_record["duration_sec"], output=out_path)
    return out_path


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
# 🔹 CLI with argparse
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="แปลนิยายจีนเป็นไทย (รองรับ cache และ glossary) - ใช้ Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trans1.py sa_0001.txt
  python trans1.py sa_0001.txt --model gemini-1.5-pro
  python trans1.py sa_0001.txt --src-dir my_chapters
        """
    )
    parser.add_argument("chapter", help="พาธของไฟล์บทที่จะแปล (เช่น chapters_src/sa_0001.txt)")
    parser.add_argument("--model", help="โมเดล Gemini (ค่าเริ่มต้น: gemini-1.5-flash)", default=None)
    parser.add_argument("--src-dir", help="ไดเรกทอรีต้นทาง", default=None)
    parser.add_argument("--out-dir", help="ไดเรกทอรีปลายทาง", default=None)
    parser.add_argument("--log-dir", help="ไดเรกทอรี logs", default=None)

    args = parser.parse_args()

    if args.model: config.model = args.model
    if args.src_dir: config.src_dir = args.src_dir
    if args.out_dir: config.out_dir = args.out_dir
    if args.log_dir: config.log_dir = args.log_dir

    init_cache_db()
    load_cache()

    try:
        process_chapter_file(args.chapter)
    finally:
        save_cache()


if __name__ == "__main__":
    main()