#!/usr/bin/env python3
"""
reprocess.py - อัปเดตบทแปลเดิมโดยใช้ Selective Retranslation + Git Versioning
เวอร์ชันนี้:
- สร้างไฟล์ _translated_updated.txt เฉพาะเมื่อมีการเปลี่ยนคำจริง
- ไม่เขียนทับไฟล์ _translated.txt
- รองรับ Windows โดยบังคับใช้ UTF-8
"""

import os
import sys
import io
import json
import subprocess
import datetime
from typing import Dict, Set
import argparse

# ---------------------------
# บังคับใช้ UTF-8 บน Windows (ป้องกัน UnicodeEncodeError)
# ---------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---------------------------
# โหลดต้นฉบับภาษาจีน
# ---------------------------
def load_chinese_source(chapter_path: str) -> str:
    if not os.path.exists(chapter_path):
        raise FileNotFoundError(f"ไม่พบต้นฉบับ: {chapter_path}")
    with open(chapter_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------------------
# Configuration
# ---------------------------
SRC_DIR = "chapters_src"
OUT_DIR = "chapters_translated"
LOG_DIR = "logs"
GLOSSARY_FILE = "glossary.yaml"
META_DIR = os.path.join(OUT_DIR, ".meta")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


# ---------------------------
# ฟังก์ชัน: ตรวจสอบว่า git มีการเปลี่ยนแปลง glossary หรือไม่
# ---------------------------
def has_glossary_changed_since(commit_ref: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", commit_ref, "HEAD", GLOSSARY_FILE],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        print("[WARNING] ไม่สามารถตรวจสอบ git diff ได้ -> ถือว่ามีการเปลี่ยนแปลง")
        return True


# ---------------------------
# โหลด metadata ของบท
# ---------------------------
def load_chapter_metadata(chapter_name: str) -> dict:
    meta_path = os.path.join(META_DIR, f"{chapter_name}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_chapter_metadata(chapter_name: str, data: dict):
    meta_path = os.path.join(META_DIR, f"{chapter_name}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------
# โหลด glossary จากปัจจุบัน และจาก commit เดิม
# ---------------------------
def load_current_glossary() -> Dict[str, str]:
    if not os.path.exists(GLOSSARY_FILE):
        raise FileNotFoundError(f"ไม่พบ {GLOSSARY_FILE}")

    import yaml
    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return {
            k: v["th"] if isinstance(v, dict) and "th" in v else v
            for k, v in data.items()
            if isinstance(v, (dict, str))
        }


def load_glossary_at_commit(commit: str) -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{GLOSSARY_FILE}"],
            capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode != 0:
            print(f"[ERROR] ไม่สามารถโหลด glossary จาก commit {commit}")
            return {}

        import yaml
        from io import StringIO
        data = yaml.safe_load(StringIO(result.stdout)) or {}
        return {
            k: v["th"] if isinstance(v, dict) and "th" in v else v
            for k, v in data.items()
            if isinstance(v, (dict, str))
        }
    except Exception as e:
        print(f"[ERROR] โหลด glossary จาก git ล้มเหลว: {e}")
        return {}


# ---------------------------
# หาคำที่เปลี่ยนแปลงระหว่างสองเวอร์ชัน
# ---------------------------
def find_changed_terms(old: Dict[str, str], new: Dict[str, str]) -> Set[str]:
    changed = set()
    all_keys = set(old.keys()) | set(new.keys())
    for k in all_keys:
        if old.get(k) != new.get(k):
            changed.add(k)
    return changed


# ---------------------------
# แทนที่คำในข้อความไทยทั้งหมด
# ---------------------------
def replace_glossary_terms_in_text(thai_text: str, old_glossary: Dict[str, str], new_glossary: Dict[str, str]) -> str:
    result = thai_text
    replacements = []
    for hanzi, new_thai in new_glossary.items():
        old_thai = old_glossary.get(hanzi)
        if old_thai and old_thai != new_thai:
            replacements.append((old_thai, new_thai))
    
    # เรียงจากยาวไปสั้น เพื่อหลีกเลี่ยงการแทนที่ซ้อน
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for old_word, new_word in replacements:
        if old_word in result:
            result = result.replace(old_word, new_word)
    return result


# ---------------------------
# Load_translation
# ---------------------------
def load_translation(chapter_name: str) -> str:
    path = os.path.join(OUT_DIR, f"{chapter_name}_translated.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบบทแปล: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------
# Reprocess บทแปลเดิม
# ---------------------------
def reprocess_chapter(chapter_path: str):
    chapter_name = os.path.splitext(os.path.basename(chapter_path))[0]
    print(f"[INFO] Starting reprocess for chapter: {chapter_name}")

    meta = load_chapter_metadata(chapter_name)
    last_processed_commit = meta.get("last_glossary_commit")

    if not os.path.exists(".git"):
        print("[ERROR] Git repo not found. Please run 'git init'")
        return

    if last_processed_commit:
        if not has_glossary_changed_since(last_processed_commit):
            print("[INFO] No glossary changes -> skipping")
            # ✅ ไม่สร้างไฟล์ updated
            meta["last_glossary_commit"] = get_current_commit_hash()
            meta["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_chapter_metadata(chapter_name, meta)
            return
        print(f"[INFO] Glossary changed since commit {last_processed_commit[:8]}")
    else:
        print("[INFO] No history found -> full reprocess")

    current_glossary = load_current_glossary()
    old_glossary = load_glossary_at_commit(last_processed_commit) if last_processed_commit else {}

    if not old_glossary:
        print("[INFO] Cannot load old glossary -> using current only")
        old_glossary = {}

    changed_terms = find_changed_terms(old_glossary, current_glossary)
    if not changed_terms:
        print("[INFO] No term changes detected")
        # ✅ ไม่สร้างไฟล์ _updated.txt
        meta["last_glossary_commit"] = get_current_commit_hash()
        meta["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_chapter_metadata(chapter_name, meta)
        return

    print(f"[CHANGE] Found {len(changed_terms)} changed terms")

    try:
        old_thai = load_translation(chapter_name)
    except Exception as e:
        print(f"[ERROR] Failed to load translation: {e}")
        return

    updated_thai = replace_glossary_terms_in_text(old_thai, old_glossary, current_glossary)

    output_path = os.path.join(OUT_DIR, f"{chapter_name}_translated_updated.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated_thai)

    meta["last_glossary_commit"] = get_current_commit_hash()
    meta["glossary_changes_applied"] = sorted(list(changed_terms))
    meta["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_chapter_metadata(chapter_name, meta)

    print(f"[SUCCESS] Updated file created: {output_path}")


# ---------------------------
# ฟังก์ชันช่วย: ดึง commit hash ปัจจุบัน
# ---------------------------
def get_current_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Update translation based on glossary changes")
    parser.add_argument("chapter", help="Path to source chapter file")

    args = parser.parse_args()
    reprocess_chapter(args.chapter)