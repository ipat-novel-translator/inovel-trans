#!/usr/bin/env python3
"""
reprocess.py - อัปเดตบทแปลเดิมโดยใช้ Selective Retranslation + Git Versioning
ตรวจสอบว่า glossary เปลี่ยนไปตั้งแต่ครั้งที่บทนี้ถูกแปลไหม
"""

import os
import re
import json
import subprocess
import datetime
from typing import Dict, List, Set
import argparse


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
    """ตรวจสอบว่า glossary.yaml เปลี่ยนตั้งแต่ commit ที่ระบุหรือไม่"""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", commit_ref, "HEAD", GLOSSARY_FILE],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        # ถ้าไม่มี git หรือ error → ถือว่าอาจเปลี่ยน (safe mode)
        print("[WARNING] ไม่สามารถตรวจสอบ git diff ได้ → ถือว่ามีการเปลี่ยนแปลง")
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

    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return {
            k: v["th"] if isinstance(v, dict) and "th" in v else v
            for k, v in data.items()
            if isinstance(v, (dict, str))
        }


def load_glossary_at_commit(commit: str) -> Dict[str, str]:
    """โหลด glossary.yaml จาก commit เก่าผ่าน git show"""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{GLOSSARY_FILE}"],
            capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode != 0:
            print(f"[ERROR] ไม่สามารถโหลด glossary จาก commit {commit}")
            return {}

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
# แยกเป็นประโยค
# ---------------------------
def split_to_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？;；!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------
# จำลองการแปลใหม่ (ควรเรียก GPT ในเวอร์ชันเต็ม)
# ---------------------------
def translate_sentence_with_glossary(sentence: str, glossary: Dict[str, str]) -> str:
    result = sentence
    sorted_items = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)
    for hanzi, thai in sorted_items:
        result = result.replace(hanzi, thai)
    return result


# ---------------------------
# โหลดต้นฉบับและแปลเดิม
# ---------------------------
def load_chinese_source(chapter_path: str) -> str:
    if not os.path.exists(chapter_path):
        raise FileNotFoundError(f"ไม่พบต้นฉบับ: {chapter_path}")
    with open(chapter_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_translation(chapter_name: str) -> str:
    path = os.path.join(OUT_DIR, f"{chapter_name}_translated.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบบทแปลเดิม: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------
# Reprocess บทแปลเดิม
# ---------------------------
def reprocess_chapter(chapter_path: str):
    chapter_name = os.path.splitext(os.path.basename(chapter_path))[0]
    print(f"[INFO] 🔍 เริ่ม reprocess บท: {chapter_name}")

    # 1. โหลด metadata
    meta = load_chapter_metadata(chapter_name)
    last_processed_commit = meta.get("last_glossary_commit")

    # 2. ตรวจสอบว่า git repository มีอยู่ไหม
    if not os.path.exists(".git"):
        print("[ERROR] ไม่พบรีพอ git กรุณาใช้ 'git init' ก่อน")
        return

    # 3. ตรวจสอบว่า glossary เปลี่ยนตั้งแต่ครั้งที่แล้วหรือไม่
    if last_processed_commit:
        if not has_glossary_changed_since(last_processed_commit):
            print("[INFO] 🟢 ไม่มีการเปลี่ยนแปลงใน glossary → ข้ามไป")
            return
        print(f"[INFO] 🔄 มีการเปลี่ยนแปลง glossary ตั้งแต่ commit {last_processed_commit[:8]}")
    else:
        print("[INFO] ⚠️ ไม่พบประวัติ → ถือว่าต้อง reprocess")

    # 4. โหลด glossary เก่าและใหม่
    current_glossary = load_current_glossary()
    old_glossary = load_glossary_at_commit(last_processed_commit) if last_processed_commit else {}

    if not old_glossary:
        print("[INFO] ไม่สามารถโหลด glossary เก่า → ใช้เฉพาะคำที่เพิ่มใหม่")
        old_glossary = {}

    # 5. หาคำที่เปลี่ยนแปลง
    changed_terms = find_changed_terms(old_glossary, current_glossary)
    if not changed_terms:
        print("[INFO] 🟡 ไม่พบการเปลี่ยนแปลงศัพท์ → อัปเดต metadata เท่านั้น")
        meta["last_glossary_commit"] = get_current_commit_hash()
        meta["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_chapter_metadata(chapter_name, meta)
        return

    print(f"[CHANGE] 🔧 พบ {len(changed_terms)} คำที่เปลี่ยน: {sorted(changed_terms)}")

    # 6. โหลดข้อมูล
    try:
        src_text = load_chinese_source(chapter_path)
        old_thai = load_translation(chapter_name)
    except Exception as e:
        print(f"[ERROR] โหลดข้อมูลไม่ได้: {e}")
        return

    # 7. แยกเป็นประโยค
    chinese_sentences = split_to_sentences(src_text)
    thai_sentences = split_to_sentences(old_thai)

    # 8. แปลใหม่เฉพาะประโยคที่มีคำเปลี่ยน
    new_thai_parts = []
    updated_count = 0

    for i, zh_sent in enumerate(chinese_sentences):
        if any(term in zh_sent for term in changed_terms):
            new_sent = translate_sentence_with_glossary(zh_sent, current_glossary)
            new_thai_parts.append(new_sent)
            updated_count += 1
            print(f"  [UPDATE] ประโยค {i+1}: {zh_sent[:30]}... → {new_sent[:40]}...")
        else:
            new_thai_parts.append(thai_sentences[i] if i < len(thai_sentences) else "")

    # 9. บันทึกบทใหม่
    output_path = os.path.join(OUT_DIR, f"{chapter_name}_translated_updated.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_thai_parts))

    # 10. อัปเดต metadata
    meta["last_glossary_commit"] = get_current_commit_hash()
    meta["glossary_changes_applied"] = sorted(list(changed_terms))
    meta["sentences_updated"] = updated_count
    meta["updated_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_chapter_metadata(chapter_name, meta)

    # 11. แสดงผล
    print(f"[SUCCESS] ✅ อัปเดตเสร็จสิ้น: {output_path}")
    print(f"  - แก้ไข {updated_count} ประโยค")
    print(f"  - บันทึก metadata ที่: {os.path.join(META_DIR, f'{chapter_name}.json')}")


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
    import yaml  # เพิ่มหลัง import อื่น ๆ เพื่อเลี่ยง global scope

    parser = argparse.ArgumentParser(description="🔄 อัปเดตบทแปลเดิมตามการเปลี่ยนแปลง glossary (ใช้ Git)")
    parser.add_argument("chapter", help="พาธไฟล์ต้นฉบับ เช่น chapters_src/sa_0001.txt")

    args = parser.parse_args()
    reprocess_chapter(args.chapter)