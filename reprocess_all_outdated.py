#!/usr/bin/env python3
"""
reprocess_all_outdated.py - ตรวจสอบ, อัปเดต, และทำความสะอาดรายงานเก่าอัตโนมัติ
เวอร์ชันที่รองรับการรายงานคำที่เปลี่ยนจริง (หลังอัปเดต reprocess.py)
รองรับกรณีที่ไม่มีการเปลี่ยนแปลง → ไม่สร้างไฟล์ _updated.txt
"""

import os
import sys
import io
import json
import subprocess
import argparse
import datetime
from pathlib import Path


# ---------------------------
# บังคับใช้ UTF-8 บน Windows
# ---------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---------------------------
# Configuration
# ---------------------------
SRC_DIR = "chapters_src"
OUT_DIR = "chapters_translated"
META_DIR = os.path.join(OUT_DIR, ".meta")
GLOSSARY_FILE = "glossary.yaml"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------
# ตรวจสอบ Git
# ---------------------------
def is_git_repo() -> bool:
    try:
        result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


# ---------------------------
# ดึง commit hash ปัจจุบัน
# ---------------------------
def get_current_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------
# ตรวจสอบการเปลี่ยนแปลง glossary
# ---------------------------
def has_glossary_changed_since(commit_ref: str) -> bool:
    if not commit_ref or commit_ref == "unknown":
        return True
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", commit_ref, "HEAD", GLOSSARY_FILE],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"[ERROR] ตรวจสอบ git diff ไม่ได้: {e}")
        return True


# ---------------------------
# โหลด metadata
# ---------------------------
def load_metadata(chapter_name: str) -> dict:
    meta_path = os.path.join(META_DIR, f"{chapter_name}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------
# ค้นหาบททั้งหมด
# ---------------------------
def find_all_chapters() -> list:
    chapters = []
    src_path = Path(SRC_DIR)
    if not src_path.exists():
        print(f"[ERROR] ไม่พบไดเรกทอรี: {SRC_DIR}")
        return []
    for file in src_path.glob("*.txt"):
        chapter_name = file.stem
        chapters.append({"name": chapter_name, "path": str(file)})
    return sorted(chapters, key=lambda x: x["name"])


# ---------------------------
# รัน reprocess สำหรับบทเดียว
# ---------------------------
def reprocess_single(chapter_path: str, dry_run: bool = False) -> tuple:
    try:
        if dry_run:
            print(f"[DRY-RUN] {sys.executable} reprocess.py {chapter_path}")
            return True, None, None, []

        result = subprocess.run(
            [sys.executable, "reprocess.py", chapter_path],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] reprocess ล้มเหลว: {result.stderr}")
            return False, None, None, []

        chapter_name = Path(chapter_path).stem
        old_path = os.path.join(OUT_DIR, f"{chapter_name}_translated.txt")
        new_path = os.path.join(OUT_DIR, f"{chapter_name}_translated_updated.txt")

        # ✅ ตรวจสอบว่าไฟล์มีอยู่จริงก่อนอ่าน
        if not os.path.exists(old_path):
            print(f"[WARNING] ไม่พบบทแปลเดิมสำหรับ {chapter_name}")
            return False, None, None, []

        # ถ้าไม่มีไฟล์ updated → ถือว่าไม่มีการเปลี่ยนแปลง
        if not os.path.exists(new_path):
            print(f"[INFO] ไม่มีการเปลี่ยนแปลงสำหรับ {chapter_name} (ไม่มีไฟล์ updated)")
            return True, old_path, None, []  # new_path = None

        changes = []
        meta_path = os.path.join(META_DIR, f"{chapter_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            changes = meta.get("glossary_changes_applied", [])

        return True, old_path, new_path, changes

    except Exception as e:
        print(f"[ERROR] รัน reprocess ไม่ได้: {e}")
        return False, None, None, []


# ---------------------------
# สร้าง HTML diff
# ---------------------------
def create_diff_html(old_text: str, new_text: str, chapter_name: str) -> str:
    from difflib import HtmlDiff
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    d = HtmlDiff()
    return d.make_file(
        old_lines, new_lines,
        fromdesc=f"{chapter_name} - เดิม",
        todesc=f"{chapter_name} - อัปเดต",
        context=True,
        numlines=3
    )


# ---------------------------
# บันทึก diff report
# ---------------------------
def save_diff_report(report_data: list, output_path: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <title>รายงานการเปลี่ยนแปลงบทแปล - {timestamp}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; background: #f9f9f9; }}
    h1, h2 {{ color: #2c3e50; }}
    .summary {{ background: #e8f4fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    th, td {{ padding: 12px; text-align: left; vertical-align: top; }}
    th {{ background-color: #3498db; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    ins {{ background-color: #d4edda; text-decoration: none; padding: 2px 4px; border-radius: 3px; }}
    del {{ background-color: #f8d7da; color: #721c24; text-decoration: line-through; padding: 2px 4px; border-radius: 3px; }}
    .reason {{ font-size: 0.9em; color: #666; font-style: italic; margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>📝 รายงานการเปลี่ยนแปลงบทแปล</h1>
  <div class="summary">
    <p><strong>เวลา:</strong> {timestamp}</p>
    <p>พบ <strong>{len(report_data)} บท</strong> ที่มีการอัปเดตเนื่องจากการเปลี่ยนแปลงใน <code>glossary.yaml</code></p>
  </div>
"""

    for item in report_data:
        full_html += f"<h2>📖 บทที่: {item['chapter']}</h2>"
        if item["changes"]:
            full_html += f"<div class='reason'>คำศัพท์ที่เปลี่ยน: {', '.join(item['changes'])}</div>"
        full_html += item["diff_html"]

    full_html += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[REPORT] 📄 สร้างรายงาน: {output_path}")


# ---------------------------
# ✅ Auto Cleanup: ลบรายงานเก่าเกิน N วัน
# ---------------------------
def cleanup_old_reports(keep_days: int = 30):
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=keep_days)
    deleted_count = 0

    if not os.path.exists(REPORTS_DIR):
        return

    for file in os.listdir(REPORTS_DIR):
        if file.startswith("diff_report_") and file.endswith(".html"):
            path = os.path.join(REPORTS_DIR, file)
            try:
                timestamp_str = file[13:27]  # YYYYMMDD_HHMMSS
                file_time = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if file_time < cutoff:
                    os.remove(path)
                    print(f"[CLEANUP] 🗑️ ลบรายงานเก่า: {file}")
                    deleted_count += 1
            except Exception as e:
                print(f"[WARNING] ไม่สามารถประมวลผลไฟล์: {file}, ข้ามไป → {e}")
                continue

    if deleted_count > 0:
        print(f"[CLEANUP] ✅ ลบไฟล์เก่าจำนวน {deleted_count} ไฟล์ เรียบร้อยแล้ว")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="🔄 ตรวจสอบและอัปเดตทุกบทที่ล้าสมัยจาก glossary")
    parser.add_argument("--dry-run", action="store_true", help="แสดงผลลัพธ์โดยไม่รันจริง")
    parser.add_argument("--force", action="store_true", help="รัน reprocess ทุกบท ไม่เช็คเงื่อนไข")
    args = parser.parse_args()

    if not is_git_repo():
        print("[ERROR] ไม่พบรีพอ git กรุณาอยู่ในโฟลเดอร์โปรเจกต์ที่ใช้ git")
        exit(1)

    current_commit = get_current_commit()
    print(f"[INFO] 🔍 กำลังทำงานที่ commit: {current_commit[:8]}")

    chapters = find_all_chapters()
    if not chapters:
        print("[INFO] ไม่พบบทแปล")
        return

    outdated = []
    updated = []
    errors = {}
    diff_report_data = []

    print(f"[INFO] พบ {len(chapters)} บท กำลังตรวจสอบ...")

    for ch in chapters:
        chapter_name = ch["name"]
        old_trans_path = os.path.join(OUT_DIR, f"{chapter_name}_translated.txt")
        
        # ✅ ข้ามบทที่ยังไม่ได้แปล
        if not os.path.exists(old_trans_path):
            print(f"[SKIP] ยังไม่ได้แปล: {chapter_name}")
            continue

        meta = load_metadata(chapter_name)
        last_commit = meta.get("last_glossary_commit")
        should_update = args.force or not last_commit or has_glossary_changed_since(last_commit)

        if should_update:
            outdated.append(chapter_name)
            print(f"[🔄 OUTDATED] {chapter_name}")

            if not args.dry_run:
                success, old_path, new_path, changes = reprocess_single(ch["path"])
                if success and new_path:  # ✅ มีการเปลี่ยนแปลงจริง
                    with open(old_path, "r", encoding="utf-8") as f:
                        old_text = f.read()
                    with open(new_path, "r", encoding="utf-8") as f:
                        new_text = f.read()
                    diff_html = create_diff_html(old_text, new_text, chapter_name)
                    diff_report_data.append({
                        "chapter": chapter_name,
                        "diff_html": diff_html,
                        "changes": changes
                    })
                    updated.append(chapter_name)
                elif success:
                    # ไม่มีการเปลี่ยนแปลง → นับเป็นสำเร็จแต่ไม่สร้าง diff
                    print(f"[✅ NO CHANGE] {chapter_name}")
                    updated.append(chapter_name)
                else:
                    errors[chapter_name] = "reprocess failed or files missing"
        else:
            print(f"[✅ UP-TO-DATE] {chapter_name}")

    if diff_report_data and not args.dry_run:
        report_path = f"reports/diff_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        save_diff_report(diff_report_data, report_path)
        cleanup_old_reports(keep_days=30)

    print("\n" + "="*60)
    print("📋 สรุปการอัปเดตบทแปล")
    print("="*60)
    print(f"🔍 ตรวจสอบทั้งหมด: {len(chapters)} บท")
    print(f"🔄 ต้องอัปเดต: {len(outdated)} บท")
    print(f"✅ อัปเดตสำเร็จ: {len(updated)} บท")
    print(f"❌ ล้มเหลว: {len(errors)} บท")
    if diff_report_data:
        print(f"📊 สร้างรายงาน: {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()