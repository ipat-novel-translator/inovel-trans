#!/usr/bin/env python3
"""
glossary_editor.py - GUI สำหรับแก้ไข glossary.yaml + กรองตามวัน/บท
เวอร์ชันที่รองรับ Flet 0.28.3+
"""

import flet as ft
import yaml
import os
import subprocess
import datetime


# ---------------------------
# Configuration
# ---------------------------
GLOSSARY_FILE = "glossary.yaml"
APP_TITLE = "📖 แก้ไขพจนานุกรมนิยายไซ่เซียน"


# ---------------------------
# อ่าน glossary.yaml + เก็บ metadata
# ---------------------------
original_glossary_metadata = {}

def load_glossary():
    global original_glossary_metadata
    if not os.path.exists(GLOSSARY_FILE):
        return {}
    
    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
            simple = {}
            original_glossary_metadata.clear()
            for k, v in data.items():
                if isinstance(v, dict):
                    simple[k] = v["th"]
                    original_glossary_metadata[k] = {
                        "added_at": v.get("added_at", ""),
                        "chapter_first_seen": v.get("chapter_first_seen"),
                        "source_type": v.get("source_type")
                    }
                elif isinstance(v, str):
                    simple[k] = v
                    original_glossary_metadata[k] = {
                        "added_at": "",
                        "chapter_first_seen": None,
                        "source_type": "auto"
                    }
            return simple
        except Exception as e:
            print(f"Error loading glossary: {e}")
            return {}


# ---------------------------
# บันทึก glossary.yaml (โครงสร้างเต็ม)
# ---------------------------
def save_glossary( dict):
    full_data = {}
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for hanzi, thai in data.items():
        if hanzi in full_data:
            if isinstance(full_data[hanzi], dict):
                full_data[hanzi]["th"] = thai
                full_data[hanzi]["added_at"] = now
            else:
                full_data[hanzi] = {
                    "th": thai,
                    "source_type": "manual",
                    "added_at": now,
                    "notes": ""
                }
        else:
            full_data[hanzi] = {
                "th": thai,
                "source_type": "manual",
                "added_at": now,
                "notes": ""
            }

    with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
        yaml.dump(full_data, f, allow_unicode=True, sort_keys=False, indent=2)


# ---------------------------
# Git Commit อัตโนมัติ
# ---------------------------
def git_commit_changes():
    try:
        result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], 
                                capture_output=True, text=True)
        if result.returncode != 0:
            return False, "ไม่พบ Git repository"

        subprocess.run(["git", "add", GLOSSARY_FILE], check=True)
        msg = f"📝 อัปเดต glossary จาก GUI: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        return True, "✅ Git commit สำเร็จ!"
    except subprocess.CalledProcessError as e:
        return False, f"❌ Git error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ---------------------------
# Main UI
# ---------------------------
def main(page: ft.Page):
    page.title = APP_TITLE
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 1200
    page.window_height = 800
    page.scroll = "auto"

    # --- ประกาศตัวแปรสำคัญก่อน ---
    page.edit_container = ft.Column(spacing=5)
    entries = []
    visible_rows = []

    # --- โหลดข้อมูล ---
    original_glossary = load_glossary()

    # --- ช่องค้นหา ---
    search_field = ft.TextField(
        label="ค้นหาคำศัพท์",
        width=250,
        dense=True,
        on_change=lambda e: filter_entries(),
    )

    # --- ตัวกรองวันที่ ---
    date_picker = ft.DatePicker(
        on_change=lambda e: update_date_label_and_filter(),
        on_dismiss=lambda e: print("ปิดตัวเลือกวัน")
    )
    page.overlay.append(date_picker)

    def update_date_label_and_filter():
        if date_picker.value:
            selected_date.value = date_picker.value.strftime("%Y-%m-%d")
        else:
            selected_date.value = "ทุกวัน"
        filter_entries()

    date_button = ft.ElevatedButton(
        "เลือกวัน",
        icon=ft.Icons.CALENDAR_TODAY,
        on_click=lambda e: page.open(date_picker),  # ✅ แก้ตรงนี้
        style=ft.ButtonStyle(padding=10)
    )
    selected_date = ft.Text("ทุกวัน", size=13, italic=True, color=ft.Colors.GREY)

    # --- ตัวกรองบทที่ ---
    chapter_filter = ft.TextField(
        label="บท",
        width=80,
        dense=True,
        text_align="center",
        on_change=lambda e: filter_entries()
    )

    # --- ปุ่มเพิ่มแถวใหม่ (ประกาศก่อนใช้!) ---
    add_button = ft.ElevatedButton(
        "➕ เพิ่มคำใหม่",
        on_click=lambda e: add_entry(),
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.BLUE)
    )

    # --- กลุ่มควบคุมตัวกรอง ---
    filter_row = ft.Row([
        search_field,
        ft.Container(width=20),
        date_button,
        selected_date,
        ft.Container(width=20),
        chapter_filter,
        ft.Container(width=20),
        add_button  # ✅ ตอนนี้ add_button ถูกประกาศแล้ว
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # --- ฟังก์ชันกรอง ---
    def filter_entries():
        query = search_field.value.lower() if search_field.value else ""
        
        # ตรวจสอบวันที่
        target_date = ""
        if hasattr(date_picker, "value") and date_picker.value:
            target_date = date_picker.value.strftime("%Y%m%d")

        # ตรวจสอบบทที่
        chapter_query = chapter_filter.value.strip()
        target_chapter = int(chapter_query) if chapter_query.isdigit() else None

        page.edit_container.controls.clear()

        for row in entries:
            hanzi = row.controls[0].value.strip()
            thai = row.controls[1].value.strip()
            
            # ดึง metadata
            entry_data = original_glossary_metadata.get(hanzi, {})

            added_at = entry_data.get("added_at", "")
            chapter_seen = entry_data.get("chapter_first_seen")

            # เงื่อนไขการกรอง
            matches_search = not query or query in hanzi.lower() or query in thai.lower()
            matches_date = not target_date or target_date in added_at
            matches_chapter = target_chapter is None or chapter_seen == target_chapter

            if matches_search and matches_date and matches_chapter:
                page.edit_container.controls.append(row)

        page.update()

    # --- ฟังก์ชัน: เพิ่มแถวใหม่ ---
    def add_entry(hanzi="", thai=""):
        row = ft.ResponsiveRow(
            columns=12,
            controls=[
                ft.TextField(
                    value=hanzi,
                    col={"sm": 4},
                    dense=True,
                    height=40,
                    text_size=16,
                    bgcolor=ft.Colors.GREY_100
                ),
                ft.TextField(
                    value=thai,
                    col={"sm": 7},
                    dense=True,
                    height=40,
                    text_size=16,
                    bgcolor=ft.Colors.WHITE
                ),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda e: delete_entry(row),
                    tooltip="ลบ",
                    icon_color=ft.Colors.RED_400
                )
            ]
        )
        entries.append(row)
        if search_field.value == "" and (not chapter_filter.value) and (not hasattr(date_picker, "value")):
            page.edit_container.controls.append(row)
        page.update()

    def delete_entry(row):
        entries.remove(row)
        page.edit_container.controls.remove(row)
        page.update()

    # --- เพิ่มแถวจากข้อมูลเดิม ---
    for k, v in original_glossary.items():
        add_entry(k, v)

    # --- ปุ่มบันทึก ---
    status_label = ft.Text("", size=14, color=ft.Colors.GREY)

    def save_clicked(e):
        new_data = {}
        errors = []
        for row in entries:
            hanzi_field = row.controls[0]
            thai_field = row.controls[1]
            hanzi = hanzi_field.value.strip()
            thai = thai_field.value.strip()
            if hanzi:
                if hanzi in new_data:
                    errors.append(f"ซ้ำ: {hanzi}")
                else:
                    new_data[hanzi] = thai
            elif thai:
                errors.append(f"มีคำไทยแต่ไม่มีคำจีน: {thai}")

        if errors:
            status_label.value = "⚠️ ข้อผิดพลาด: " + "; ".join(errors)
            status_label.color = ft.Colors.RED
            page.update()
            return

        try:
            save_glossary(new_data)
            success, msg = git_commit_changes()
            status_label.value = msg
            status_label.color = ft.Colors.GREEN if success else ft.Colors.RED
        except Exception as e:
            status_label.value = f"❌ บันทึกไม่สำเร็จ: {str(e)}"
            status_label.color = ft.Colors.RED

        page.update()

    save_button = ft.ElevatedButton(
        "💾 บันทึกและ Git Commit",
        on_click=save_clicked,
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.GREEN)
    )

    # --- จัดวางหน้าจอ ---
    page.add(
        ft.Text("แก้ไขพจนานุกรมศัพท์นิยายไซ่เซียน", size=24, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        filter_row,
        ft.Row([save_button]),
        status_label,
        ft.Divider(),
        ft.Text("คำศัพท์ (ภาษาจีน → ภาษาไทย)", size=16, weight=ft.FontWeight.BOLD),
        page.edit_container
    )


# ---------------------------
# เริ่มต้นแอป
# ---------------------------
if __name__ == "__main__":
    ft.app(target=main)