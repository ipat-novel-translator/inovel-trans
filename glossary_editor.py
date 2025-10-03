#!/usr/bin/env python3
"""
glossary_editor.py - GUI สำหรับแก้ไข glossary.yaml + กรองตามวัน/บท
เวอร์ชันที่รองรับ Flet 0.28.3+ และแก้ไขปัญหา source_type ถูกเปลี่ยนทั้งหมด
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
                    simple[k] = v.get("th", k)
                    added_at = v.get("added_at", "")
                    if not added_at:
                        added_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    original_glossary_metadata[k] = {
                        "added_at": added_at,
                        "chapter_first_seen": v.get("chapter_first_seen"),
                        "source_type": v.get("source_type", "manual"),
                        "notes": v.get("notes", "")
                    }
                elif isinstance(v, str):
                    simple[k] = v
                    original_glossary_metadata[k] = {
                        "added_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "chapter_first_seen": None,
                        "source_type": "manual",
                        "notes": ""
                    }
            return simple
        except Exception as e:
            print(f"Error loading glossary: {e}")
            return {}


# ---------------------------
# บันทึก glossary.yaml (เวอร์ชันปลอดภัย)
# ---------------------------
def save_glossary(new_data: dict): 
    """
    บันทึกเฉพาะคำที่อยู่ใน new_data
    โดยรักษาคำอื่นจากไฟล์เดิม
    """
    # โหลดไฟล์เดิม
    original_data = {}
    if os.path.exists(GLOSSARY_FILE):
        with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
            original_data = yaml.safe_load(f) or {}
    
    # เริ่มจากข้อมูลเดิม
    full_data = original_data.copy()
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # อัปเดตหรือเพิ่มคำใหม่
    for hanzi, thai in new_data.items():
        if hanzi in full_data and isinstance(full_data[hanzi], dict):
            # อัปเดต dict เดิม
            full_data[hanzi]["th"] = thai
            full_data[hanzi]["source_type"] = "manual"
        else:
            # สร้างใหม่
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

    page.edit_container = ft.Column(spacing=5)
    entries = []
    visible_rows = []

    original_glossary = load_glossary()

    search_field = ft.TextField(
        label="ค้นหาคำศัพท์",
        width=250,
        dense=True,
        on_change=lambda e: filter_entries(),
    )

    # --- วันที่: จาก ... ถึง ... ---
    start_date_picker = ft.DatePicker(on_change=lambda e: update_date_filters())
    end_date_picker = ft.DatePicker(on_change=lambda e: update_date_filters())
    page.overlay.extend([start_date_picker, end_date_picker])

    start_date_field = ft.TextField(
        label="วันที่เริ่ม",
        width=120,
        dense=True,
        read_only=True,
        on_click=lambda e: page.open(start_date_picker),
        hint_text="คลิกเลือก"
    )
    end_date_field = ft.TextField(
        label="วันที่สิ้นสุด",
        width=120,
        dense=True,
        read_only=True,
        on_click=lambda e: page.open(end_date_picker),
        hint_text="คลิกเลือก"
    )

    def update_date_filters():
        if start_date_picker.value:
            start_date_field.value = start_date_picker.value.strftime("%Y-%m-%d")
        else:
            start_date_field.value = ""
        if end_date_picker.value:
            end_date_field.value = end_date_picker.value.strftime("%Y-%m-%d")
        else:
            end_date_field.value = ""
        filter_entries()

    # --- บท: จาก ... ถึง ... ---
    chapter_from = ft.TextField(
        label="บทเริ่ม",
        width=80,
        dense=True,
        text_align="center",
        on_change=lambda e: filter_entries()
    )
    chapter_to = ft.TextField(
        label="บทสิ้นสุด",
        width=80,
        dense=True,
        text_align="center",
        on_change=lambda e: filter_entries()
    )

    add_button = ft.ElevatedButton(
        "➕ เพิ่มคำใหม่",
        on_click=lambda e: add_entry(),
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.BLUE)
    )

    filter_row = ft.Row([
        search_field,
        ft.Container(width=10),
        start_date_field,
        ft.Text("ถึง", size=13, color=ft.Colors.GREY),
        end_date_field,
        ft.Container(width=10),
        chapter_from,
        ft.Text("ถึง", size=13, color=ft.Colors.GREY),
        chapter_to,
        ft.Container(width=10),
        add_button
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

    def filter_entries():
        query = search_field.value.lower() if search_field.value else ""

        # แปลงวันที่เริ่ม-สิ้นสุด เป็นรูปแบบ "YYYYMMDD"
        start_str = start_date_picker.value.strftime("%Y%m%d") if start_date_picker.value else None
        end_str = end_date_picker.value.strftime("%Y%m%d") if end_date_picker.value else None

        # แปลงบทเริ่ม-สิ้นสุด เป็น int
        try:
            chap_from = int(chapter_from.value) if chapter_from.value.strip().isdigit() else None
        except:
            chap_from = None
        try:
            chap_to = int(chapter_to.value) if chapter_to.value.strip().isdigit() else None
        except:
            chap_to = None

        page.edit_container.controls.clear()
        for row in entries:
            hanzi = row.data["hanzi_field"].value.strip()
            thai = row.data["thai_field"].value.strip()
            meta = original_glossary_metadata.get(hanzi, {})
            added_at = meta.get("added_at", "")  # เช่น "20250405_123456"
            chapter_seen = meta.get("chapter_first_seen")  # อาจเป็น int หรือ None

            # กรองตามคำค้น
            matches_search = not query or query in hanzi.lower() or query in thai.lower()

            # กรองตามช่วงวันที่
            matches_date = True
            if added_at and len(added_at) >= 8:
                entry_date = added_at[:8]  # "20250405"
                if start_str and entry_date < start_str:
                    matches_date = False
                if end_str and entry_date > end_str:
                    matches_date = False
            else:
                if start_str or end_str:
                    matches_date = False

            # กรองตามช่วงบท
            matches_chapter = True
            if chapter_seen is not None:
                if chap_from is not None and chapter_seen < chap_from:
                    matches_chapter = False
                if chap_to is not None and chapter_seen > chap_to:
                    matches_chapter = False
            else:
                if chap_from is not None or chap_to is not None:
                    matches_chapter = False

            if matches_search and matches_date and matches_chapter:
                page.edit_container.controls.append(row)

        page.update()

    def add_entry(hanzi="", thai=""):
        hanzi_field = ft.TextField(
            value=hanzi,
            expand=2,
            dense=True,
            height=40,
            text_size=16,
            bgcolor=ft.Colors.GREY_100
        )
        thai_field = ft.TextField(
            value=thai,
            expand=5,
            dense=True,
            height=40,
            text_size=16,
            bgcolor=ft.Colors.WHITE
        )
        delete_btn = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_size=18,  # เล็กลง
            on_click=lambda e: delete_entry(row),
            tooltip="ลบ",
            icon_color=ft.Colors.RED_400,
            padding=4  # ลด padding
        )

        # ใช้ Row แทน ResponsiveRow เพื่อควบคุม layout ได้ดีขึ้น
        row = ft.Row(
            controls=[
                hanzi_field,
                ft.Container(width=10),
                thai_field,
                ft.Container(width=5),
                delete_btn
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0
        )
        # เก็บ reference ไว้ใน row.data เพื่อเข้าถึงค่าได้ง่าย
        row.data = {"hanzi_field": hanzi_field, "thai_field": thai_field}

        entries.append(row)
        # แสดงทันทีถ้ายังไม่มีการกรอง
        if not search_field.value and not start_date_picker.value and not end_date_picker.value and not chapter_from.value and not chapter_to.value:
            page.edit_container.controls.append(row)
        page.update()

    def delete_entry(row):
        entries.remove(row)
        if row in page.edit_container.controls:
            page.edit_container.controls.remove(row)
        page.update()

    for k, v in original_glossary.items():
        add_entry(k, v)

    status_label = ft.Text("", size=14, color=ft.Colors.GREY)

    def save_clicked(e):
        new_data = {}
        errors = []
        for row in entries:
            hanzi = row.data["hanzi_field"].value.strip()
            thai = row.data["thai_field"].value.strip()
            if hanzi:
                original_val = original_glossary.get(hanzi)
                if original_val != thai:
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


if __name__ == "__main__":
    ft.app(target=main)