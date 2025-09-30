

# 📌 **Reference: ระบบแปลนิยายจีน → ไทย (Smart Novel Translator System)**

> โปรเจกต์พัฒนาเครื่องมือแปลนิยายไซ่เซียนจากภาษาจีนเป็นภาษาไทย โดยเน้นความแม่นยำ, การควบคุมเวอร์ชัน, และการอัปเดตอัตโนมัติเมื่อมีการแก้ไขคำศัพท์

---

## 🔧 ส่วนประกอบหลักของระบบ

| ไฟล์ | หน้าที่ |
|------|--------|
| `trans1.py` | แปลบทใหม่จากไฟล์ `.txt` |
| `reprocess.py` | อัปเดตบทแปลเดิมบทเดียวที่ล้าสมัยจาก `glossary.yaml` |
| `reprocess_all_outdated.py` | ตรวจสอบและอัปเดตทุกบทที่ “ล้าสมัย” + สร้างรายงาน HTML |
| `glossary_editor.py` | GUI สำหรับแก้ไข `glossary.yaml` ด้วย Flet |
| `glossary.yaml` | พจนานุกรมศัพท์เฉพาะ (ชื่อ, สำนัก, วิชา, ตำแหน่ง) |
| `.meta/` | โฟลเดอร์เก็บ metadata เช่น เวอร์ชัน glossary ที่ใช้ในการแปลแต่ละบท |
| `reports/diff_report_*.html` | รายงานเปรียบเทียบการเปลี่ยนแปลงก่อน-หลัง reprocess |

---

## 🛠️ เทคโนโลยีที่ใช้

- **Python 3.13**
- **Flet 0.28.3** (GUI framework สำหรับสร้างแอป multi-platform)
- **Git** (จัดการเวอร์ชัน glossary)
- **SQLite** (cache คำแปล)
- **OpenAI API** (GPT-4o-mini สำหรับแปลและสกัดคำ)

---

## ⚠️ ปัญหาสำคัญที่พบ & วิธีแก้ไข

### 1. **Flet 0.28.3 มีการเปลี่ยนแปลง API หลายจุด**

| ปัญหา | วิธีแก้ไข |
|------|----------|
| ❌ `ft.colors.GREY_100` → ไม่มี | ✅ ใช้ `ft.Colors.GREY_100` (ตัวใหญ่) |
| ❌ `ft.icons.DELETE_OUTLINE` → ไม่มี | ✅ ใช้ `ft.Icons.DELETE_OUTLINE` |
| ❌ `DatePicker` ไม่มี `pick_date()` | ✅ ใช้ `page.open(date_picker)` แทน |
| ❌ `page.edit_container` ยังไม่ประกาศ | ✅ ประกาศ `page.edit_container = ft.Column()` ก่อนใช้งาน |

> 💡 **สรุป:** Flet เวอร์ชันใหม่เปลี่ยนจาก `lowercase` → `PascalCase` สำหรับ constants

---

### 2. **Syntax Error จากการ copy-paste โค้ด**

| ปัญหา | วิธีแก้ไข |
|------|----------|
| ❌ `if hanzi in current_` | ✅ ต้องเป็น `if hanzi in current_` |
| ❌ ขาด `:` หลังเงื่อนไข | ✅ ตรวจสอบให้ครบ |
| ❌ ตัดบรรทัดกลางคำ → เช่น `full_data` → `full_` | ✅ ใช้ editor ที่ดี (VS Code) + ตรวจสอบด้วย `python -m py_compile` |

---

### 3. **GUI ไม่ทำงานเนื่องจาก scope และ closure**

| ปัญหา | วิธีแก้ไข |
|------|----------|
| ❌ `UnboundLocalError: add_button` | ✅ ประกาศตัวแปรก่อนใช้ใน `filter_row` |
| ❌ `NameError: cannot access free variable 'edit_container'` | ✅ ใช้ `page.edit_container` แทนตัวแปร global |

---

## ✅ ฟีเจอร์ที่พัฒนาแล้ว

### ✅ 1. แปลบทใหม่ (`trans1.py`)
- ใช้ glossary + cache + Aho-Corasick
- แบ่ง chunk อัจฉริยะ
- บันทึก metadata (เช่น `last_glossary_commit`)

### ✅ 2. อัปเดตบทเก่าอัตโนมัติ
- ตรวจว่าบทไหน “ล้าสมัย” จากการเปลี่ยนแปลง glossary
- แปลเฉพาะประโยคที่ได้รับผลกระทบ (Selective Retranslation)

### ✅ 3. GUI แก้ไข glossary (`glossary_editor.py`)
- ค้นหาคำศัพท์แบบ real-time
- **กรองตามวันที่เพิ่ม (`added_at`)**
- **กรองตามบทที่ปรากฏครั้งแรก (`chapter_first_seen`)**
- กด “บันทึก” → auto `git add && git commit`

### ✅ 4. รายงานการเปลี่ยนแปลง
- สร้าง `diff_report_YYYYMMDD_HHMMSS.html`
- แสดงการเปลี่ยนแปลงแบบเปรียบเทียบ (ก่อน/หลัง)
- ลบไฟล์เก่าเกิน 30 วันอัตโนมัติ

---

## 📁 โครงสร้างโฟลเดอร์

```
Trans1/
├── chapters_src/
├── chapters_translated/
│   └── .meta/
├── logs/
├── reports/
├── backups/
├── glossary.yaml
├── trans1.py
├── reprocess.py
├── reprocess_all_outdated.py
└── glossary_editor.py
```

---

## 🚀 วิธีใช้งาน

```bash
# 1. แปลบทใหม่
python trans1.py chapters_src/sa_0001.txt

# 2. แก้ glossary ผ่าน GUI
python glossary_editor.py

# 3. อัปเดตบทที่ล้าสมัย
python reprocess_all_outdated.py
```

---

ฉันกำลังพัฒนาระบบแปลนิยายจีนเป็นไทย ตอนนี้ฉันต้องการ...