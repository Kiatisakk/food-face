# let him cook.

**▶ ลองเล่น: <https://kiatisakk.github.io/food-face/>** · [English](README.md)

**A very unnecessary image-to-food experiment.**

เว็บสำหรับ Stupid Hackathon ที่รับรูปภาพหนึ่งรูป แล้วพาไปผ่านครัวแปลก ๆ ถึง 4 ขั้น: จากภาพโมเสกอาหาร ไปเป็นรหัส Emoji ด้วย AI ถอดรหัสกลับมา และปิดท้ายด้วยการเอาพิกเซลไปเรียงให้เหมือนภาพอาหารจริง

> **หนึ่งรูป สี่ครัว และคำถามเดียว: เราทำไปเพื่ออะไร?**

โปรเจกต์นี้รวม Frontend, Python API, โมเดล VQ-VAE และ Food Match ไว้ใน **Git repository เดียว** ไม่ต้อง Clone โปรเจกต์อื่นมาติดตั้งแยก

## Quick start

### Requirements

- **Python 3.10+** และ `pip`
- **Node.js + npm** สำหรับคำสั่งเริ่มเว็บและทดสอบ
- macOS / Linux / Windows (คำสั่ง npm ใช้ `scripts/py.mjs` หา Python และ venv ให้ตามระบบ)
- อินเทอร์เน็ตสำหรับติดตั้ง Dependencies และโหลดข้อมูล Food Match ในครัวที่ 4

จากโฟลเดอร์หลักของ Repository:

```bash
git clone git@github.com:Kiatisakk/food-face.git
cd food-face

npm run setup:python   # ครั้งแรกเท่านั้น: สร้าง .venv และติดตั้ง EmoCodec
npm start              # รันเว็บไซต์ + FastAPI บนพอร์ต 5173
```

เข้าเว็บที่ **http://127.0.0.1:5173** แล้วกด **“ช่างมัน ให้เขาทำ”** เพื่อเข้าครัว

ถ้ามีโปรเจกต์อยู่แล้ว ให้ข้ามขั้น `git clone` และ `cd` แล้วรัน `npm start` ได้เลย หากติดตั้ง Dependencies แล้ว ไม่ต้องรัน `setup:python` ซ้ำ

**ไม่ต้อง Build Frontend** — เว็บไซต์เป็น HTML, CSS และ JavaScript ที่เสิร์ฟจาก FastAPI โดยตรง

**ไม่มี Python ก็ใช้ได้:** `npm run start:static` หรือเปิดเว็บบน GitHub Pages ครัว 2–3 จะรันโมเดลตัวเดียวกัน (ส่งออกเป็น ONNX) ในเบราว์เซอร์แทน FastAPI ได้ไฟล์ `.emotxt` / `.emoc` ตรงกับฝั่ง Python ทุกไบต์ ดูรายละเอียดใน [README ภาษาอังกฤษ](README.md#how-it-runs-browser-or-local-api)

## How it works — 4 kitchens

```text
                    Your photo
                        |
                        v
     01  FOOD KITCHEN       Image → Food Emoji Mosaic
                        |
                [send to next kitchen]
                        v
     02  EMOJI ENCODER      Image → VQ-VAE → Token IDs → Emoji
                        |
                [send recipe to decoder]
                        v
     03  EMOJI DECODER      Emoji → Token IDs → VQ-VAE → Image
                        |
                [send decoded image]
                        v
     04  FOOD MATCH         Rearrange your pixels to resemble food
                        |
                        v
                   Final dish
```

| ครัว | ทำอะไร | เทคโนโลยี |
| --- | --- | --- |
| **01 — Food Kitchen** | เปลี่ยนรูปเป็นโมเสก Emoji อาหาร ปรับความละเอียด/ความมั่ว/สี และออกใบเสร็จปลอม | Canvas, Web Worker, JavaScript |
| **02 — Emoji Encoder** | แปลงภาพเป็น Token IDs ของโมเดล แล้วแสดงเป็นข้อความ Emoji; คัดลอกหรือดาวน์โหลดได้ | PyTorch VQ-VAE, FastAPI |
| **03 — Emoji Decoder** | รับสูตร Emoji หรือไฟล์จากครัวก่อนหน้า แล้วสร้างภาพกลับด้วยโมเดลเดียวกัน | VQ-VAE Decoder, FastAPI |
| **04 — Food Match** | เทียบสีกับภาพอาหารในฐานข้อมูล 50,000 รูป แล้วจัดเรียงพิกเซลต้นทางให้คล้ายเมนูที่เลือก | Color signatures, browser Workers, pixel reassignment |

### 01 / Food Kitchen

อัปโหลดรูปหรือใช้รูปเชฟตัวอย่าง จากนั้นปรับ **ความละเอียด 10–300** (เพิ่มทีละ 10, ค่าเริ่มต้น 10), ความมั่ว และการราดสีเดิมบน Emoji ได้ หน้าเว็บจะแสดงภาพโมเสกกับใบเสร็จที่คำนวณตามอาหารบนภาพ

ระบบเตรียมระดับความละเอียดไว้ล่วงหน้าแบบไม่บล็อกการโต้ตอบ เมื่อพร้อมแล้วกด **“ส่งต่อเข้าครัวถัดไป”** เพื่อใช้ผลลัพธ์ใน Encoder ได้ทันที หรือดาวน์โหลด PNG จากครัวนี้แยกต่างหาก

ครัวแรกทำงานในเบราว์เซอร์ **ไม่อัปโหลดภาพขึ้นเซิร์ฟเวอร์**

### 02 / Emoji Encoder

รับภาพจากครัวแรกหรือเลือกรูปใหม่ รองรับ **PNG / JPEG / WebP**, ไฟล์สูงสุด **8 MB** และภาพไม่เกิน **2048×2048 px** ตามข้อจำกัด API

Encoder ใช้โมเดล **VQ-VAE** แปลงภาพเป็นตารางรหัสจาก Codebook ขนาด 256 รายการ แล้วแมปแต่ละรหัสเป็น Emoji แบบคงที่ รหัสเหล่านี้เป็น **ตัวแทนเวกเตอร์ที่โมเดลเรียนรู้** ไม่ได้หมายถึงวัตถุที่ Emoji นั้นแสดง

สามารถคัดลอกสูตรหรือดาวน์โหลดได้สองรูปแบบ:

- `.emotxt` — ข้อความ JSON ที่มี Emoji tokens, metadata และ checksum
- `.emoc` — Binary ที่เก็บ token IDs และ metadata อย่างกะทัดรัดกว่า

กด **“ส่งสูตรไปครัวถอดรหัส”** เพื่อนำผลไปกรอกใน Decoder โดยไม่ต้องคัดลอกเอง

### 03 / Emoji Decoder

วาง **ข้อความ `.emotxt` แบบเต็ม** หรืออัปโหลด `.emotxt` / `.emoc` แล้วกด **“ประกอบกลับเป็นรูป”** สามารถดาวน์โหลด PNG หรือส่งภาพที่ Decode แล้วไปครัวที่ 4 ได้โดยตรง

**ข้อจำกัดสำคัญ:** EmoCodec เป็น **lossy image compression ไม่ใช่ Encryption** ภาพที่ถอดกลับอาจสูญเสียรายละเอียด และต้องใช้ **Checkpoint ตัวเดียวกันทุกไบต์** กับตอน Encode เพื่อให้ Model ID ตรงกัน

### 04 / Food Match — The Final Dish

ครัวปิดท้ายใช้พิกเซลของรูปที่ Decode มาแล้ว ไปค้นหาว่าภาพอาหารรูปใดในฐานข้อมูล **50,000 ภาพ** มีการกระจายสีใกล้เคียงที่สุด จากนั้นจัดเรียงพิกเซลของรูปต้นทางให้คล้ายเมนูอาหารที่เลือก

- กดปุ่ม **“ส่งรูปนี้ไปค้นหาเมนูอาหารในครัวสุดท้าย”** หลัง Decode โดยไม่ต้องอัปโหลดซ้ำ
- หรืออัปโหลดรูปใหม่ใน Food Match เองก็ได้
- เลือกโหมดค้นหา หมวดอาหาร ชื่อเมนู ความละเอียด และบันทึกภาพผลลัพธ์
- แยก Canvas/CSS/Workers ไว้ใน iframe ของเว็บเดียวกัน เพื่อไม่ให้ชนกับครัวอื่น
- **Lazy loading:** ไม่เริ่มโหลดฐานข้อมูล Food Match ตั้งแต่เปิดหน้าแรก

Food Match **ไม่ใช่ Neural Decoder อีกตัว** แต่เป็นอัลกอริทึมค้นหาภาพด้วยลายเซ็นสีและจัดเรียงพิกเซล จึงเป็นการทดลองคนละแนวทางกับ VQ-VAE

ดัชนีและรูปอาหารไม่ได้อยู่ใน Repo นี้ แต่โหลดจาก **[Food Match data บน GitHub Pages](https://kiatisakk.github.io/food-match/)** ซึ่งต้องใช้เน็ต ข้อมูล Food-101 มีข้อจำกัดด้านการใช้งานเพื่อการวิจัย โปรดตรวจสอบสิทธิ์ก่อนนำไปใช้เชิงพาณิชย์

## Project structure

```text
food-face/
├── index.html                       # หน้า Welcome และทั้ง 4 ครัว
├── style.css                        # ธีมหลัก
├── app.js                           # ครัว Emoji Mosaic และการส่งภาพ
├── renderer.js                      # Renderer + การเตรียมเฟรม
├── menu.js                          # อาหาร Emoji, ราคา, แคลอรี่
├── codec.js / codec.css             # Emoji Encoder/Decoder UI
├── match.js / match.css             # ส่วนปิดจบ + handoff ไป Food Match
├── assets/                          # รูปและไฟล์หน้าเว็บ
├── package.json                     # คำสั่ง run / setup / test
│
├── emioji-encoder/
│   ├── apps/api/main.py             # FastAPI + static routes
│   ├── src/emocodec/
│   │   ├── models/vqvae.py          # VQ-VAE Architecture
│   │   ├── codec/                   # Emoji mapping / .emotxt / .emoc
│   │   ├── service.py               # Encode, Decode, Benchmark
│   │   └── training/                # Training utilities
│   ├── checkpoints/photos_v1.pt     # โมเดลที่ใช้ในเว็บ
│   ├── notebooks/
│   │   └── EmoCodec_Train_Colab.ipynb
│   ├── tests/
│   └── pyproject.toml
│
└── food-match/
    ├── index.html / style.css       # UI ของครัวสุดท้าย
    ├── app.js                       # Crop, search, score, animate
    ├── bridge.js                    # ส่ง PNG จาก Decoder ผ่าน postMessage
    ├── core.js / sigcore.js         # Pixel assignment + color signatures
    ├── worker.js                    # Search Worker
    ├── score-worker.js              # Scoring Worker
    ├── assign-worker.js             # Pixel-assignment Worker
    ├── test.js
    └── tools/                       # ตัวสร้างข้อมูลสำหรับการทดลอง
```

ไม่มี Git submodule หรือ nested repository ใน `emioji-encoder/` และ `food-match/` — ทั้งหมดเป็นส่วนหนึ่งของ Repo หลัก

## Commands

| คำสั่ง | ใช้ทำอะไร |
| --- | --- |
| `npm run setup:python` | สร้าง `emioji-encoder/.venv` และติดตั้ง Python dependencies |
| `npm start` | เปิด Food Face พร้อม FastAPI บน `127.0.0.1:5173` |
| `npm run start:static` | รันแบบ Static โดย **ไม่เปิด Python API** — ครัว Encoder/Decoder รันโมเดลในเบราว์เซอร์แทน |
| `npm run test:api` | ทดสอบ EmoCodec + FastAPI |
| `npm run test:match` | ทดสอบการค้นหา/จัดพิกเซลของ Food Match |

### Train โมเดลใหม่ด้วย Google Colab

ไฟล์ Notebook: [`emioji-encoder/notebooks/EmoCodec_Train_Colab.ipynb`](emioji-encoder/notebooks/EmoCodec_Train_Colab.ipynb)

อัปโหลด Notebook ไปยัง [Google Colab](https://colab.research.google.com/), เปิด GPU Runtime แล้วเทรนตามขั้นตอนในไฟล์ ดาวน์โหลด Checkpoint ที่ได้กลับมาไว้ใน `emioji-encoder/checkpoints/` จากนั้นเลือกใช้โดยตั้ง `EMOCODEC_CHECKPOINT` ก่อนรัน:

```bash
EMOCODEC_CHECKPOINT=emioji-encoder/checkpoints/photos_v1.pt npm start
```

หรือใช้ Path ของ Checkpoint อื่นที่มีอยู่จริง โมเดลเริ่มต้น `photos_v1.pt` (~3.6 MB) อยู่ใน Repo แล้ว **ไม่จำเป็นต้องเทรนเองเพื่อทดลองเว็บ**

คู่มือเพิ่มเติม: [EmoCodec README](emioji-encoder/README.md) · [Colab Guide](emioji-encoder/docs/COLAB_GUIDE.md) · [Dataset Notes](emioji-encoder/docs/DATASETS.md)

## API

เมื่อรัน `npm start` จะใช้พอร์ตเดียวกันทั้งหน้าเว็บและ API:

| Endpoint | หน้าที่ |
| --- | --- |
| `GET /` | หน้า Food Face |
| `GET /api/status` | ตรวจสอบโมเดลที่กำลังใช้งาน |
| `POST /api/encode` | รับไฟล์ภาพ แล้วคืนรหัส Emoji, Binary และภาพ Reconstruction |
| `POST /api/decode` | รับสูตร Emoji หรือไฟล์เข้ารหัส แล้วคืนภาพที่ Decode |
| `GET /food-match/index.html?embedded=1` | UI Food Match ภายในส่วนปิดจบ |

ลองเช็กสถานะจาก Terminal:

```bash
curl http://127.0.0.1:5173/api/status
```

## Troubleshooting

**`[Errno 48] address already in use` / Port 5173 ถูกใช้งาน**

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN      # macOS / Linux
netstat -ano | findstr :5173          # Windows
```

ปิดเซิร์ฟเวอร์เก่าด้วย `Ctrl+C` ใน Terminal นั้น หรือหยุดเฉพาะ PID ที่ตรวจสอบแล้วว่าเป็นเซิร์ฟเวอร์เก่าของคุณ จากนั้นรัน `npm start` ใหม่

**Encoder ไม่พร้อม / ไม่พบโมเดล** — ตรวจว่าได้รัน `npm run setup:python` แล้ว และมีไฟล์ `emioji-encoder/checkpoints/photos_v1.pt`; ลองเปิด `/api/status`

**Food Match ไม่แสดงเมนู** — ตรวจการเชื่อมต่ออินเทอร์เน็ตและการเข้าถึงข้อมูลที่ `https://kiatisakk.github.io/food-match/data/` เพราะข้อมูลไม่ได้อยู่ใน Repo และอาจโหลดช้าหรือบริการต้นทางไม่พร้อม

**เปิด `index.html` ตรง ๆ** — หน้า Static บางส่วนอาจแสดงได้ แต่ API และการโหลดข้อมูล/Workers บางรายการจะไม่ทำงานตามปกติ ควรเปิดผ่าน `npm start`

## Tech stack

**Frontend:** HTML, CSS, Vanilla JavaScript, Canvas 2D, Web Workers, iframe + same-origin `postMessage`

**AI / Backend:** Python, PyTorch (VQ-VAE), FastAPI, Uvicorn, Pillow, NumPy

**Food Match:** Color signatures, histogram-based search, pixel reassignment, Food-101 research dataset index

## Limitations & credits

- ผลจากครัวแรกเป็น **งานภาพ/ความบันเทิง** ราคากับแคลอรี่ในใบเสร็จเป็นค่าที่แต่งขึ้น ไม่ใช่ข้อมูลโภชนาการ
- EmoCodec **ไม่ใช่ระบบเข้ารหัสเพื่อความปลอดภัย** ไม่ควรส่งรูปส่วนตัวโดยคิดว่า Emoji จะทำให้ข้อมูลเป็นความลับ และภาพที่ Decode ได้เป็นแบบสูญเสียรายละเอียด
- ตอนรันด้วย `npm start` การ Encode/Decode จะส่งภาพไปยัง **FastAPI บนเครื่อง** ส่วนบน GitHub Pages ทุกอย่างประมวลผลในเบราว์เซอร์ ไม่อัปโหลดภาพไปไหน
- Food Match ใช้ดัชนีภาพ Food-101 และอาศัยข้อมูลภายนอก การทำงาน/ความเร็วขึ้นอยู่กับเครือข่ายและแหล่งข้อมูล ทั้งนี้ผลเป็นการ **ย้ายพิกเซล** ไม่ใช่สร้างภาพอาหารใหม่ด้วย Generative AI
- ต้นทาง [Food-101 (ETH Zürich)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — ตรวจเงื่อนไขและสิทธิ์การใช้ภาพก่อนเผยแพร่หรือใช้เชิงพาณิชย์
- มีม: [Would you let him cook? — Imgflip](https://imgflip.com/i/7imzuy) และ [Let Him Cook — Know Your Meme](https://knowyourmeme.com/photos/2517089-let-him-cook-let-that-boy-cook)

---

*0 Michelin stars. Many experiments. Let him cook.*
