# let him cook

เว็บสำหรับแข่ง stupid: โยนรูปหน้าลงไป แล้วมันจะหั่นเป็นพิกเซล เปลี่ยนแต่ละช่องเป็นอีโมจิอาหารที่สีใกล้ที่สุด พร้อมใบเสร็จที่ไม่มีประโยชน์ ด้านล่างเป็น Emoji Encoder ที่แปลงรูปเป็นรหัสอีโมจิและประกอบกลับเป็นรูปด้วยโมเดล EmoCodec

หน้าแรกเป็นมีม shef กด “ช่างมัน ให้เขาทำ” เพื่อเข้าครัว กดชื่อเว็บเพื่อกลับไปหน้าแรก เว็บแบ่งเป็นสี่ครัวตามลำดับ: (1) ครัวอาหารทำภาพโมเสกอีโมจิ (2) Emoji Encoder แปลงภาพเป็นรหัสอีโมจิและแสดงเฉพาะสูตรพร้อมปุ่มคัดลอก/.emotxt/.emoc โดยยังไม่แสดงภาพ Decode (3) Emoji Decoder แยกเป็นส่วนด้านล่าง รับสูตรเต็มจากการวางข้อความหรืออัปโหลด .emotxt/.emoc แล้วค่อยกดประกอบกลับเพื่อแสดงภาพ (4) Food Match รับภาพที่ Decode แล้วไปลองเรียงพิกเซลเป็นจานอาหารที่ใกล้เคียง กด “ส่งต่อเข้าครัวถัดไป ↓” ในครัวแรกเพื่อส่งภาพโมเสกเข้าช่อง Encoder อัตโนมัติ หรือกด “ส่งสูตรไปครัวถอดรหัส ↓” หลัง Encode เพื่อเติมสูตรใน Decoder โดยยังไม่ Decode จนกดปุ่มเอง ทุกส่วนใช้ UI โทนครีมกับสีซอสมะเขือเทศเดียวกัน

ครัวอาหารประมวลผลรูปในเบราว์เซอร์ ส่วน Emoji Encoder ส่งรูปให้ FastAPI ที่รันบนเครื่องคุณ ใช้โมเดลที่ฝึกไว้เพื่อสร้างรหัสอีโมจิและประกอบรูปกลับ รองรับ PNG/JPEG/WebP ไม่เกิน 8 MB และด้านละไม่เกิน 2048 px ดาวน์โหลดหรือคัดลอกข้อความเต็ม .emotxt, ดาวน์โหลด .emoc และรูป PNG ได้ รูปที่ประกอบกลับสูญเสียรายละเอียดและต้องใช้โมเดลเดียวกับตอนแปลง

หน้าเว็บเป็น HTML + CSS + JS ธรรมดา ไม่ต้อง build โดยใช้ API จากโฟลเดอร์ `./emioji-encoder` ภายใน repository เดียวกัน

## รัน

```sh
cd food-face       # จากโฟลเดอร์ stupidhack
npm run setup:python # ติดตั้ง Python dependencies (ทำครั้งแรก)
npm start          # เปิด http://127.0.0.1:5173 พร้อม API ในพอร์ตเดียวกัน
```

โปรเจกต์ EmoCodec อยู่ภายใน `emioji-encoder/` โดย `npm run setup:python` จะสร้าง Python virtual environment และติดตั้ง Dependencies ให้เอง เซิร์ฟเวอร์เลือก `emioji-encoder/checkpoints/photos_v1.pt` ที่เก็บใน repository กำหนดโมเดลเองได้ด้วย `EMOCODEC_CHECKPOINT=/absolute/path/model.pt npm start`

หากต้องการใช้เฉพาะครัวอาหารแบบเดิม ใช้ `npm run start:static` หรือดับเบิลคลิก `index.html` ได้ ในโหมดนี้ Encoder จะแสดงสถานะเชื่อมต่อไม่ได้และปิดปุ่มแปลงรูป

เมื่อเปิดไฟล์ตรง ๆ รูปตัวอย่างในครัวจะเป็นหน้าเชฟที่วาดขึ้น เพื่อให้ canvas และการบันทึก PNG ใช้ได้แม้เบราว์เซอร์จำกัดภาพจาก `file://` หน้าเปิดยังแสดงมีม shef ตามปกติ


## โครงสร้างโปรเจกต์แบบรวม

```text
food-face/
├── package.json
├── index.html                 # ครัว Food Emoji
├── codec.js                   # UI Encoder/Decoder
└── emioji-encoder/            # Backend และโมเดล AI ใน repo เดียวกัน
    ├── apps/api/main.py       # FastAPI
    ├── src/emocodec/          # Encoder/Decoder แบบ VQ-VAE
    ├── checkpoints/photos_v1.pt # โมเดลที่เทรนแล้ว (tracked)
    └── notebooks/EmoCodec_Train_Colab.ipynb
```

Clone repo นี้แล้วรัน `npm run setup:python` และ `npm start` จาก root ได้ ไม่ต้อง Clone หรือเตรียม sibling directory อีกชุด ไม่ต้อง Push `.venv`, `datasets`, หรือ `outputs` ขึ้น GitHub โมเดล `photos_v1.pt` ขนาดประมาณ 3.6 MB ถูก commit เพื่อให้ทดลอง Encode/Decode ได้หลังติดตั้ง

> หมายเหตุ: ตัว Server ใช้ CPU โดยค่าเริ่มต้น และโมเดลเป็น Learned Lossy Codec ไม่ใช่ Encryption. ใช้ GitHub Repo นี้เป็น source เดียว ส่วน checkout เดิมนอกโฟลเดอร์เก็บไว้สำรองชั่วคราว


## 04 / ครัวปิดจบ — Food Match

นำโค้ดจาก food-match มารวมไว้ในโฟลเดอร์ food-match/ ภายใต้ repo เดียวกัน (ไม่มี nested Git หรือ submodule)
โดยแสดงเป็นส่วนที่ 4 ต่อจาก Emoji Decoder ใช้ iframe same-origin แยก Worker/Canvas/CSS ไม่ให้ชนกับ Food Face

- เลื่อนท้ายหน้า หรือกด 04 Food Match ในเมนู เพื่อเปิดระบบค้นหาอาหารแบบ lazy loading
- กดปุ่มส่งภาพไปครัวสุดท้าย หลัง Decode เพื่อส่ง PNG ไป Food Match โดยตรง ไม่ต้องอัปโหลดซ้ำ
- Food Match รองรับการอัปโหลดภาพอื่นเอง แสดงรูปอาหารที่ใกล้สุด และจัดพิกเซลของผู้ใช้เป็นภาพอาหาร
- ดัชนี 50,000 ภาพและภาพ Food-101 เก็บที่ https://kiatisakk.github.io/food-match/data/ ไม่ถูกคัดลอกเข้า Git
- ต้องมีอินเทอร์เน็ตเพื่อใช้งานส่วน Food Match และ Food-101 จำกัดการใช้งานเพื่อการวิจัย
- โมเดล AI สำหรับ Encode/Decode ยังคงใช้ emioji-encoder/checkpoints/photos_v1.pt

รันจาก repo root ด้วย npm run setup:python (ครั้งแรก) แล้ว npm start หากพอร์ต 5173 มีเซิร์ฟเวอร์เก่า ให้ปิดก่อน

## ไฟล์

| ไฟล์ | หน้าที่ |
| --- | --- |
| `index.html` | โครงหน้าเว็บ |
| `style.css` | หน้าเปิดและครัวโทนครีมกับสีซอสมะเขือเทศ รองรับมือถือและ reduced motion |
| `menu.js` | รายการอาหาร: อีโมจิ, ชื่อ, ราคา, kcal — แก้ตรงนี้เพื่อเพิ่มเมนู |
| `app.js` | จัดการรูป อัปเดตภาพจากแคช ทำใบเสร็จ บันทึก PNG |
| `codec.js` | เชื่อม API, จัดการแท็บ Encoder/Decoder, ตรวจไฟล์ และคัดลอก/ดาวน์โหลดผลลัพธ์ |
| `codec.css` | UI ครัว Encoder ที่ใช้สี ฟอนต์ และรูปแบบปุ่มของเว็บแรก |
| `renderer.js` | วาดอาหารจาก sprite atlas และเตรียมภาพแต่ละระดับใน Web Worker |
| `assets/` | ภาพมีมในเครื่องและ favicon |

ความละเอียดเพิ่มทีละ 10 ตั้งแต่ 10 ถึง 300 (เริ่มต้น 50) ทุกครั้งที่อัปโหลดรูป Web Worker จะเตรียมภาพและยอดใบเสร็จครบ 30 ระดับไว้เบื้องหลัง โดยทำระดับที่เลือกก่อนแล้วไล่เตรียมระดับใกล้เคียง สไลเดอร์เรียกภาพที่เตรียมไว้ผ่าน `requestAnimationFrame` และ crossfade สั้น ๆ โดยไม่วาดอีโมจิใหม่ขณะลาก

เมื่อเปลี่ยนรูป ความมั่ว ซอส หรือสั่งจานใหม่ จะยกเลิกงานชุดเก่า ล้างแคช และเตรียมชุดใหม่ หากเลือกขั้นที่ยังไม่พร้อม จะให้ขั้นนั้นทำก่อน พร้อมแสดงความคืบหน้า ถ้าเบราว์เซอร์ไม่มี Web Worker/OffscreenCanvas จะเตรียมภาพช่วงที่หน้าเว็บว่างแทน ภาพตัวอย่างมีด้านยาว 720 พิกเซล แคชครบ 30 ระดับใช้พื้นที่ bitmap ไม่เกินประมาณ 60 MiB พร้อมข้อมูลสีและอาหารแต่ละช่อง และคืนพื้นที่เมื่อเริ่มชุดใหม่ ส่วน PNG ใช้ข้อมูลอาหารชุดเดียวกับภาพที่เตรียมไว้ จึงตรงกับใบเสร็จ และวาดด้วยความละเอียดสูงสุด 4096 พิกเซล

## ภาพมีม

- `shef.png`: ภาพที่ผู้ใช้ให้มา
- `let-him-cook-rat.jpg`: [Would you let him cook? — Imgflip](https://imgflip.com/i/7imzuy)
- `let-him-cook.png`: [Let Him Cook — Know Your Meme](https://knowyourmeme.com/photos/2517089-let-him-cook-let-that-boy-cook)
