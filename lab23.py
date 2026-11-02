import re
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---------- fungsi dari template ----------
def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def potong(teks, ukuran=800, tumpang=100):
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]

# ======================= LAB 2: bangun indeks =======================
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")

def bangun_koleksi(nama, ukuran, tumpang):
    kol = client.get_or_create_collection(
        name=nama, embedding_function=emb,
        metadata={"hnsw:space": "cosine"})
    for pdf in Path("dokumen").glob("*.pdf"):
        chunks = potong(baca_pdf(pdf), ukuran, tumpang)
        kol.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))])
        print(f"[{nama}] {pdf.name} → {len(chunks)} chunk")
    return kol

kol_500  = bangun_koleksi("pedoman_500", 500, 75)
kol_1000 = bangun_koleksi("pedoman_1000", 1000, 150)

def cari(kol, pertanyaan, k=3):
    h = kol.query(query_texts=[pertanyaan], n_results=k)
    print(f"\nQ: {pertanyaan}")
    for doc, meta, jarak in zip(h["documents"][0], h["metadatas"][0], h["distances"][0]):
        print(f"  [chunk {meta['chunk']}] jarak={jarak:.3f}")
        print("    ", re.sub(r"\s+", " ", doc)[:200], "...")

print("\n" + "=" * 60 + "\nLAB 2 - langkah 2: 5 pertanyaan biasa (koleksi 1000)\n" + "=" * 60)
for q in [
    "Berapa SKS minimal untuk gelar Sarjana Teknik?",
    "Apa syarat kelulusan tahap akhir?",
    "Mata kuliah apa yang tidak boleh mendapat nilai D?",
    "Berapa SKS maksimal mata kuliah lintas program studi?",
    "Apa fasilitas penunjang pendidikan di Program Studi Informatika?",
]:
    cari(kol_1000, q)

print("\n" + "=" * 60 + "\nLAB 2 - langkah 3: 2 pertanyaan parafrase\n" + "=" * 60)
for q in [
    "Berapa banyak nilai jelek yang masih boleh dimiliki saat lulus?",   # dokumen: "nilai D"
    "Bukti kemampuan bahasa Inggris apa yang diminta sebelum wisuda?",   # dokumen: EPT / yudisium
]:
    cari(kol_1000, q)

# ======================= LAB 3: Hit@k =======================
UJI = [
    ("Berapa SKS minimal yang harus ditempuh untuk meraih gelar Sarjana Teknik?", "minimal 144 SKS"),
    ("Berapa SKS maksimal mata kuliah pilihan yang boleh diambil dari program studi lain?", "maksimal 4 sks"),
    ("Berapa banyak nilai D yang masih ditoleransi untuk lulusan Informatika?", "Informatika: 4 matakuliah"),
    ("Dokumen apa yang harus diserahkan sebelum yudisium terkait kemampuan bahasa Inggris?", "English Proficiency Test"),
    ("Pihak mana yang mengajar mata kuliah Bahasa Inggris untuk Fakultas Teknik?", "FKIP"),
    ("Berapa SKS yang harus sudah ditempuh sebelum boleh mengambil Kerja Praktik?", "menempuh 96 sks"),
    ("Apa kode mata kuliah Tugas Akhir?", "INF499"),
    ("Mata kuliah apa yang harus diambil lebih dulu sebelum Pembelajaran Mesin?", "(P) Kecerdasan Buatan"),
    ("Berapa lama durasi program magang di industri?", "4 -6 bulan"),
    ("Sejak kapan Program Studi Informatika mulai menyelenggarakan pendidikan?", "tahun ajaran 2024/2025"),
    ("Di jalan apa letak kampus Kalijudan?", "Jl. Kalijudan 37"),
    ("Apa kode mata kuliah Keamanan Data dan Informasi?", "INF257"),
    ("Apa kode mata kuliah Komputasi Awan?", "INF307"),
    ("Apa kode mata kuliah Pengantar Pemrograman?", "INF105"),
    ("Selain magang, kegiatan apa lagi yang termasuk dalam program MBKM?", "pertukaran pelajar"),
]

norm = lambda s: re.sub(r"\s+", " ", s).lower()   # abaikan spasi/baris baru & huruf besar

def hit_at_k(kol, uji, k=3):
    benar, gagal = 0, []
    for no, (pertanyaan, kunci) in enumerate(uji, 1):
        docs = kol.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        if any(norm(kunci) in norm(d) for d in docs):
            benar += 1
        else:
            gagal.append(no)
    return benar / len(uji), gagal

print("\n" + "=" * 60 + "\nLAB 3 - Hit@k\n" + "=" * 60)
print("Ukuran | k | Hit@k | pertanyaan gagal (nomor)")
for nama, kol in [("500", kol_500), ("1000", kol_1000)]:
    for k in (1, 3, 5):
        skor, gagal = hit_at_k(kol, UJI, k)
        print(f"{nama:>6} | {k} | {skor:.2f}  | {gagal}")