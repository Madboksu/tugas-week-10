import random
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

# ======================= LAB 1 =======================
pdf_contoh = sorted(Path("dokumen").glob("*.pdf"))[0]
teks = baca_pdf(pdf_contoh)
print(teks[:500])                      # langkah 1

for ukuran in (500, 1000, 1500):       # langkah 2
    tumpang = int(ukuran * 0.15)       # overlap 15%
    chunks = potong(teks, ukuran, tumpang)
    print(f"\n##### Ukuran {ukuran}, overlap {tumpang}: {len(chunks)} chunk #####")
    for c in random.sample(chunks, min(3, len(chunks))):   # langkah 3
        print("-" * 50)
        print(c)

# ======================= LAB 2 =======================
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

# dua koleksi sekaligus (dipakai juga di Lab 3)
kol_500  = bangun_koleksi("pedoman_500", 500, 75)
kol_1000 = bangun_koleksi("pedoman_1000", 1000, 150)

def cari(kol, pertanyaan, k=3):
    h = kol.query(query_texts=[pertanyaan], n_results=k)
    print(f"\nQ: {pertanyaan}")
    for doc, meta, jarak in zip(h["documents"][0], h["metadatas"][0], h["distances"][0]):
        print(f"  [{meta['sumber']} · chunk {meta['chunk']}] jarak={jarak:.3f}")
        print("   ", doc[:200].replace("\n", " "), "...")

# langkah 2: 5 pertanyaan biasa (ganti dengan pertanyaan dari dokumen Anda)
for q in ["pertanyaan 1", "pertanyaan 2", "pertanyaan 3", "pertanyaan 4", "pertanyaan 5"]:
    cari(kol_1000, q)

# langkah 3: 2 pertanyaan parafrase (hindari kata yang sama dengan dokumen)
for q in ["parafrase 1", "parafrase 2"]:
    cari(kol_1000, q)

# ======================= LAB 3 =======================
# (pertanyaan, kata kunci yang HARUS ada di chunk jawaban) → isi 15 pasang
UJI = [
    ("Apa syarat ...?", "kata kunci"),
    # ... total 15
]

def hit_at_k(kol, uji, k=3):
    benar = 0
    for pertanyaan, kunci in uji:
        docs = kol.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        if any(kunci.lower() in d.lower() for d in docs):
            benar += 1
    return benar / len(uji)

print("\nUkuran chunk | k | Hit@k")
for nama, kol in [("500", kol_500), ("1000", kol_1000)]:
    for k in (1, 3, 5):
        print(f"{nama:>12} | {k} | {hit_at_k(kol, UJI, k):.2f}")