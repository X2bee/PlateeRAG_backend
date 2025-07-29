import os
import asyncio
from document_processor import DocumentProcessor
from config.persistent_config import (
    get_all_persistent_configs, 
    refresh_all_configs, 
    save_all_configs, 
    export_config_summary,
    PersistentConfig
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # test 파일과 동일 폴더(data/)
TEST_FILES = [
    "Nest.js vs Rust 비교 보고서.doc"
]

async def test_all_functions():
    processor = DocumentProcessor()

    print("\n1. [get_supported_types] 지원 파일타입:", processor.get_supported_types())

    for filename in TEST_FILES:
        file_path = os.path.join(DATA_DIR, filename)
        print("\n\n====== [파일] %s ======" % filename)
        if not os.path.exists(file_path):
            print(f"- 파일 없음: {file_path}")
            continue

        # 2. get_file_info, get_file_category, validate_file_format
        file_info = processor.get_file_info(file_path)
        print("[get_file_info] :", file_info)
        is_valid, ext = processor.validate_file_format(file_path)
        print(f"[validate_file_format] {ext} -> {is_valid}")
        category = processor.get_file_category(ext)
        print(f"[get_file_category] {ext} -> {category}")

        # 3. extract_text_from_file
        try:
            text = await processor.extract_text_from_file(file_path)
            print(f"[extract_text_from_file] 추출 텍스트 (앞 500자):\n{text[:500]}")
        except Exception as e:
            print(f"[extract_text_from_file] 오류: {e}")
            continue

        # 4. clean_text / clean_code_text
        if category == "code":
            cleaned = processor.clean_code_text(text, ext)
            print("[clean_code_text] 결과 (앞 300자):\n", cleaned[:300])
        else:
            cleaned = processor.clean_text(text)
            print("[clean_text] 결과 (앞 300자):\n", cleaned[:300])

        # 5. chunk_text / chunk_code_text
        try:
            if category == "code":
                chunks = processor.chunk_code_text(cleaned, ext)
            else:
                chunks = processor.chunk_text(cleaned)
            print(f"[chunk_{category}_text] 청크 개수:", len(chunks))
            if chunks:
                print(f"  > 첫번째 청크 (앞 200자):\n{chunks[0][:200]}")
        except Exception as e:
            print(f"[chunk_{category}_text] 오류: {e}")
            chunks = []

        # 6. estimate_chunks_count
        estimated_chunks = processor.estimate_chunks_count(cleaned)
        print(f"[estimate_chunks_count] 예상 청크 개수: {estimated_chunks}")

        # 7. 결과 저장
        out_path = file_path + ".parsed.txt"
        try:
            with open(out_path, "w", encoding="utf-8") as out_f:
                for i, chunk in enumerate(chunks):
                    out_f.write(f"\n=== Chunk {i+1} ===\n")
                    out_f.write(chunk)
                    out_f.write("\n")
            print(f"[저장] {out_path}")
        except Exception as e:
            print(f"[저장 오류] {e}")

if __name__ == "__main__":
    asyncio.run(test_all_functions())
