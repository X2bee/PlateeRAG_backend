import requests
import json

MAPPING_TABLE = {
    "패션의류": "50000000",
    "패션잡화": "50000001",
    "화장품/미용": "50000002",
    "디지털/가전": "50000003",
    "가구/인테리어": "50000004",
    "출산/육아": "50000005",
    "식품": "50000006",
    "스포츠/레저": "50000007",
    "생활/건강": "50000008",
    "여가/생활편의": "50000009",
    "도서": "50005542"
}

def _get_category_code_by_name(category_name: str) -> tuple:
    """카테고리명으로 카테고리 코드를 찾습니다. 대소문자, 띄어쓰기, 특수문자 무시하고 유사 매칭합니다."""
    def normalize_text(text):
        """텍스트 정규화: 소문자 변환, 띄어쓰기/특수문자 제거"""
        import re
        text = text.lower()
        text = re.sub(r'[_\-\s&+/]', '', text)  # 띄어쓰기, 언더바, 하이픈, &, +, / 제거
        return text

    def is_similar(str1, str2, max_diff=1):
        """문자열 유사도 체크 (편집 거리 기반)"""
        if abs(len(str1) - len(str2)) > max_diff:
            return False

        # 간단한 편집거리 계산
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n] <= max_diff

    normalized_input = normalize_text(category_name)

    # 1. 정확 매칭 시도
    for category, code in MAPPING_TABLE.items():
        normalized_category = normalize_text(category)
        if normalized_category == normalized_input:
            return category, code

    # 2. 부분 매칭 시도 (입력이 카테고리에 포함되거나 카테고리가 입력에 포함)
    for category, code in MAPPING_TABLE.items():
        normalized_category = normalize_text(category)
        if normalized_input in normalized_category or normalized_category in normalized_input:
            return category, code

    # 3. 유사 매칭 시도 (편집거리 1 이하)
    for category, code in MAPPING_TABLE.items():
        normalized_category = normalize_text(category)
        if is_similar(normalized_category, normalized_input, max_diff=1):
            return category, code

    # 3. 기본값 반환 (패션의류)
    return "패션의류", "50000000"

def get_naver_shopping_insight(category_name: str, category_code: str = None, recent_days:int = 1):
    if category_code is None:
        matched_category, category_code = _get_category_code_by_name(category_name)
        category_name = matched_category

    url = "https://datalab.naver.com/shoppingInsight/getKeywordRank.naver"
    params = {
        "timeUnit": "date",
        "cid": category_code
    }

    headers = {
        "referer": "https://datalab.naver.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    }

    try:
        response = requests.post(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            try:
                raw_data = response.json()
                # date, datetime, ranks만 남기고 나머지 필드 제거
                processed_data = []
                for item in raw_data:
                    # ranks에서 linkId 제거
                    cleaned_ranks = []
                    for rank_item in item.get("ranks", []):
                        cleaned_rank = {
                            "rank": rank_item.get("rank"),
                            "keyword": rank_item.get("keyword")
                        }
                        cleaned_ranks.append(cleaned_rank)

                    cleaned_item = {
                        "date": item.get("date"),
                        "datetime": item.get("datetime"),
                        "ranks": cleaned_ranks
                    }
                    processed_data.append(cleaned_item)

                # 최신 데이터부터 지정된 개수만큼만 반환
                if recent_days > 0:
                    processed_data = processed_data[-recent_days:]

                print("Processed JSON Response:")
                result = json.dumps(processed_data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 원본 텍스트 출력
                print("Raw Response:")
                result = response.text

            return result

        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response.text}")
