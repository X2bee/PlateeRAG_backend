import requests
import json
from urllib.parse import quote
import logging

logger = logging.getLogger(__name__)

def process_hsmoa_response(raw_data, include_image=False):
    """
    HSMOA API 응답 데이터를 전처리합니다.
    필요한 정보만 추출: 카테고리, 이미지주소, 상품명, name_query, 가격, 리뷰개수, 사이트

    Args:
        raw_data (dict): 원본 API 응답 데이터
        include_image (bool): 이미지 정보 포함 여부 (기본값: False)

    Returns:
        dict: 전처리된 데이터
    """
    if not raw_data or 'results' not in raw_data:
        return raw_data

    processed_results = []

    for item in raw_data.get('results', []):
        processed_item = {
            'category1': item.get('category1'),
            'category2': item.get('category2'),
            'category3': item.get('category3'),
            'name': item.get('name'),
            'name_query': item.get('name_query'),
            'price': item.get('price'),
            'review_count': item.get('review_count'),
            'site': item.get('site')
        }

        # include_image가 True일 때만 이미지 정보 포함
        if include_image:
            processed_item['image'] = item.get('image')

        processed_results.append(processed_item)

    # 기본 메타데이터는 유지
    processed_data = {
        'limit': raw_data.get('limit'),
        'offset': raw_data.get('offset'),
        'results': processed_results
    }

    return processed_data

def process_hsmoa_v1_response(raw_data, include_image=False):
    """
    HSMOA V1 API 응답 데이터를 전처리합니다.

    Args:
        raw_data (dict): 원본 API 응답 데이터
        include_image (bool): 이미지 정보 포함 여부 (기본값: False)

    Returns:
        dict: 전처리된 데이터
    """
    if not raw_data or 'results' not in raw_data:
        return raw_data

    processed_results = []

    for item in raw_data.get('results', []):
        processed_item = {
            'pid': item.get('pid'),
            'recent_broadcast_end_datetime': item.get('recent_broadcast_end_datetime'),
            'review_count': item.get('review_count'),
            'name': item.get('name'),
            'price': item.get('price'),
            'site': item.get('site'),
            'category1': item.get('category1'),
            'category2': item.get('category2'),
            'category3': item.get('category3')
        }

        # include_image가 True일 때만 이미지 정보 포함
        if include_image:
            processed_item['image'] = item.get('image')

        processed_results.append(processed_item)

    # 기본 메타데이터는 유지
    processed_data = {
        'limit': raw_data.get('limit'),
        'results': processed_results
    }

    return processed_data

def search_hsmoa_v1_api(query, source="tvshop", limit=10, direction="past", include_image=False):
    """
    HSMOA V1 API를 사용하여 검색을 수행합니다.

    Args:
        query (str): 검색어
        source (str): 소스 (기본값: "tvshop")
        limit (int): 검색 결과 개수 제한 (기본값: 10)
        direction (str): 방향 (기본값: "past")
        include_image (bool): 이미지 정보 포함 여부 (기본값: False)

    Returns:
        dict: API 응답 결과
    """
    base_url = "https://api.hsmoa.net/v1/search"

    # 검색어 URL 인코딩
    encoded_query = quote(query)

    params = {
        "query": encoded_query,
        "source": source,
        "limit": limit,
        "direction": direction
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)

        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                raw_data = response.json()

                # 응답 데이터 전처리
                processed_data = process_hsmoa_v1_response(raw_data, include_image=include_image)

                print("Processed V1 JSON Response:")
                print(json.dumps(processed_data, indent=2, ensure_ascii=False))
                return processed_data
            except json.JSONDecodeError:
                print("Raw Response:")
                print(response.text)
                return response.text
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def search_hsmoa_popular_api(query, source="uniq_tvshop", limit=10, include_image=False):
    """
    HSMOA 인기 상품 API를 사용하여 검색을 수행합니다.

    Args:
        query (str): 검색어
        source (str): 소스 (기본값: "uniq_tvshop")
        limit (int): 검색 결과 개수 제한 (기본값: 10)
        include_image (bool): 이미지 정보 포함 여부 (기본값: False)

    Returns:
        dict: API 응답 결과
    """
    base_url = "https://api.hsmoa.net/v1/search"

    # 검색어 URL 인코딩
    encoded_query = quote(query)

    params = {
        "query": encoded_query,
        "source": source,
        "limit": limit
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)

        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                raw_data = response.json()

                # 응답 데이터 전처리 (V1과 동일한 구조)
                processed_data = process_hsmoa_v1_response(raw_data, include_image=include_image)

                print("Processed Popular JSON Response:")
                print(json.dumps(processed_data, indent=2, ensure_ascii=False))
                return processed_data
            except json.JSONDecodeError:
                print("Raw Response:")
                print(response.text)
                return response.text
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def search_hsmoa_api(query, offset=0, limit=20, order="rel", site="", platform="web", include_image=False):
    """
    HSMOA API를 사용하여 검색을 수행합니다.

    Args:
        query (str): 검색어
        offset (int): 검색 결과 시작 위치 (기본값: 0)
        limit (int): 검색 결과 개수 제한 (기본값: 20)
        order (str): 정렬 순서 (기본값: "rel")
        site (str): 사이트 필터 (기본값: "")
        platform (str): 플랫폼 (기본값: "web")
        include_image (bool): 이미지 정보 포함 여부 (기본값: False)

    Returns:
        dict: API 응답 결과
    """
    base_url = "https://api.hsmoa.net/v2/search/ep"

    # 검색어 URL 인코딩
    encoded_query = quote(query)

    params = {
        "query": encoded_query,
        "offset": offset,
        "limit": limit,
        "order": order,
        "site": site,
        "platform": platform
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)

        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                raw_data = response.json()

                # 응답 데이터 전처리
                processed_data = process_hsmoa_response(raw_data, include_image=include_image)

                print("Processed JSON Response:")
                print(json.dumps(processed_data, indent=2, ensure_ascii=False))
                return processed_data
            except json.JSONDecodeError:
                print("Raw Response:")
                print(response.text)
                return response.text
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def _product_search_tool(query: str, search_type: str = "popular", limit: int = 10, include_image: bool = False):
    if search_type not in ["popular", "future", "past", "sales"]:
        logger.info(f"Invalid type '{search_type}' provided. Defaulting to 'popular'.")
        search_type = "popular"

    if search_type == "popular":
        return search_hsmoa_popular_api(query, limit=limit, include_image=include_image)
    elif search_type == "future":
        return search_hsmoa_v1_api(query, source="tvshop", limit=limit, direction="future", include_image=include_image)
    elif search_type == "past":
        return search_hsmoa_v1_api(query, source="tvshop", limit=limit, direction="past", include_image=include_image)
    elif search_type == "sales":
        return search_hsmoa_api(query, limit=limit, include_image=include_image)
    else:
        return {"error": "Invalid search type specified."}
