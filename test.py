import requests
import logging
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_calling():
    """API 호출 테스트 함수"""

    # API 설정
    api_endpoint = "http://localhost:8010/loader/hf/dataset/info"
    method_upper = "GET"
    timeout = 30

    # 요청 데이터 (로그에서 가져온 파라미터)
    request_data = {
        'dataset_path': 'CocoRoF/electronic-commerce-v1',
        'hugging_face_user_id': 'CocoRoF',
        'hugging_face_token': ''
    }

    # 공통 헤더 설정
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'PlateeRAG-APICallingTool/1.0'
    }

    try:
        if method_upper == "GET":
            params = request_data if request_data else None
            logger.info(f"Making GET request to {api_endpoint} with params: {params}")
            response = requests.get(
                api_endpoint,
                params=params,
                headers=headers,
                timeout=timeout
            )
            logger.info(f"GET request to {api_endpoint} completed with status code: {response}")

            # 응답 상태 코드 확인
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"API call successful: {result}")
                    print("=== API Response ===")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return result
                except ValueError:
                    logger.info(f"API call successful, but response is not JSON: {response.text}")
                    print("=== API Response (Text) ===")
                    print(response.text)
                    return response.text
            else:
                logger.error(f"API call failed with status code {response.status_code}. Response: {response.text}")
                print(f"Error: {response.status_code} - {response.text}")
                return None

    except requests.exceptions.Timeout:
        error_msg = f"API call timed out after {timeout} seconds"
        logger.error(error_msg)
        print(error_msg)
        return None
    except requests.exceptions.ConnectionError:
        error_msg = f"Failed to connect to API endpoint: {api_endpoint}"
        logger.error(error_msg)
        print(error_msg)
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return None
    except (ValueError, TypeError) as e:
        error_msg = f"Invalid data format: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return None

def test_multiple_endpoints():
    """여러 엔드포인트 테스트"""

    endpoints = [
        {
            "url": "http://localhost:8010/loader/hf/dataset/info",
            "params": {
                'dataset_path': 'CocoRoF/electronic-commerce-v1',
                'hugging_face_user_id': 'CocoRoF',
                'hugging_face_token': ''
            }
        },
        {
            "url": "http://192.168.219.101:8000/api/node/categories",
            "params": {}
        }
    ]

    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n=== Test {i}: {endpoint['url']} ===")
        test_single_endpoint(endpoint['url'], endpoint['params'])

def test_single_endpoint(api_endpoint, request_data):
    """단일 엔드포인트 테스트"""

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'PlateeRAG-APICallingTool/1.0'
    }

    try:
        params = request_data if request_data else None
        logger.info(f"Making GET request to {api_endpoint} with params: {params}")

        response = requests.get(
            api_endpoint,
            params=params,
            headers=headers,
            timeout=30
        )

        logger.info(f"GET request to {api_endpoint} completed with status code: {response.status_code}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except ValueError:
                print(response.text)
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    print("=== API 호출 테스트 시작 ===")

    # 메인 테스트 실행
    result = test_api_calling()

    print("\n=== 추가 엔드포인트 테스트 ===")
    # 여러 엔드포인트 테스트
    test_multiple_endpoints()

    print("\n=== 테스트 완료 ===")
