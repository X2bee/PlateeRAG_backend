# controller/gaudi_proxy.py
from fastapi import APIRouter, Request, HTTPException
import httpx

router = APIRouter(prefix="/api/gaudi", tags=["gaudi-proxy"])

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_gaudi_requests(path: str, request: Request):
    """Gaudi 서비스로 모든 요청 프록시"""
    gaudi_service_url = "http://172.17.0.1:8080"
    
    async with httpx.AsyncClient() as client:
        # 요청 바디 및 헤더 복사
        body = await request.body()
        headers = dict(request.headers)
        
        # Gaudi 서비스로 포워딩
        response = await client.request(
            method=request.method,
            url=f"{gaudi_service_url}/api/gaudi/{path}",
            content=body,
            headers=headers,
            params=request.query_params
        )
        
        return response.json()