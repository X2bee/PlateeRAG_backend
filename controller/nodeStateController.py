from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger("node-controller-state")
router = APIRouter(prefix="/api/node-state", tags=["node-state"])

@router.get("/registry", response_model=Dict[str, Any])
async def get_node_registry_from_state(request: Request):
    """
    app.state에서 노드 레지스트리 정보를 가져옵니다.
    """
    try:
        # app.state에서 직접 접근
        node_registry = request.app.state.node_registry
        node_count = request.app.state.node_count
        
        return {
            "status": "success",
            "source": "app.state",
            "node_count": node_count,
            "available_nodes": [node["id"] for node in node_registry],
            "registry_data": node_registry
        }
        
    except AttributeError as e:
        logger.error(f"app.state attributes not found: {e}")
        raise HTTPException(status_code=500, detail="Node registry not initialized in app.state")
    except Exception as e:
        logger.error(f"Error getting node registry from state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/nodes", response_model=List[Dict[str, Any]])
async def get_all_nodes_from_state(request: Request):
    """
    app.state에서 모든 노드 정보를 반환합니다.
    """
    try:
        return request.app.state.node_registry
        
    except AttributeError:
        raise HTTPException(status_code=500, detail="Node registry not initialized in app.state")
    except Exception as e:
        logger.error(f"Error getting all nodes from state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/node/{node_id}", response_model=Dict[str, Any])
async def get_node_by_id_from_state(node_id: str, request: Request):
    """
    app.state에서 특정 ID의 노드 정보를 반환합니다.
    """
    try:
        node_registry = request.app.state.node_registry
        
        for node in node_registry:
            if node["id"] == node_id:
                return node
        
        raise HTTPException(status_code=404, detail=f"Node with id '{node_id}' not found")
        
    except AttributeError:
        raise HTTPException(status_code=500, detail="Node registry not initialized in app.state")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node by id from state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_nodes_from_state(
    request: Request,
    query: str = None,
    tags: str = None,
    category: str = None,
    function: str = None
):
    """
    app.state에서 노드를 검색합니다.
    """
    try:
        node_registry = request.app.state.node_registry
        filtered_nodes = node_registry.copy()
        
        # query로 description과 nodeName 검색
        if query:
            query_lower = query.lower()
            filtered_nodes = [
                node for node in filtered_nodes
                if query_lower in node.get("description", "").lower() or 
                   query_lower in node.get("nodeName", "").lower()
            ]
        
        # tags로 필터링
        if tags:
            search_tags = [tag.strip().lower() for tag in tags.split(",")]
            filtered_nodes = [
                node for node in filtered_nodes
                if any(
                    search_tag in [node_tag.lower() for node_tag in node.get("tags", [])]
                    for search_tag in search_tags
                )
            ]
        
        # category로 필터링
        if category:
            filtered_nodes = [
                node for node in filtered_nodes
                if node.get("categoryId", "").lower() == category.lower()
            ]
        
        # function으로 필터링
        if function:
            filtered_nodes = [
                node for node in filtered_nodes
                if node.get("functionId", "").lower() == function.lower()
            ]
        
        return filtered_nodes
        
    except AttributeError:
        raise HTTPException(status_code=500, detail="Node registry not initialized in app.state")
    except Exception as e:
        logger.error(f"Error searching nodes from state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/parameters/categorized/{node_id}", response_model=Dict[str, Any])
async def get_categorized_parameters_from_state(node_id: str, request: Request):
    """
    app.state에서 특정 노드의 파라미터를 기본/고급으로 분류하여 반환합니다.
    """
    try:
        node_registry = request.app.state.node_registry
        
        for node in node_registry:
            if node["id"] == node_id:
                parameters = node.get("parameters", [])
                
                basic_params = []
                advanced_params = []
                
                for param in parameters:
                    if param.get("optional", False):
                        advanced_params.append(param)
                    else:
                        basic_params.append(param)
                
                return {
                    "node_id": node_id,
                    "node_name": node.get("nodeName", ""),
                    "description": node.get("description", ""),
                    "basic_parameters": basic_params,
                    "advanced_parameters": advanced_params,
                    "has_advanced": len(advanced_params) > 0,
                    "source": "app.state"
                }
        
        raise HTTPException(status_code=404, detail=f"Node with id '{node_id}' not found")
        
    except AttributeError:
        raise HTTPException(status_code=500, detail="Node registry not initialized in app.state")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting categorized parameters from state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/refresh")
async def refresh_node_registry(request: Request):
    """
    노드 레지스트리를 다시 로드하여 app.state에 업데이트합니다.
    """
    try:
        from editor.node_composer import run_discovery, generate_json_spec, get_node_registry
        
        # 노드 재발견
        logger.info("Refreshing node registry...")
        run_discovery()
        generate_json_spec("constants/exported_nodes.json")
        
        # app.state 업데이트
        request.app.state.node_registry = get_node_registry()
        request.app.state.node_count = len(request.app.state.node_registry)
        
        return {
            "status": "success",
            "message": "Node registry refreshed successfully",
            "node_count": request.app.state.node_count,
            "timestamp": "refreshed"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing node registry: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
