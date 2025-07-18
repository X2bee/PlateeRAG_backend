"""
ExecutionMeta 관련 공통 서비스 함수들
"""
import json
import logging
from service.database.models.executor import ExecutionMeta

logger = logging.getLogger("execution-meta-service")


async def get_or_create_execution_meta(
    db_manager, 
    interaction_id: str, 
    workflow_id: str, 
    workflow_name: str, 
    first_msg: str = None
) -> ExecutionMeta:
    """
    ExecutionMeta를 조회하거나 새로 생성합니다.
    
    Args:
        db_manager: 데이터베이스 매니저
        interaction_id: 상호작용 ID
        workflow_id: 워크플로우 ID
        workflow_name: 워크플로우 이름
        first_msg: 초기 메시지 (선택적)
        
    Returns:
        ExecutionMeta 객체
    """
    try:
        # 기존 ExecutionMeta 조회
        query = """
        SELECT * FROM execution_meta 
        WHERE interaction_id = %s AND workflow_id = %s
        """
        
        # SQLite인 경우 파라미터 플레이스홀더 변경
        if db_manager.config_db_manager.db_type == "sqlite":
            query = query.replace("%s", "?")
        
        result = db_manager.config_db_manager.execute_query(query, (interaction_id, workflow_id))
        
        if result and len(result) > 0:
            # 기존 데이터가 있으면 반환
            row = result[0]
            execution_meta = ExecutionMeta(
                id=row.get('id'),
                interaction_id=row['interaction_id'],
                workflow_id=row['workflow_id'],
                workflow_name=row['workflow_name'],
                interaction_count=row['interaction_count'],
                metadata=json.loads(row['metadata']) if row.get('metadata') else {}
            )
            logger.info("Found existing ExecutionMeta for interaction_id: %s, workflow_id: %s", interaction_id, workflow_id)
            return execution_meta
        else:
            # 새로 생성
            metadata = {}
            if first_msg:
                metadata["first_message"] = first_msg
                # 채팅 모드인 경우 추가 메타데이터 포함
                if "chat" in first_msg.lower() or "Chat" in first_msg:
                    metadata["chat_mode"] = "default_mode"
                else:
                    metadata["placeholder"] = first_msg
            
            execution_meta = ExecutionMeta(
                interaction_id=interaction_id,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                interaction_count=0,
                metadata=metadata
            )
            
            # DB에 저장
            insert_query = """
            INSERT INTO execution_meta (interaction_id, workflow_id, workflow_name, interaction_count, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            
            if db_manager.config_db_manager.db_type == "sqlite":
                insert_query = insert_query.replace("%s", "?")
            
            metadata_json = json.dumps(execution_meta.metadata, ensure_ascii=False)
            db_manager.config_db_manager.execute_query(
                insert_query, 
                (interaction_id, workflow_id, workflow_name, 0, metadata_json)
            )
            
            logger.info("Created new ExecutionMeta for interaction_id: %s, workflow_id: %s", interaction_id, workflow_id)
            return execution_meta
            
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Error handling ExecutionMeta: %s", str(e))
        # 실패해도 기본값으로 계속 진행
        return ExecutionMeta(
            interaction_id=interaction_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            interaction_count=0,
            metadata={}
        )


async def update_execution_meta_count(db_manager, execution_meta: ExecutionMeta):
    """
    ExecutionMeta의 interaction_count를 1 증가시킵니다.
    
    Args:
        db_manager: 데이터베이스 매니저
        execution_meta: ExecutionMeta 객체
    """
    try:
        update_query = """
        UPDATE execution_meta 
        SET interaction_count = interaction_count + 1, updated_at = CURRENT_TIMESTAMP
        WHERE interaction_id = %s AND workflow_id = %s
        """
        
        if db_manager.config_db_manager.db_type == "sqlite":
            update_query = update_query.replace("%s", "?")
        
        db_manager.config_db_manager.execute_query(
            update_query, 
            (execution_meta.interaction_id, execution_meta.workflow_id)
        )
        
        logger.info("Updated interaction_count for interaction_id: %s, workflow_id: %s", execution_meta.interaction_id, execution_meta.workflow_id)
        
    except (KeyError, ValueError) as e:
        logger.error("Error updating ExecutionMeta count: %s", str(e))
        # 카운트 업데이트 실패해도 전체 실행은 계속 진행
