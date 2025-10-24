from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import uuid
import json
import pyarrow.csv as pv
from pathlib import Path
from contextlib import asynccontextmanager
from controller.node.nodeApiController import register_node_api_routes

from controller.node.router import node_router
from controller.admin.router import admin_router
from controller.workflow.router import workflow_router
from controller.workflow.endpoints.auto_generation import router as auto_generation_router
from controller.rag.router import rag_router
from controller.audio.router import audio_router
from controller.data_manager.router import data_manager_router
from controller.model.router import model_router
from controller.tools.router import tool_router

from controller.trainController import router as trainRouter
from controller.llmController import router as llmRouter
from controller.performanceController import router as performanceRouter
from controller.interactionController import router as interactionRouter
from controller.huggingface.huggingfaceController import router as huggingfaceRouter
from controller.appController import router as appRouter
from controller.authController import router as authRouter
from controller.vast_proxy_controller import router as vastProxyRouter
from controller.promptController import router as promptRouter
from controller.mcpController import router as mcpRouter
from editor.node_composer import run_discovery, generate_json_spec, get_node_registry
from editor.async_workflow_executor import execution_manager
from config.config_composer import config_composer
from service.database.models import APPLICATION_MODELS
from service.database.models.prompts import Prompts
from service.database.models.workflow import WorkflowStoreMeta

from service.database import AppDatabaseManager
from service.embedding.embedding_factory import EmbeddingFactory
from service.stt.stt_factory import STTFactory
from service.tts.tts_factory import TTSFactory
from service.guarder.guarder_factory import GuarderFactory
from service.vast.proxy_client import VastProxyClient
from service.vector_db.vector_manager import VectorManager
from service.retrieval.document_processor.document_processor import DocumentProcessor
from service.retrieval.document_info_generator.document_info_generator import DocumentInfoGenerator
from service.data_manager.data_manager_register import DataManagerRegistry
from service.mlflow.mlflow_artifact_service import MLflowArtifactService
from service.sync.workflow_deploy_sync import sync_workflow_deploy_meta
from controller.helper.utils.workflow_helpers import workflow_data_synchronizer

def print_xgen_logo():
    logo = """
    ██╗  ██╗ ██████╗ ███████╗███╗   ██╗
    ╚██╗██╔╝██╔════╝ ██╔════╝████╗  ██║
     ╚███╔╝ ██║  ███╗█████╗  ██╔██╗ ██║
     ██╔██╗ ██║   ██║██╔══╝  ██║╚██╗██║
    ██╔╝ ██╗╚██████╔╝███████╗██║ ╚████║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝

    🚀 PlateeRAG Backend with XGEN Engine 🚀
    """
    print(logo)

def print_step_banner(step_num, title, description=""):
    """각 단계별 배너 출력"""
    banner = f"""
    ┌{'─' * 60}┐
    │  STEP {step_num}: {title:<50}│
    {f'│  {description:<58}│' if description else ''}
    └{'─' * 60}┘
    """
    print(banner)

def generate_prompt_uid(act_text: str) -> str:
    """act 텍스트로부터 prompt_uid 생성"""
    # 공백을 _로 변경하고 소문자로 변환
    base_uid = act_text.replace(' ', '_').lower()
    # 특수문자 제거 (영문자, 숫자, _만 유지)
    base_uid = ''.join(c for c in base_uid if c.isalnum() or c == '_')
    # UUID 8자리 추가
    unique_suffix = str(uuid.uuid4())[:8]
    return f"{base_uid}_{unique_suffix}"

def load_prompts_from_csv(app_db, csv_path: str):
    """
    CSV 파일에서 프롬프트 데이터를 로드하여 데이터베이스에 저장
    표준 형식: ID, Prompt UID, Title, Content, Language, Public, Template, User ID, Username, Full Name, Metadata, Created At, Updated At
    """
    try:
        # CSV 파일 존재 확인
        if not Path(csv_path).exists():
            logger.warning(f"⚠️  Prompts CSV file not found: {csv_path}")
            return {"success": False, "error": "CSV file not found"}

        # 이미 템플릿 프롬프트가 존재하는지 확인
        existing_templates = app_db.find_by_condition(Prompts, {"is_template": True}, limit=1)
        if existing_templates and len(existing_templates) > 0:
            logger.info(f"⚠️  Template prompts already exist in database ({len(existing_templates)} records). Skipping CSV import.")
            return {"success": True, "inserted_count": 0, "skipped": True, "message": "Templates already exist"}

        # PyArrow로 CSV 파일 읽기
        table = pv.read_csv(csv_path)
        logger.info(f"📄 Loaded CSV with {len(table)} rows")

        # 데이터를 딕셔너리 리스트로 변환
        data = table.to_pydict()

        # 표준 형식 컬럼 확인 (다운로드 형식)
        required_columns = ['Prompt UID', 'Title', 'Content', 'Language', 'Public', 'Template']
        missing_columns = [col for col in required_columns if col not in data]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}. Expected standard format from download."
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

        inserted_count = 0
        skipped_count = 0

        # 각 행 처리
        for i in range(len(data['Prompt UID'])):
            try:
                prompt_uid = data['Prompt UID'][i]
                title = data['Title'][i]
                content = data['Content'][i]
                language = data['Language'][i]
                public_available = True
                is_template = True

                # None 또는 빈 문자열 체크
                if not prompt_uid or not title or not content or not language:
                    logger.warning(f"⚠️  Skipping row {i+1}: Missing required data")
                    skipped_count += 1
                    continue

                # Prompt 객체 생성
                new_prompt = Prompts(
                    user_id=None,
                    prompt_uid=str(prompt_uid).strip(),
                    prompt_title=str(title).strip(),
                    prompt_content=str(content).strip(),
                    public_available=public_available,
                    is_template=is_template,
                    language=str(language).strip(),
                    metadata=None
                )

                # 중복 체크 (prompt_uid 기준)
                existing = app_db.find_by_condition(
                    Prompts,
                        {"prompt_uid": new_prompt.prompt_uid},
                    limit=1
                )

                if not existing or len(existing) == 0:
                    result = app_db.insert(new_prompt)
                    if result and result.get("result") == "success":
                        inserted_count += 1
                        logger.debug(f"✅ Inserted prompt: {new_prompt.prompt_uid} ({new_prompt.language})")
                    else:
                        skipped_count += 1
                        logger.warning(f"⚠️  Failed to insert prompt {i+1}: {new_prompt.prompt_uid}")
                else:
                    skipped_count += 1
                    logger.debug(f"🔄 Skipped existing prompt: {new_prompt.prompt_uid}")

            except Exception as e:
                logger.warning(f"⚠️  Failed to process row {i+1}: {e}")
                skipped_count += 1
                continue

        logger.info(f"✅ Successfully processed {inserted_count + skipped_count} prompt records")
        logger.info(f"📝 Inserted: {inserted_count}, Skipped: {skipped_count}")
        return {"success": True, "inserted_count": inserted_count, "skipped_count": skipped_count}

    except Exception as e:
        error_msg = f"Error loading prompts from CSV: {e}"
        logger.error(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

def load_workflow_templates(app_db, templates_dir: str):
    """
    JSON 파일에서 워크플로우 스토어 템플릿을 로드하여 데이터베이스에 저장
    """
    try:
        # 디렉토리 존재 확인
        if not Path(templates_dir).exists():
            logger.warning(f"⚠️  Workflow templates directory not found: {templates_dir}")
            return {"success": False, "error": "Templates directory not found"}

        # 이미 템플릿이 존재하는지 확인
        existing_templates = app_db.find_by_condition(WorkflowStoreMeta, {"is_template": True}, limit=1)
        if existing_templates and len(existing_templates) > 0:
            logger.info(f"⚠️  Workflow templates already exist in database ({len(existing_templates)} records). Skipping template import.")
            return {"success": True, "inserted_count": 0, "skipped": True, "message": "Templates already exist"}

        # JSON 파일 목록 가져오기
        json_files = list(Path(templates_dir).glob("*.json"))
        if not json_files:
            logger.warning(f"⚠️  No JSON files found in: {templates_dir}")
            return {"success": False, "error": "No template files found"}

        logger.info(f"📄 Found {len(json_files)} workflow template files")

        inserted_count = 0
        skipped_count = 0

        for json_file in json_files:
            try:
                # JSON 파일 읽기 및 전처리
                with open(json_file, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)

                # JSON을 다시 파싱하여 유니코드 이스케이프 문자 정리
                # ensure_ascii=False로 한글 등이 제대로 저장되도록 함
                workflow_data = json.loads(
                    json.dumps(workflow_data, ensure_ascii=False)
                )

                # 필수 필드 확인
                workflow_id = workflow_data.get('workflow_id')
                workflow_name = workflow_data.get('workflow_name')

                if not workflow_id or not workflow_name:
                    logger.warning(f"⚠️  Skipping {json_file.name}: Missing workflow_id or workflow_name")
                    skipped_count += 1
                    continue

                # description 추출 (없으면 workflow_name 사용)
                description = workflow_data.get('description', workflow_name)

                # nodes와 edges 개수 계산
                nodes = workflow_data.get('nodes', [])
                edges = workflow_data.get('edges', [])
                node_count = len(nodes)
                edge_count = len(edges)

                # startnode와 endnode 존재 여부 확인
                has_startnode = any(
                    node.get('data', {}).get('functionId') == 'startnode'
                    for node in nodes
                )
                has_endnode = any(
                    node.get('data', {}).get('functionId') == 'endnode'
                    for node in nodes
                )

                # is_completed 판단 (startnode와 endnode가 모두 있고, 최소 1개 이상의 edge가 있으면)
                is_completed = has_startnode and has_endnode and edge_count > 0

                # tags 추출 (metadata에 있을 수 있음)
                tags = workflow_data.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]

                # WorkflowStoreMeta 객체 생성
                new_template = WorkflowStoreMeta(
                    user_id=None,  # 템플릿이므로 None
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    workflow_upload_name=workflow_name,
                    node_count=node_count,
                    edge_count=edge_count,
                    has_startnode=has_startnode,
                    has_endnode=has_endnode,
                    is_completed=is_completed,
                    metadata={},  # 추가 메타데이터가 필요하면 여기에
                    current_version=1.0,
                    latest_version=1.0,
                    is_template=True,
                    description=description,
                    tags=tags,
                    workflow_data=workflow_data  # 전체 JSON 데이터
                )

                # 중복 체크 (workflow_id 기준)
                existing = app_db.find_by_condition(
                    WorkflowStoreMeta,
                    {"workflow_id": workflow_id},
                    limit=1
                )

                if not existing or len(existing) == 0:
                    result = app_db.insert(new_template)
                    if result and result.get("result") == "success":
                        inserted_count += 1
                        logger.debug(f"✅ Inserted workflow template: {workflow_name}")
                    else:
                        skipped_count += 1
                        logger.warning(f"⚠️  Failed to insert template: {workflow_name}")
                else:
                    skipped_count += 1
                    logger.debug(f"🔄 Skipped existing template: {workflow_name}")

            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Failed to parse JSON file {json_file.name}: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"⚠️  Failed to process template {json_file.name}: {e}")
                skipped_count += 1
                continue

        logger.info(f"✅ Successfully processed {inserted_count + skipped_count} workflow template files")
        logger.info(f"📝 Inserted: {inserted_count}, Skipped: {skipped_count}")
        return {"success": True, "inserted_count": inserted_count, "skipped_count": skipped_count}

    except Exception as e:
        error_msg = f"Error loading workflow templates: {e}"
        logger.error(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    try:
        # ⚠️ 주의: 이 자동 복구는 긴급 상황용입니다.
        # 정상적으로는 Step 7.6의 자동 로드가 Manager를 복원합니다.
        # fix_existing_managers는 Redis 데이터 손상 시에만 필요합니다.
        try:
            from fix_existing_managers import recover_all_managers
            # recover_all_managers()  # 기본적으로 비활성화 (필요 시 주석 해제)
            logger.info("ℹ️  긴급 복구 스크립트는 비활성화되어 있습니다 (정상)")
        except Exception as e:
            logger.debug(f"긴급 복구 스크립트 로드 실패 (무시 가능): {e}")

        print_xgen_logo()
        logger.info("🌟 Starting XGEN application lifespan...")

        # 1. 데이터베이스 설정만 먼저 초기화
        print_step_banner(1, "DATABASE INITIALIZATION", "Initializing core database configuration")
        logger.info("⚙️  Step 1: Database configuration initialization starting...")

        database_config = config_composer.initialize_database_config_only()
        if not database_config:
            logger.error("❌ Failed to initialize database configuration")
            return
        logger.info("✅ Step 1: Database configuration initialized successfully!")

        # 2. 애플리케이션 데이터베이스 초기화 (모든 모델 테이블 생성)
        print_step_banner(2, "APPLICATION DATABASE SETUP", "Creating tables and running migrations")
        logger.info("⚙️  Step 2: Application database initialization starting...")

        app_db = AppDatabaseManager(database_config)
        app_db.register_models(APPLICATION_MODELS)

        if app_db.initialize_database():
            app.state.app_db = app_db
            logger.info("✅ Step 2: Application database initialized successfully!")

            # Run database migrations
            logger.info("🔄 Running database migrations...")
            if app_db.run_migrations():
                logger.info("✅ Database migrations completed successfully!")
            else:
                logger.warning("⚠️  Database migrations failed, but continuing startup")
        else:
            logger.error("❌ Failed to initialize application database")
            app.state.app_db = None
            return

        # 3. 나머지 설정들 초기화 (이제 DB 테이블이 존재함)
        print_step_banner(3, "CONFIGURATION SETUP", "Loading remaining system configurations")
        logger.info("⚙️  Step 3: System configuration initialization starting...")

        configs = config_composer.initialize_remaining_configs()
        app.state.config_composer = config_composer
        logger.info("✅ Step 3: System configurations loaded successfully!")

        # 4. RAG 서비스 초기화 (벡터 DB와 임베딩 제공자)
        print_step_banner(4, "RAG SERVICES INITIALIZATION", "Setting up vector DB and embedding services")
        try:
            print_step_banner(4.1, "EMBEDDING SERVICE SETUP", "Setting up embedding services")
            embedding_client = EmbeddingFactory.create_embedding_client(config_composer)
            app.state.embedding_client = embedding_client

            print_step_banner(4.2, "VECTOR SERVICE SETUP", "Setting up vector services")
            vector_manager = VectorManager(config_composer)
            app.state.vector_manager = vector_manager

            print_step_banner(4.3, "DOCUMENT PROCESSOR SETUP", "Setting up document processing services")
            document_processor = DocumentProcessor(config_composer)
            app.state.document_processor = document_processor

            print_step_banner(4.4, "DOCUMENT INFO GENERATOR SETUP", "Setting up document info generation services")
            document_info_generator = DocumentInfoGenerator(config_composer)
            app.state.document_info_generator = document_info_generator
            logger.info("✅ Step 4: RAG services components initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Step 4: Failed to initialize RAG services: {e}")
            # RAG 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
            app.state.rag_service = None
            app.state.vector_manager = None
            app.state.embedding_client = None
            app.state.document_processor = None

        # 5. STT 서비스 초기화
        if config_composer.get_config_by_name("IS_AVAILABLE_STT").value:
            print_step_banner(5, "STT SERVICE SETUP", "Setting up Speech-to-Text services")
            try:
                logger.info("⚙️  Step 5: STT service initialization starting...")
                stt_client = STTFactory.create_stt_client(config_composer)
                app.state.stt_service = stt_client
                logger.info("✅ Step 5: STT service initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Step 5: Failed to initialize STT service: {e}")
                # STT 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
                app.state.stt_service = None
        else:
            print_step_banner(5, "STT SERVICE SETUP", "STT service is disabled in configuration")
            app.state.stt_service = None

        # 5.5. TTS 서비스 초기화
        if config_composer.get_config_by_name("IS_AVAILABLE_TTS").value:
            print_step_banner(5.5, "TTS SERVICE SETUP", "Setting up Text-to-Speech services")
            try:
                logger.info("⚙️  Step 5.5: TTS service initialization starting...")
                tts_client = TTSFactory.create_tts_client(config_composer)
                app.state.tts_service = tts_client
                logger.info("✅ Step 5.5: TTS service initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Step 5.5: Failed to initialize TTS service: {e}")
                # TTS 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
                app.state.tts_service = None
        else:
            print_step_banner(5.5, "TTS SERVICE SETUP", "TTS service is disabled in configuration")
            app.state.tts_service = None

        # 5.7. Guarder 서비스 초기화
        if config_composer.get_config_by_name("IS_AVAILABLE_GUARDER").value:
            print_step_banner(5.7, "GUARDER SERVICE SETUP", "Setting up Text Moderation services")
            try:
                logger.info("⚙️  Step 5.7: Guarder service initialization starting...")
                guarder_client = GuarderFactory.create_guarder_client(config_composer)
                app.state.guarder_service = guarder_client
                logger.info("✅ Step 5.7: Guarder service initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Step 5.7: Failed to initialize Guarder service: {e}")
                # Guarder 서비스 초기화 실패 시에도 애플리케이션 시작은 계속
                app.state.guarder_service = None
        else:
            print_step_banner(5.7, "GUARDER SERVICE SETUP", "Guarder service is disabled in configuration")
            app.state.guarder_service = None

        # 6. Vast proxy client 생성
        print_step_banner(6, "VAST PROXY SETUP", "Initializing remote VastAI proxy")
        logger.info("⚙️  Step 6: VAST proxy client initialization starting...")
        vast_proxy_client = VastProxyClient(config_composer)
        app.state.vast_proxy_client = vast_proxy_client
        # 기존 의존성을 사용하는 코드 호환을 위해 동일 객체를 vast_service로도 노출
        app.state.vast_service = vast_proxy_client
        logger.info("✅ Step 6: VAST proxy client initialized successfully")

        # 7. 워크플로우 실행 매니저 초기화
        print_step_banner(7, "WORKFLOW MANAGER SETUP", "Setting up workflow execution engine")
        logger.info("⚙️  Step 7: Workflow execution manager initialization starting...")
        app.state.execution_manager = execution_manager
        logger.info("✅ Step 7: Workflow execution manager initialized successfully!")

        print_step_banner(7.5, "DATA MANAGER REGISTRY SETUP", "Setting up data manager registry")
        logger.info("⚙️  Step 7.5: Data manager registry initialization starting...")
        app.state.data_manager_registry = DataManagerRegistry(app_db_manager=app.state.app_db)
        logger.info("✅ Step 7.5: Data manager registry initialized successfully!")

        # ⭐ 7.6. 저장된 매니저 메타데이터 초기화
        print_step_banner(7.6, "MANAGER METADATA INIT", "Initializing manager metadata (Lazy Loading)")
        logger.info("⚙️  Step 7.6: Manager metadata initialization...")

        try:
            registry = app.state.data_manager_registry

            # 🧹 고아 Manager 자동 정리 (선택 사항)
            AUTO_CLEANUP_ORPHANED = os.getenv("AUTO_CLEANUP_ORPHANED_MANAGERS", "false").lower() == "true"

            if AUTO_CLEANUP_ORPHANED:
                logger.info("🧹 고아 Manager 자동 정리 시작...")
                orphaned_count = 0
                orphaned_manager_ids = []

                cursor = 0
                all_manager_ids = set()

                # 모든 Manager ID 수집
                while True:
                    cursor, keys = registry.redis_manager.redis_client.scan(
                        cursor=cursor,
                        match="manager:*:*",
                        count=100
                    )
                    for key in keys:
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        parts = key.split(':')
                        if len(parts) >= 3:
                            all_manager_ids.add(parts[1])
                    if cursor == 0:
                        break

                # 고아 Manager 찾아서 정리
                for manager_id in all_manager_ids:
                    owner = registry.redis_manager.get_manager_owner(manager_id)
                    if not owner:
                        # Owner 없으면 관련 키 전부 삭제
                        try:
                            patterns = [
                                f"manager:{manager_id}:owner",
                                f"manager:{manager_id}:dataset_id",
                                f"manager:{manager_id}:state",
                                f"manager:{manager_id}:created_at",
                                f"manager:{manager_id}:resource_stats",
                            ]
                            for pattern in patterns:
                                registry.redis_manager.redis_client.delete(pattern)
                            orphaned_manager_ids.append(manager_id)
                            orphaned_count += 1
                            logger.debug(f"  └─ 🗑️  고아 Manager 삭제: {manager_id}")
                        except Exception as e:
                            logger.warning(f"  └─ ⚠️  삭제 실패: {manager_id} - {e}")

                # DB Sync Config에서도 정리
                if orphaned_manager_ids:
                    try:
                        from service.database.models.db_sync_config import DBSyncConfig
                        deleted_db_configs = 0
                        for manager_id in orphaned_manager_ids:
                            configs = app.state.app_db.find_by_condition(
                                DBSyncConfig,
                                {'manager_id': manager_id},
                                limit=10
                            )
                            for config in configs:
                                app.state.app_db.delete(DBSyncConfig, config.id)
                                deleted_db_configs += 1

                        if deleted_db_configs > 0:
                            logger.info(f"  └─ 🗑️  DB Sync Config {deleted_db_configs}개 정리")
                    except Exception as e:
                        logger.warning(f"  └─ ⚠️  DB Sync Config 정리 실패: {e}")

                if orphaned_count > 0:
                    logger.info(f"  └─ 🧹 고아 Manager {orphaned_count}개 정리 완료")
                else:
                    logger.info(f"  └─ ✅ 고아 Manager 없음")

            # Redis에서 유효한 매니저 수 확인
            cursor = 0
            valid_manager_ids = set()

            while True:
                cursor, keys = registry.redis_manager.redis_client.scan(
                    cursor=cursor,
                    match="manager:*:owner",
                    count=100
                )

                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')

                    parts = key.split(':')
                    if len(parts) >= 3:
                        manager_id = parts[1]
                        valid_manager_ids.add(manager_id)

                if cursor == 0:
                    break

            logger.info(f"✅ Step 7.6: 발견된 매니저: {len(valid_manager_ids)}개")
            logger.info(f"  └─ 💡 Lazy Loading 전략: API 호출 시 자동 복원됩니다")
            logger.info(f"  └─ 자동 복원 경로: DataManagerRegistry.get_manager()")

        except Exception as e:
            logger.error(f"❌ Step 7.6: 메타데이터 초기화 실패: {e}", exc_info=True)
        # 7.7. MLflow artifact service initialization
        print_step_banner(7.7, "MLFLOW ARTIFACT SERVICE", "Integrating MLflow tracking and artifacts")
        mlflow_tracking_uri = os.getenv("MLFLOW_URL", "").strip()
        mlflow_default_experiment_id = os.getenv("MLFLOW_DEFAULT_EXPERIMENT_ID")
        mlflow_cache_dir = os.getenv("MLFLOW_CACHE_DIR")
        mlflow_token = os.getenv("MLFLOW_TRACKING_TOKEN")

        if mlflow_tracking_uri:
            try:
                logger.info("⚙️  Step 7.7: MLflow artifact service initialization starting...")
                mlflow_service = MLflowArtifactService(
                    tracking_uri=mlflow_tracking_uri,
                    default_experiment_id=mlflow_default_experiment_id,
                    cache_dir=mlflow_cache_dir,
                    tracking_token=mlflow_token,
                )
                app.state.mlflow_service = mlflow_service
                logger.info("✅ Step 7.7: MLflow artifact service initialized successfully!")
            except Exception as mlflow_error:
                app.state.mlflow_service = None
                logger.error(
                    "❌ Step 7.7: Failed to initialize MLflow service: %s",
                    mlflow_error,
                    exc_info=True,
                )
        else:
            app.state.mlflow_service = None
            logger.warning("⚠️  MLflow tracking URI not configured. MLflow integration is disabled.")

        # 7.8. DB Sync Scheduler 초기화
        print_step_banner(7.8, "DB SYNC SCHEDULER SETUP", "Setting up database synchronization scheduler")
        logger.info("⚙️  Step 7.8: DB Sync Scheduler initialization starting...")

        try:
            from controller.helper.singletonHelper import initialize_db_sync_scheduler

            # 스케줄러 초기화
            db_sync_scheduler = initialize_db_sync_scheduler(app.state)

            logger.info(f"✅ Step 7.8: DB Sync Scheduler initialized successfully!")
            logger.info(f"  └─ Scheduler running: {db_sync_scheduler.scheduler.running}")
            logger.info(f"  └─ Loaded sync configs: {len(db_sync_scheduler.sync_configs)}")

        except Exception as e:
            logger.error(f"❌ Step 7.8: Failed to initialize DB Sync Scheduler: {e}", exc_info=True)
            app.state.db_sync_scheduler = None
            logger.warning("⚠️  DB Sync Scheduler is disabled. Sync endpoints will not be available.")

        print_step_banner(8, "SYSTEM VALIDATION", "Validating configurations and directories")
        logger.info("⚙️  Step 8: System validation starting...")

        config_composer.ensure_directories()
        validation_result = config_composer.validate_critical_configs()
        if not validation_result["valid"]:
            for error in validation_result["errors"]:
                logger.error(f"❌ Configuration error: {error}")
        for warning in validation_result["warnings"]:
            logger.warning(f"⚠️  Configuration warning: {warning}")
        logger.info("✅ Step 8: System validation completed!")

        # 9. 워크플로우 데이터 동기화
        print_step_banner(9, "WORKFLOW DATA SYNC", "Synchronizing workflow filesystem and database")
        logger.info("⚙️  Step 9: Workflow data synchronization starting...")

        try:
            logger.info("🔄 Step 9: SKIPPED - Workflow data sync is currently disabled")
            # sync_result = await workflow_data_synchronizer(app.state.app_db)
            # if sync_result["success"]:
            #     logger.info(f"✅ Step 9: Workflow data sync completed successfully! "
            #                f"Added {sync_result['files_added_to_db']} to DB, "
            #                f"Created {sync_result['files_created_from_db']} files, "
            #                f"Removed {sync_result['orphaned_db_entries_removed']} orphaned entries, "
            #                f"Processed {sync_result['users_processed']} users")
            # else:
            #     logger.warning(f"⚠️  Step 9: Workflow data sync completed with issues. "
            #                  f"Errors: {sync_result.get('errors', [])}")
        except Exception as e:
            logger.error(f"❌ Step 9: Failed to sync workflow data: {e}")

        # 9.5. 워크플로우-배포 메타데이터 동기화
        print_step_banner(9.5, "WORKFLOW-DEPLOY SYNC", "Synchronizing workflow and deploy metadata")
        logger.info("⚙️  Step 9.5: Workflow-deploy metadata synchronization starting...")

        try:
            sync_result = sync_workflow_deploy_meta(app.state.app_db)
            if sync_result["success"]:
                logger.info(f"✅ Step 9.5: Workflow-deploy sync completed successfully! "
                           f"Created {sync_result['created_deploys']} new deploy entries from "
                           f"{sync_result['total_workflows']} total workflows")
            else:
                logger.warning(f"⚠️  Step 9.5: Workflow-deploy sync completed with issues. "
                             f"Errors: {sync_result.get('errors', [])}")
        except Exception as e:
            logger.error(f"❌ Step 9.5: Failed to sync workflow-deploy metadata: {e}")

        print_step_banner(10, "NODE DISCOVERY", "Discovering and registering XGEN nodes")
        logger.info("⚙️  Step 10: Node discovery starting...")

        run_discovery()
        registry_path = configs["node"].REGISTRY_FILE_PATH.value
        generate_json_spec(registry_path)
        app.state.node_registry = get_node_registry()
        app.state.node_count = len(app.state.node_registry)

        # 노드 API 라우트 등록
        logger.info("🔗 Registering node API routes...")
        register_node_api_routes()
        logger.info("✅ Node API routes registered successfully!")

        logger.info(f"✅ Step 10: Node discovery completed! Registered {app.state.node_count} nodes")

        # 11. 프롬프트 템플릿 데이터 로드
        print_step_banner(11, "PROMPT TEMPLATES LOADING", "Loading prompt templates from CSV to database")
        logger.info("⚙️  Step 11: Prompt templates loading starting...")

        try:
            constants_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants")
            prompts_csv_path = os.path.join(constants_dir, "prompts_processed.csv")
            load_result = load_prompts_from_csv(app.state.app_db, prompts_csv_path)

            if load_result["success"]:
                if load_result.get("skipped", False):
                    logger.info(f"✅ Step 11: Prompt templates already exist - skipped loading")
                else:
                    inserted = load_result.get("inserted_count", 0)
                    skipped = load_result.get("skipped_count", 0)
                    logger.info(f"✅ Step 11: Prompt templates loaded successfully! "
                               f"Inserted: {inserted}, Skipped: {skipped}")
            else:
                logger.warning(f"⚠️  Step 11: Prompt templates loading completed with issues. "
                             f"Error: {load_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"❌ Step 11: Failed to load prompt templates: {e}")

        # 12. 워크플로우 스토어 템플릿 로드
        print_step_banner(12, "WORKFLOW STORE TEMPLATES", "Loading workflow store templates from JSON files")
        logger.info("⚙️  Step 12: Workflow store templates loading starting...")

        try:
            constants_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants")
            templates_dir = os.path.join(constants_dir, "workflow_store_template")
            load_result = load_workflow_templates(app.state.app_db, templates_dir)

            if load_result["success"]:
                if load_result.get("skipped", False):
                    logger.info(f"✅ Step 12: Workflow store templates already exist - skipped loading")
                else:
                    inserted = load_result.get("inserted_count", 0)
                    skipped = load_result.get("skipped_count", 0)
                    logger.info(f"✅ Step 12: Workflow store templates loaded successfully! "
                               f"Inserted: {inserted}, Skipped: {skipped}")
            else:
                logger.warning(f"⚠️  Step 12: Workflow store templates loading completed with issues. "
                             f"Error: {load_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"❌ Step 12: Failed to load workflow store templates: {e}")

        print_step_banner("FINAL", "XGEN STARTUP COMPLETE", "All systems operational! 🎉")
        logger.info("🎉 XGEN application startup complete! Ready to serve requests.")

        # SSE 세션 관리자 자동 정리 태스크 시작
        print_step_banner("SSE", "SSE SESSION MANAGER", "Starting SSE session cleanup task")
        logger.info("⚙️  SSE Session Manager: Starting cleanup task...")
        try:
            from service.retrieval.sse_session_manager import sse_session_manager
            sse_session_manager.start_cleanup_task(interval_minutes=10)
            logger.info("✅ SSE Session Manager: Cleanup task started (10 min interval)")
        except Exception as e:
            logger.error(f"❌ SSE Session Manager: Failed to start cleanup task: {e}")

    except Exception as e:
        logger.error(f"💥 Error during startup: {e}")
        logger.info("🔄 Application will continue despite startup error")

    yield

    logger.info("🛑 XGEN application shutdown...")
    print("""
    ┌─────────────────────────────────────────────────────┐
    │                XGEN SHUTDOWN                        │
    │           Gracefully stopping all services         │
    └─────────────────────────────────────────────────────┘
    """)
    try:
        # SSE 세션 관리자 정리
        try:
            from service.retrieval.sse_session_manager import sse_session_manager
            logger.info("🔄 Stopping SSE session manager cleanup task...")
            sse_session_manager.stop_cleanup_task()
            logger.info("✅ SSE session manager cleanup task stopped")
        except Exception as e:
            logger.error(f"❌ Failed to stop SSE session manager: {e}")

        if hasattr(app.state, 'db_sync_scheduler') and app.state.db_sync_scheduler:
            logger.info("🔄 Shutting down DB Sync Scheduler...")
            from controller.helper.singletonHelper import shutdown_db_sync_scheduler
            shutdown_db_sync_scheduler(app.state)
            logger.info("✅ DB Sync Scheduler shutdown complete")
        # Data Manager Registry 정리
        if hasattr(app.state, 'data_manager_registry') and app.state.data_manager_registry:
            logger.info("🔄 Cleaning up data manager registry...")
            app.state.data_manager_registry.cleanup()
            logger.info("✅ Data manager registry cleanup complete")

        # 워크플로우 실행 매니저 정리
        if hasattr(app.state, 'execution_manager') and app.state.execution_manager:
            logger.info("🔄 Shutting down workflow execution manager...")
            app.state.execution_manager.shutdown()
            logger.info("✅ Workflow execution manager shutdown complete")

        # STT 서비스 정리
        if hasattr(app.state, 'stt_service') and app.state.stt_service:
            logger.info("🔄 Cleaning up STT service...")
            await app.state.stt_service.cleanup()
            logger.info("✅ STT service cleanup complete")

        # TTS 서비스 정리
        if hasattr(app.state, 'tts_service') and app.state.tts_service:
            logger.info("🔄 Cleaning up TTS service...")
            await app.state.tts_service.cleanup()
            logger.info("✅ TTS service cleanup complete")

        # Guarder 서비스 정리
        if hasattr(app.state, 'guarder_service') and app.state.guarder_service:
            logger.info("🔄 Cleaning up Guarder service...")
            await app.state.guarder_service.cleanup()
            logger.info("✅ Guarder service cleanup complete")

        # 애플리케이션 데이터베이스 정리
        if hasattr(app.state, 'app_db') and app.state.app_db:
            app.state.app_db.close()
            logger.info("✅ Application database connection closed")

        config_composer.save_all()
        logger.info("✅ Configurations saved on shutdown")
        logger.info("👋 XGEN shutdown complete. Goodbye!")
    except Exception as e:
        logger.error(f"💥 Error during shutdown: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("plateerag-backend")

app = FastAPI(
    title="PlateeRAG Backend",
    description="API for training models with customizable parameters - Enhanced with app.state",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 시작시 기본값, config 로드 후 업데이트 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(node_router)
app.include_router(admin_router)
app.include_router(workflow_router)
app.include_router(auto_generation_router)
app.include_router(rag_router)
app.include_router(audio_router)
app.include_router(data_manager_router)
app.include_router(model_router)
app.include_router(tool_router)

app.include_router(authRouter)
app.include_router(llmRouter)
app.include_router(performanceRouter)
app.include_router(trainRouter)
app.include_router(interactionRouter)
app.include_router(appRouter)
app.include_router(vastProxyRouter)
app.include_router(huggingfaceRouter)
app.include_router(promptRouter)
app.include_router(mcpRouter)

if __name__ == "__main__":
    try:
        host = os.environ.get("APP_HOST", "0.0.0.0")
        port = int(os.environ.get("APP_PORT", "8000"))
        debug = os.environ.get("DEBUG_MODE", "false").lower() in ('true', '1', 'yes', 'on')

        print(f"Starting server on {host}:{port} (debug={debug})")
        if debug:
            uvicorn.run("main:app", host=host, port=port, reload=True)
        else:
            uvicorn.run("main:app", host=host, port=port, reload=False)
    except Exception as e:
        logger.warning(f"Failed to load config for uvicorn: {e}")
        logger.info("Using default values for uvicorn")
        uvicorn.run("main:app", host="0.0.0.0", port=10, reload=False)
