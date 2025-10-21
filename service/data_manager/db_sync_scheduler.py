"""
DB Auto-Sync Scheduler
외부 DB에서 주기적으로 데이터를 가져와 동기화하는 스케줄러
"""
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# ⚠️ 순환 import 방지: 지연 import로 변경
# from service.data_manager.encryption_helper import get_encryption_helper
from service.database.models.db_sync_config import DBSyncConfig

logger = logging.getLogger("db-sync-scheduler")


class DBSyncScheduler:
    """DB 자동 동기화 스케줄러"""
    
    def __init__(self, data_manager_registry, app_db_manager):
        """
        Args:
            data_manager_registry: DataManager 레지스트리
            app_db_manager: 애플리케이션 DB 매니저
        """
        self.registry = data_manager_registry
        self.app_db = app_db_manager
        self.scheduler = BackgroundScheduler()
        self.sync_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._encryption = None  # 지연 초기화
        
        logger.info("✅ DBSyncScheduler 초기화 완료")
    
    @property
    def encryption(self):
        """암호화 헬퍼 지연 로딩 (순환 import 방지)"""
        if self._encryption is None:
            from service.data_manager.encryption_helper import get_encryption_helper
            self._encryption = get_encryption_helper()
        return self._encryption

    # ==================== 라이프사이클 관리 ====================
    
    def start(self) -> None:
        """스케줄러 시작 및 기존 설정 복원"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("🚀 DBSyncScheduler 시작됨")
            self._restore_sync_configs()

    def stop(self) -> None:
        """스케줄러 중지"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("⏹️  DBSyncScheduler 중지됨")

    # ==================== CRUD 작업 ====================
    
    def add_db_sync(
        self,
        manager_id: str,
        user_id: str,
        db_config: Dict[str, Any],
        sync_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        DB 자동 동기화 설정 추가 또는 업데이트
        
        Args:
            manager_id: DataManager ID
            user_id: 사용자 ID
            db_config: DB 연결 정보 (host, port, database, username, password 등)
            sync_config: 스케줄 설정 (schedule_type, interval_minutes, cron_expression 등)
            
        Returns:
            성공 여부 및 메시지를 포함한 딕셔너리
        """
        try:
            # Manager 존재 확인
            manager = self.registry.get_manager(manager_id, user_id)
            if not manager:
                raise ValueError(f"Manager '{manager_id}'를 찾을 수 없습니다")
            
            # 기존 설정 확인
            existing_configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            is_update = bool(existing_configs)
            config_model = existing_configs[0] if is_update else DBSyncConfig()
            
            # 기본 정보 설정
            config_model.manager_id = manager_id
            config_model.user_id = user_id
            
            # DB 연결 정보 저장
            config_model.db_type = db_config.get('db_type', 'postgresql')
            config_model.db_host = db_config.get('host')
            config_model.db_port = db_config.get('port')
            config_model.db_name = db_config.get('database')
            config_model.db_username = db_config.get('username')
            
            # 비밀번호 암호화
            if db_config.get('password'):
                config_model.db_password = self.encryption.encrypt(db_config['password'])
            
            # 스케줄 설정
            config_model.enabled = sync_config.get('enabled', True)
            config_model.schedule_type = sync_config['schedule_type']
            config_model.interval_minutes = sync_config.get('interval_minutes')
            config_model.cron_expression = sync_config.get('cron_expression')
            
            # 쿼리/테이블 설정
            config_model.query = sync_config.get('query')
            config_model.table_name = sync_config.get('table_name')
            config_model.schema_name = sync_config.get('schema_name')
            config_model.chunk_size = sync_config.get('chunk_size')
            
            # 옵션 설정
            config_model.detect_changes = sync_config.get('detect_changes', True)
            config_model.notification_enabled = sync_config.get('notification_enabled', False)
            
            # DB 저장
            if is_update:
                self.app_db.update(config_model)
                logger.info(f"✅ DB 동기화 설정 업데이트: {manager_id}")
            else:
                config_model.sync_count = 0
                self.app_db.insert(config_model)
                logger.info(f"✅ DB 동기화 설정 추가: {manager_id}")
            
            # 메모리에 로드 및 스케줄 등록
            self._load_config_to_memory(config_model)
            
            return {
                'success': True,
                'manager_id': manager_id,
                'message': 'DB 자동 동기화가 설정되었습니다'
            }
            
        except Exception as e:
            logger.error(f"❌ DB 동기화 추가 실패: {e}", exc_info=True)
            raise

    def remove_db_sync(self, manager_id: str, user_id: str) -> bool:
        """DB 자동 동기화 설정 제거"""
        try:
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                logger.warning(f"제거할 동기화 설정이 없습니다: {manager_id}")
                return False
            
            # DB에서 삭제
            self.app_db.delete(DBSyncConfig, configs[0].id)
            
            # 메모리 및 스케줄러에서 제거
            with self._lock:
                if manager_id in self.sync_configs:
                    try:
                        self.scheduler.remove_job(self.sync_configs[manager_id]['sync_id'])
                    except Exception as e:
                        logger.warning(f"스케줄 제거 실패: {e}")
                    del self.sync_configs[manager_id]
            
            logger.info(f"✅ DB 동기화 제거: {manager_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ DB 동기화 제거 실패: {e}", exc_info=True)
            return False

    # ==================== 상태 제어 ====================
    
    def pause_db_sync(self, manager_id: str, user_id: str) -> bool:
        """동기화 일시 중지"""
        try:
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                return False
            
            config = configs[0]
            config.enabled = False
            self.app_db.update(config)
            
            with self._lock:
                if manager_id in self.sync_configs:
                    self.scheduler.pause_job(self.sync_configs[manager_id]['sync_id'])
                    self.sync_configs[manager_id]['enabled'] = False
            
            logger.info(f"⏸️  DB 동기화 일시 중지: {manager_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 동기화 일시 중지 실패: {e}", exc_info=True)
            return False

    def resume_db_sync(self, manager_id: str, user_id: str) -> bool:
        """동기화 재개"""
        try:
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                return False
            
            config = configs[0]
            config.enabled = True
            self.app_db.update(config)
            
            with self._lock:
                if manager_id in self.sync_configs:
                    self.scheduler.resume_job(self.sync_configs[manager_id]['sync_id'])
                    self.sync_configs[manager_id]['enabled'] = True
            
            logger.info(f"▶️  DB 동기화 재개: {manager_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 동기화 재개 실패: {e}", exc_info=True)
            return False

    # ==================== 조회 ====================
    
    def get_sync_status(self, manager_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """동기화 상태 조회 (MLflow 정보 포함)"""
        try:
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                return None
            
            config = configs[0]
            
            # 다음 실행 시간 조회
            next_run_time = None
            if manager_id in self.sync_configs:
                try:
                    sync_id = self.sync_configs[manager_id]['sync_id']
                    job = self.scheduler.get_job(sync_id)
                    if job and job.next_run_time:
                        next_run_time = job.next_run_time.isoformat()
                except Exception as e:
                    logger.warning(f"다음 실행 시간 조회 실패: {e}")
            
            return {
                'sync_id': f"sync_{manager_id}",
                'manager_id': manager_id,
                'enabled': config.is_active(),
                'db_type': config.db_type,
                'db_host': config.db_host or 'N/A',
                'db_name': config.db_name,
                'db_username': config.db_username or 'N/A',
                'schedule_type': config.schedule_type,
                'schedule_description': config.get_schedule_description(),
                'interval_minutes': config.interval_minutes,
                'cron_expression': config.cron_expression,
                'query': config.query,
                'table_name': config.table_name,
                'schema_name': config.schema_name,
                'chunk_size': config.chunk_size,
                'detect_changes': config.detect_changes,
                'last_sync_info': config.get_last_sync_info(),
                'last_sync': config.last_sync_at,
                'last_sync_status': config.last_sync_status,
                'last_error': config.last_error,
                'sync_count': config.sync_count,
                'next_run_time': next_run_time,
                'created_at': config.created_at,
                'updated_at': config.updated_at,
                'mlflow_info': config.get_mlflow_info(),
            }
            
        except Exception as e:
            logger.error(f"❌ 상태 조회 실패: {e}", exc_info=True)
            return None

    def list_all_syncs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """모든 동기화 설정 목록 조회"""
        try:
            if user_id:
                configs = self.app_db.find_by_condition(
                    DBSyncConfig,
                    {'user_id': user_id},
                    limit=1000
                )
            else:
                configs = self.app_db.find_all(DBSyncConfig, limit=1000)
            
            result = []
            for config in configs:
                status = self.get_sync_status(config.manager_id, config.user_id)
                if status:
                    result.append(status)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 목록 조회 실패: {e}", exc_info=True)
            return []

    # ==================== 수동 실행 ====================
    
    def trigger_manual_sync(self, manager_id: str, user_id: str) -> Dict[str, Any]:
        """수동 동기화 즉시 실행"""
        with self._lock:
            if manager_id not in self.sync_configs:
                raise ValueError("동기화 설정을 찾을 수 없습니다")
            
            config = self.sync_configs[manager_id]
            if config['user_id'] != user_id:
                raise ValueError("권한이 없습니다")
        
        logger.info(f"🔧 수동 동기화 트리거: {manager_id}")
        result = self._execute_sync(config)
        
        return {
            'success': result['status'] in ['success', 'no_changes'],
            'message': '수동 동기화가 완료되었습니다',
            'sync_result': result
        }

    # ==================== 내부 메서드 ====================
    
    def _restore_sync_configs(self) -> None:
        """DB에서 동기화 설정 복원 및 Manager 자동 로드"""
        try:
            all_configs = self.app_db.find_all(DBSyncConfig, limit=1000)
            
            restored_count = 0
            manager_load_count = 0
            
            for config in all_configs:
                try:
                    # Manager가 메모리에 없으면 Redis/MinIO에서 복원 시도
                    manager = self.registry.get_manager(config.manager_id, config.user_id)
                    if not manager:
                        logger.info(
                            f"🔄 Manager 자동 복원 시도: "
                            f"{config.manager_id} (user: {config.user_id})"
                        )
                        
                        # Manager 복원 시도
                        try:
                            manager = self.registry.load_manager(
                                config.manager_id,
                                config.user_id
                            )
                            if manager:
                                logger.info(f"  ✅ Manager 복원 성공: {config.manager_id}")
                                manager_load_count += 1
                            else:
                                logger.warning(
                                    f"  ⚠️  Manager 복원 실패: {config.manager_id} - "
                                    f"Redis/MinIO에 데이터 없음"
                                )
                                # 복원 실패한 경우 스케줄을 비활성화
                                config.enabled = False
                                self.app_db.update(config)
                                logger.info(f"  ⏸️  동기화 자동 비활성화: {config.manager_id}")
                                continue
                        except Exception as load_error:
                            logger.error(
                                f"  ❌ Manager 복원 중 에러: {config.manager_id} - {load_error}"
                            )
                            # 복원 실패한 경우 스케줄을 비활성화
                            config.enabled = False
                            self.app_db.update(config)
                            continue
                    
                    # 설정을 메모리에 로드
                    self._load_config_to_memory(config)
                    restored_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  설정 복원 실패: {config.manager_id} - {e}")
            
            logger.info(
                f"✅ 동기화 설정 복원 완료: "
                f"설정 {restored_count}개, Manager 복원 {manager_load_count}개"
            )
            
        except Exception as e:
            logger.error(f"❌ 동기화 설정 복원 실패: {e}", exc_info=True)

    def _load_config_to_memory(self, config: DBSyncConfig) -> None:
        """DB 모델을 메모리에 로드하고 스케줄러에 등록"""
        
        # 비밀번호 복호화
        decrypted_password = None
        if config.db_password:
            try:
                decrypted_password = self.encryption.decrypt(config.db_password)
            except Exception as e:
                logger.error(f"❌ 비밀번호 복호화 실패: {e}")
        
        # 메모리 설정 구성
        memory_config = {
            'sync_id': f"sync_{config.manager_id}",
            'manager_id': config.manager_id,
            'user_id': config.user_id,
            'enabled': config.enabled,
            'db_config': {
                'db_type': config.db_type,
                'host': config.db_host,
                'port': config.db_port,
                'database': config.db_name,
                'username': config.db_username,
                'password': decrypted_password
            },
            'schedule_type': config.schedule_type,
            'interval_minutes': config.interval_minutes,
            'cron_expression': config.cron_expression,
            'query': config.query,
            'table_name': config.table_name,
            'schema_name': config.schema_name,
            'chunk_size': config.chunk_size,
            'detect_changes': config.detect_changes,
            'last_checksum': config.last_checksum,
            'mlflow_enabled': config.mlflow_enabled
        }
        
        with self._lock:
            self.sync_configs[config.manager_id] = memory_config
        
        # 활성화된 경우 스케줄러에 등록
        if config.enabled:
            self._add_scheduler_job(memory_config)

    def _add_scheduler_job(self, config: Dict[str, Any]) -> None:
        """스케줄러에 작업 추가"""
        sync_id = config['sync_id']
        
        # 기존 작업 제거
        try:
            self.scheduler.remove_job(sync_id)
        except Exception:
            pass
        
        # 트리거 생성
        if config['schedule_type'] == 'interval':
            trigger = IntervalTrigger(minutes=config.get('interval_minutes', 60))
        elif config['schedule_type'] == 'cron':
            cron_expr = config.get('cron_expression', '0 * * * *')
            parts = cron_expr.split()
            trigger = CronTrigger(
                minute=parts[0] if len(parts) > 0 else '*',
                hour=parts[1] if len(parts) > 1 else '*',
                day=parts[2] if len(parts) > 2 else '*',
                month=parts[3] if len(parts) > 3 else '*',
                day_of_week=parts[4] if len(parts) > 4 else '*'
            )
        else:
            raise ValueError(f"지원되지 않는 schedule_type: {config['schedule_type']}")
        
        # 작업 등록
        self.scheduler.add_job(
            func=self._execute_sync,
            trigger=trigger,
            args=[config],
            id=sync_id,
            name=f"DB Sync: {config['manager_id']}",
            replace_existing=True
        )
        
        logger.info(f"📅 스케줄 등록: {sync_id}")

    def _execute_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """동기화 실행 (MLflow 자동 업로드 포함)"""
        manager_id = config['manager_id']
        user_id = config['user_id']
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 DB 동기화 시작: manager={manager_id}, user={user_id}")
            
            # Manager 가져오기 (자동 복원 포함)
            manager = self.registry.get_manager(manager_id, user_id)
            
            if not manager:
                logger.warning(f"  ⚠️  Manager 메모리에 없음, 복원 시도...")
                
                # get_manager가 내부적으로 Redis/MinIO에서 복원을 시도하지만 실패한 경우
                # 가능한 원인:
                # 1. Redis에 manager:xxx:owner 키가 없음
                # 2. Redis에 manager:xxx:dataset 매핑이 없음
                # 3. MinIO에 데이터가 없음
                
                # 에러 메시지를 더 명확하게 작성
                raise ValueError(
                    f"Manager '{manager_id}'를 복원할 수 없습니다.\n"
                    f"가능한 원인:\n"
                    f"  1. Redis에 Manager 소유자 정보 없음\n"
                    f"  2. Manager-Dataset 매핑 정보 손실\n"
                    f"  3. 복원 스크립트를 실행해주세요: python fix_existing_managers.py\n"
                    f"  (user_id: {user_id})"
                )
            
            # DB에서 데이터 로드
            result = manager.db_load_dataset(
                db_config=config['db_config'],
                query=config.get('query'),
                table_name=config.get('table_name'),
                chunk_size=config.get('chunk_size')
            )
            
            # 변경 감지
            changes_detected = True
            if config.get('detect_changes', True):
                current_checksum = result.get('source_info', {}).get('checksum')
                last_checksum = config.get('last_checksum')
                
                if last_checksum and current_checksum == last_checksum:
                    logger.info(f"  └─ ℹ️  데이터 변경 없음")
                    changes_detected = False
                    
                    # DB 업데이트 (변경 없음)
                    self._update_sync_status(
                        manager_id=manager_id,
                        status='no_changes',
                        checksum=current_checksum,
                        start_time=start_time
                    )
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    return {
                        'status': 'no_changes',
                        'message': '데이터 변경이 없습니다',
                        'duration_seconds': duration,
                        'num_rows': result.get('num_rows', 0),
                    }
                
                # 체크섬 업데이트
                with self._lock:
                    self.sync_configs[manager_id]['last_checksum'] = current_checksum
            
            # MLflow 자동 업로드 (변경사항이 있을 때만)
            mlflow_result = None
            if changes_detected and config.get('mlflow_enabled', False):
                logger.info(f"  └─ 📊 MLflow 자동 업로드 시작...")
                mlflow_result = self._upload_to_mlflow(manager_id, user_id, config)
            
            # DB 업데이트 (성공)
            current_checksum = result.get('source_info', {}).get('checksum')
            self._update_sync_status(
                manager_id=manager_id,
                status='success',
                checksum=current_checksum or config.get('last_checksum'),
                start_time=start_time
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ DB 동기화 완료: {duration:.2f}초")
            
            response = {
                'status': 'success',
                'message': 'DB 동기화 완료',
                'duration_seconds': duration,
                'num_rows': result.get('num_rows', 0),
                'num_columns': result.get('num_columns', 0),
                'changes_detected': changes_detected,
            }
            
            # MLflow 결과 추가
            if mlflow_result:
                response['mlflow_upload'] = mlflow_result
            
            return response
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            logger.error(f"❌ DB 동기화 실패: {e}", exc_info=True)
            
            # DB 업데이트 (실패)
            self._update_sync_status(
                manager_id=manager_id,
                status='failed',
                error=error_message,
                start_time=start_time
            )
            
            return {
                'status': 'failed',
                'message': f'동기화 실패: {error_message}',
                'error': error_message[:500],
                'duration_seconds': duration,
            }

    def _update_sync_status(
        self,
        manager_id: str,
        status: str,
        checksum: Optional[str] = None,
        error: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> None:
        """동기화 상태를 DB에 업데이트"""
        try:
            db_configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            if not db_configs:
                logger.warning(f"⚠️  DB Config를 찾을 수 없음: {manager_id}")
                return
            
            db_config = db_configs[0]
            
            # 공통 업데이트
            if start_time:
                db_config.last_sync_at = start_time.isoformat()
            else:
                db_config.last_sync_at = datetime.now().isoformat()
            
            db_config.last_sync_status = status
            
            # 상태별 처리
            if status == 'success':
                db_config.sync_count = (db_config.sync_count or 0) + 1
                if checksum:
                    db_config.last_checksum = checksum
                db_config.last_error = None
                
            elif status == 'failed':
                if error:
                    db_config.last_error = error[:500]
                    
            elif status == 'no_changes':
                # 변경 없음은 sync_count를 증가시키지 않음
                if checksum:
                    db_config.last_checksum = checksum
                db_config.last_error = None
            
            self.app_db.update(db_config)
            logger.debug(f"✅ DB 상태 업데이트 완료: {manager_id} - {status}")
            
        except Exception as e:
            logger.error(f"❌ DB 상태 업데이트 실패: {e}", exc_info=True)

    def _upload_to_mlflow(
        self,
        manager_id: str,
        user_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """MLflow에 데이터셋 자동 업로드"""
        try:
            # DB에서 최신 설정 가져오기
            db_configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            if not db_configs:
                raise ValueError("DB Sync Config를 찾을 수 없습니다")
            
            db_config_model = db_configs[0]
            
            # MLflow 설정 확인
            if not db_config_model.mlflow_enabled:
                return {'skipped': True, 'reason': 'MLflow 비활성화'}
            
            if not db_config_model.mlflow_experiment_name:
                raise ValueError("MLflow 실험 이름이 설정되지 않았습니다")
            
            # Manager 가져오기
            manager = self.registry.get_manager(manager_id, user_id)
            if not manager:
                raise ValueError(f"Manager '{manager_id}'를 찾을 수 없습니다")
            
            # 데이터셋 이름 생성
            dataset_name = db_config_model.get_next_mlflow_dataset_name()
            experiment_name = db_config_model.mlflow_experiment_name
            
            logger.info(
                f"  └─ 📊 MLflow 업로드: "
                f"experiment={experiment_name}, dataset={dataset_name}"
            )
            
            # MLflow 업로드 실행
            upload_result = manager.upload_to_mlflow(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                mlflow_tracking_uri=db_config_model.mlflow_tracking_uri,
                artifact_path='dataset',
                description=f'Auto-synced from DB (sync #{db_config_model.sync_count + 1})',
                tags={
                    'source': 'db_auto_sync',
                    'manager_id': manager_id,
                    'sync_count': str(db_config_model.sync_count + 1),
                    'db_type': config.get('db_config', {}).get('db_type', 'unknown')
                },
                format='parquet'
            )
            
            if upload_result.get('success'):
                mlflow_info = upload_result.get('mlflow_info', {})
                run_id = mlflow_info.get('run_id')
                
                # 성공 기록
                self._update_mlflow_status(
                    manager_id=manager_id,
                    success=True,
                    run_id=run_id
                )
                
                logger.info(f"  └─ ✅ MLflow 업로드 완료: run_id={run_id}")
                
                return {
                    'success': True,
                    'dataset_name': dataset_name,
                    'experiment_name': experiment_name,
                    'run_id': run_id,
                    'upload_count': db_config_model.mlflow_upload_count
                }
            else:
                error_msg = upload_result.get('message', 'Unknown error')
                raise Exception(error_msg)
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"  └─ ❌ MLflow 업로드 실패: {e}")
            
            # 실패 기록
            self._update_mlflow_status(
                manager_id=manager_id,
                success=False,
                error=error_message
            )
            
            return {
                'success': False,
                'error': error_message[:500]
            }

    def _update_mlflow_status(
        self,
        manager_id: str,
        success: bool,
        run_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """MLflow 업로드 상태를 DB에 업데이트"""
        try:
            db_configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            if not db_configs:
                logger.warning(f"⚠️  DB Config를 찾을 수 없음: {manager_id}")
                return
            
            db_config = db_configs[0]
            
            if success:
                db_config.mlflow_upload_count = (db_config.mlflow_upload_count or 0) + 1
                db_config.mlflow_last_run_id = run_id
                db_config.mlflow_last_upload_at = datetime.now().isoformat()
                db_config.mlflow_last_error = None
            else:
                if error:
                    db_config.mlflow_last_error = error[:500]
            
            self.app_db.update(db_config)
            logger.debug(f"✅ MLflow 상태 업데이트 완료: {manager_id}")
            
        except Exception as e:
            logger.error(f"❌ MLflow 상태 업데이트 실패: {e}", exc_info=True)