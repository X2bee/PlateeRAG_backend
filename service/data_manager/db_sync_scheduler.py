"""
DB Auto-Sync Scheduler
외부 DB에서 주기적으로 데이터를 가져옴
"""
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from service.data_manager.encryption_helper import get_encryption_helper
logger = logging.getLogger("db-sync-scheduler")

class DBSyncScheduler:
    """DB 자동 동기화 스케줄러"""
    def __init__(self, data_manager_registry, app_db_manager):
        self.registry = data_manager_registry
        self.app_db = app_db_manager
        self.scheduler = BackgroundScheduler()
        self.sync_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.encryption = get_encryption_helper()  # ✨ 암호화 헬퍼
        
        logger.info("✅ DBSyncScheduler 초기화 완료")

    def start(self):
        """스케줄러 시작"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("🚀 DBSyncScheduler 시작됨")
            self._restore_sync_configs()

    def stop(self):
        """스케줄러 중지"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("⏹️  DBSyncScheduler 중지됨")

    def add_db_sync(
        self,
        manager_id: str,
        user_id: str,
        db_config: Dict[str, Any],  # ✨ DB 연결 정보
        sync_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """DB 자동 동기화 추가"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
            # Manager 존재 확인
            manager = self.registry.get_manager(manager_id, user_id)
            if not manager:
                raise ValueError(f"Manager {manager_id}를 찾을 수 없습니다")
            
            # 기존 설정 확인
            existing = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            if existing:
                config_model = existing[0]
            else:
                config_model = DBSyncConfig()
                config_model.manager_id = manager_id
                config_model.user_id = user_id
            
            # ✨ DB 연결 정보 저장 (비밀번호 암호화)
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
            config_model.query = sync_config.get('query')
            config_model.table_name = sync_config.get('table_name')
            config_model.schema_name = sync_config.get('schema_name')
            config_model.chunk_size = sync_config.get('chunk_size')
            config_model.detect_changes = sync_config.get('detect_changes', True)
            config_model.notification_enabled = sync_config.get('notification_enabled', False)
            
            if existing:
                self.app_db.update(config_model)
                logger.info(f"✅ DB 동기화 설정 업데이트: {manager_id}")
            else:
                config_model.sync_count = 0
                self.app_db.insert(config_model)
                logger.info(f"✅ DB 동기화 설정 추가: {manager_id}")
            
            # 메모리에 로드 및 스케줄 추가
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
        """DB 자동 동기화 제거"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                return False
            
            self.app_db.delete(DBSyncConfig, configs[0].id)
            
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
            logger.error(f"❌ DB 동기화 제거 실패: {e}")
            return False

    def pause_db_sync(self, manager_id: str, user_id: str) -> bool:
        """동기화 일시 중지"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
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
            logger.error(f"동기화 일시 중지 실패: {e}")
            return False

    def resume_db_sync(self, manager_id: str, user_id: str) -> bool:
        """동기화 재개"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
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
            logger.error(f"동기화 재개 실패: {e}")
            return False

    def get_sync_status(self, manager_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """동기화 상태 조회"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
            configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id, 'user_id': user_id},
                limit=1
            )
            
            if not configs:
                return None
            
            config = configs[0]
            
            next_run_time = None
            if manager_id in self.sync_configs:
                try:
                    sync_id = self.sync_configs[manager_id]['sync_id']
                    job = self.scheduler.get_job(sync_id)
                    if job:
                        next_run_time = job.next_run_time.isoformat() if job.next_run_time else None
                except Exception:
                    pass
            
            return {
                'sync_id': f"sync_{manager_id}",
                'manager_id': manager_id,
                'enabled': config.enabled,
                'db_type': config.db_type,
                'db_host': config.db_host,
                'db_name': config.db_name,
                'schedule_type': config.schedule_type,
                'interval_minutes': config.interval_minutes,
                'cron_expression': config.cron_expression,
                'query': config.query,
                'table_name': config.table_name,
                'last_sync': config.last_sync_at,
                'last_sync_status': config.last_sync_status,
                'sync_count': config.sync_count,
                'next_run_time': next_run_time
            }
            
        except Exception as e:
            logger.error(f"상태 조회 실패: {e}")
            return None

    def list_all_syncs(self, user_id: str = None) -> List[Dict[str, Any]]:
        """모든 동기화 설정 목록"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
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
            logger.error(f"목록 조회 실패: {e}")
            return []

    def trigger_manual_sync(self, manager_id: str, user_id: str) -> Dict[str, Any]:
        """수동 동기화 실행"""
        with self._lock:
            if manager_id not in self.sync_configs:
                raise ValueError("동기화 설정을 찾을 수 없습니다")
            
            config = self.sync_configs[manager_id]
            if config['user_id'] != user_id:
                raise ValueError("권한이 없습니다")
        
        result = self._execute_sync(config)
        
        return {
            'success': True,
            'message': '수동 동기화가 완료되었습니다',
            'sync_result': result
        }

    # ========== 내부 메서드 ==========

    def _restore_sync_configs(self):
        """✨ DB에서 동기화 설정 복원 (비밀번호 복호화)"""
        try:
            from service.database.models.db_sync_config import DBSyncConfig
            
            all_configs = self.app_db.find_all(DBSyncConfig, limit=1000)
            
            restored_count = 0
            for config in all_configs:
                try:
                    self._load_config_to_memory(config)
                    restored_count += 1
                except Exception as e:
                    logger.warning(f"⚠️  설정 복원 실패: {config.manager_id}, {e}")
            
            logger.info(f"✅ {restored_count}개 동기화 설정 복원 완료")
            
        except Exception as e:
            logger.error(f"❌ 동기화 설정 복원 실패: {e}", exc_info=True)

    def _load_config_to_memory(self, config):
        """DB 모델을 메모리에 로드 (비밀번호 복호화)"""
        
        # ✨ 비밀번호 복호화
        decrypted_password = None
        if config.db_password:
            try:
                decrypted_password = self.encryption.decrypt(config.db_password)
            except Exception as e:
                logger.error(f"비밀번호 복호화 실패: {e}")
        
        memory_config = {
            'sync_id': f"sync_{config.manager_id}",
            'manager_id': config.manager_id,
            'user_id': config.user_id,
            'enabled': config.enabled,
            
            # DB 연결 정보
            'db_config': {
                'db_type': config.db_type,
                'host': config.db_host,
                'port': config.db_port,
                'database': config.db_name,
                'username': config.db_username,
                'password': decrypted_password  # 복호화된 비밀번호
            },
            
            # 스케줄
            'schedule_type': config.schedule_type,
            'interval_minutes': config.interval_minutes,
            'cron_expression': config.cron_expression,
            
            # 쿼리/테이블
            'query': config.query,
            'table_name': config.table_name,
            'schema_name': config.schema_name,
            'chunk_size': config.chunk_size,
            'detect_changes': config.detect_changes,
            'last_checksum': config.last_checksum
        }
        
        with self._lock:
            self.sync_configs[config.manager_id] = memory_config
        
        if config.enabled:
            self._add_scheduler_job(memory_config)

    def _add_scheduler_job(self, config: Dict[str, Any]):
        """스케줄러에 작업 추가"""
        sync_id = config['sync_id']
        
        try:
            self.scheduler.remove_job(sync_id)
        except Exception:
            pass
        
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
        
        self.scheduler.add_job(
            func=self._execute_sync,
            trigger=trigger,
            args=[config],
            id=sync_id,
            name=f"DB Sync: {config['manager_id']}",
            replace_existing=True
        )
        
        logger.info(f"📅 스케줄 추가: {sync_id}")

    def _execute_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """동기화 실행"""
        from service.database.models.db_sync_config import DBSyncConfig
        
        manager_id = config['manager_id']
        user_id = config['user_id']
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 DB 동기화 시작: manager={manager_id}")
            
            manager = self.registry.get_manager(manager_id, user_id)
            if not manager:
                raise ValueError(f"Manager {manager_id}를 찾을 수 없습니다")
            
            # ✨ 저장된 DB 설정 사용
            result = manager.db_load_dataset(
                db_config=config['db_config'],
                query=config.get('query'),
                table_name=config.get('table_name'),
                chunk_size=config.get('chunk_size')
            )
            
            # 변경 감지
            if config.get('detect_changes', True):
                current_checksum = result.get('source_info', {}).get('checksum')
                last_checksum = config.get('last_checksum')
                
                if last_checksum and current_checksum == last_checksum:
                    logger.info(f"  └─ ℹ️  데이터 변경 없음")
                    
                    db_configs = self.app_db.find_by_condition(
                        DBSyncConfig,
                        {'manager_id': manager_id},
                        limit=1
                    )
                    if db_configs:
                        db_config_model = db_configs[0]
                        db_config_model.last_sync_at = start_time.isoformat()
                        db_config_model.last_sync_status = 'no_changes'
                        self.app_db.update(db_config_model)
                    
                    return {'status': 'no_changes', 'message': '데이터 변경이 없습니다'}
                
                with self._lock:
                    self.sync_configs[manager_id]['last_checksum'] = current_checksum
            
            # DB 업데이트
            db_configs = self.app_db.find_by_condition(
                DBSyncConfig,
                {'manager_id': manager_id},
                limit=1
            )
            
            if db_configs:
                db_config_model = db_configs[0]
                db_config_model.last_sync_at = start_time.isoformat()
                db_config_model.last_sync_status = 'success'
                db_config_model.sync_count += 1
                db_config_model.last_checksum = config.get('last_checksum')
                db_config_model.last_error = None
                self.app_db.update(db_config_model)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ DB 동기화 완료: {duration:.2f}초")
            
            return {'status': 'success', 'message': 'DB 동기화 완료', 'duration_seconds': duration}
            
        except Exception as e:
            logger.error(f"❌ DB 동기화 실패: {e}", exc_info=True)
            
            try:
                db_configs = self.app_db.find_by_condition(
                    DBSyncConfig,
                    {'manager_id': manager_id},
                    limit=1
                )
                
                if db_configs:
                    db_config_model = db_configs[0]
                    db_config_model.last_sync_at = start_time.isoformat()
                    db_config_model.last_sync_status = 'failed'
                    db_config_model.last_error = str(e)[:500]
                    self.app_db.update(db_config_model)
            except Exception as update_error:
                logger.error(f"DB 업데이트 실패: {update_error}")
            
            return {'status': 'failed', 'message': f'동기화 실패: {str(e)}', 'error': str(e)}