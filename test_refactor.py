#!/usr/bin/env python3
"""
Test script to verify the refactored database models and ExecutionMeta service
"""
import sys
sys.path.append('/plateerag_backend')

try:
    # Test import of the new service
    from service.database.execution_meta_service import get_or_create_execution_meta, update_execution_meta_count
    print("✓ Successfully imported execution_meta_service functions")
    
    # Test import of updated controllers
    from controller.workflowController import router as workflow_router
    print("✓ Successfully imported workflowController")
    
    from controller.interactionController import router as interaction_router
    print("✓ Successfully imported interactionController")
    
    from controller.chatController import router as chat_router
    print("✓ Successfully imported chatController")
    
    from controller.performanceController import router as performance_router
    print("✓ Successfully imported performanceController")
    
    # Test database models imports
    from service.database.models.executor import ExecutionMeta, ExecutionIO
    print("✓ Successfully imported ExecutionMeta and ExecutionIO models")
    
    from service.database.models.performance import NodePerformance, WorkflowExecution
    print("✓ Successfully imported performance models")
    
    from service.database.models.user import User, UserSession
    print("✓ Successfully imported user models")
    
    from service.database.connection import AppDatabaseManager
    print("✓ Successfully imported AppDatabaseManager")
    
    from service.database import AppDatabaseManager as AppDBManager
    print("✓ Successfully imported AppDatabaseManager from package")
    
    # Test main.py imports
    from main import app
    print("✓ Successfully imported FastAPI app from main.py")
    
    print("\n✅ All imports successful! The database folder move is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Error details: {e.__class__.__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print(f"Error details: {e.__class__.__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
