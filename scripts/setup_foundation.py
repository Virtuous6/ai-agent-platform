#!/usr/bin/env python3
"""
Foundation Setup Script

Sets up the LangGraph foundation including database schema, package structure,
and initial workflow loading. Validates the entire foundation implementation.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("📚 Checking dependencies...")
    
    missing_deps = []
    required_packages = [
        ('langgraph', 'LangGraph'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('pgvector', 'pgvector'),
        ('networkx', 'NetworkX'),
        ('tenacity', 'Tenacity')
    ]
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {display_name} available")
        except ImportError:
            logger.warning(f"   ❌ {display_name} not installed")
            missing_deps.append(package)
    
    # Check core dependencies
    core_packages = [
        ('langchain', 'LangChain'),
        ('langchain_openai', 'LangChain OpenAI'),
        ('supabase', 'Supabase'),
        ('yaml', 'PyYAML')
    ]
    
    for package, display_name in core_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {display_name} available")
        except ImportError:
            logger.error(f"   ❌ {display_name} MISSING (required)")
            missing_deps.append(package)
    
    return missing_deps

def verify_package_structure():
    """Verify that all required package directories and __init__.py files exist."""
    logger.info("📦 Verifying package structure...")
    
    required_structure = [
        'orchestrator/__init__.py',
        'orchestrator/langgraph/__init__.py',
        'orchestrator/langgraph/workflow_engine.py',
        'orchestrator/langgraph/runbook_converter.py',
        'orchestrator/langgraph/state_schemas.py',
        'memory/__init__.py',
        'tools/registry/__init__.py',
        'runbooks/engine/__init__.py',
        'database/migrations/001_vector_memory.sql'
    ]
    
    missing_files = []
    for file_path in required_structure:
        full_path = Path(project_root) / file_path
        if full_path.exists():
            logger.info(f"   ✅ {file_path}")
        else:
            logger.warning(f"   ❌ {file_path} MISSING")
            missing_files.append(file_path)
    
    return missing_files

def test_imports():
    """Test that all new packages can be imported correctly."""
    logger.info("🔗 Testing imports...")
    
    import_tests = [
        ('orchestrator', 'AgentOrchestrator'),
        ('orchestrator.langgraph.state_schemas', 'RunbookState'),
        ('orchestrator.langgraph.workflow_engine', 'LangGraphWorkflowEngine'),
        ('orchestrator.langgraph.runbook_converter', 'RunbookToGraphConverter')
    ]
    
    failed_imports = []
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"   ✅ {module_name}.{class_name}")
        except ImportError as e:
            logger.warning(f"   ❌ {module_name}.{class_name} - {str(e)}")
            failed_imports.append(f"{module_name}.{class_name}")
        except AttributeError as e:
            logger.warning(f"   ❌ {module_name}.{class_name} - {str(e)}")
            failed_imports.append(f"{module_name}.{class_name}")
    
    return failed_imports

def test_langgraph_integration():
    """Test LangGraph integration without requiring actual dependencies."""
    logger.info("🧠 Testing LangGraph integration...")
    
    try:
        from orchestrator.langgraph import LangGraphWorkflowEngine
        
        # Create engine with mock agents and tools
        mock_agents = {'general': None, 'technical': None, 'research': None}
        mock_tools = {'web_search': None}
        
        engine = LangGraphWorkflowEngine(mock_agents, mock_tools)
        
        # Test availability check
        is_available = engine.is_available()
        if is_available:
            logger.info("   ✅ LangGraph fully available")
        else:
            logger.info("   ⚠️  LangGraph in fallback mode (dependencies not installed)")
        
        # Test close method
        asyncio.create_task(engine.close())
        logger.info("   ✅ Engine lifecycle methods working")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ LangGraph integration failed: {str(e)}")
        return False

def test_orchestrator_integration():
    """Test that the orchestrator can use LangGraph integration."""
    logger.info("🎼 Testing orchestrator integration...")
    
    try:
        from orchestrator import AgentOrchestrator
        
        # Create orchestrator
        orchestrator = AgentOrchestrator()
        
        # Test that LangGraph method exists
        if hasattr(orchestrator, 'process_with_langgraph'):
            logger.info("   ✅ process_with_langgraph method available")
        else:
            logger.error("   ❌ process_with_langgraph method missing")
            return False
        
        # Test runbook selection method
        if hasattr(orchestrator, '_select_runbook_for_message'):
            logger.info("   ✅ runbook selection method available")
        else:
            logger.error("   ❌ runbook selection method missing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Orchestrator integration failed: {str(e)}")
        return False

def test_runbook_structure():
    """Test that the answer-question runbook can be loaded."""
    logger.info("📖 Testing runbook structure...")
    
    try:
        runbook_path = Path(project_root) / 'runbooks/active/answer-question.yaml'
        
        if not runbook_path.exists():
            logger.warning("   ⚠️  answer-question.yaml not found")
            return False
        
        import yaml
        with open(runbook_path, 'r') as file:
            runbook_data = yaml.safe_load(file)
        
        # Validate runbook structure
        required_sections = ['metadata', 'steps', 'outputs']
        for section in required_sections:
            if section in runbook_data:
                logger.info(f"   ✅ {section} section present")
            else:
                logger.warning(f"   ❌ {section} section missing")
                return False
        
        # Check for expected steps
        steps = runbook_data.get('steps', [])
        if len(steps) > 0:
            logger.info(f"   ✅ {len(steps)} steps defined")
        else:
            logger.warning("   ❌ No steps defined in runbook")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Runbook structure test failed: {str(e)}")
        return False

async def test_database_connection():
    """Test database connection and check if migration is needed."""
    logger.info("🗄️  Testing database connection...")
    
    try:
        from database.supabase_logger import SupabaseLogger
        
        # Try to create logger instance
        supabase_logger = SupabaseLogger()
        
        # Test basic connection
        health = supabase_logger.health_check()
        if health.get('status') == 'healthy':
            logger.info("   ✅ Database connection successful")
        else:
            logger.warning("   ⚠️  Database connection issues")
            return False
        
        # Check if vector tables exist (basic check)
        try:
            # This would fail if migration hasn't been run
            result = supabase_logger.client.table("runbook_executions").select("id").limit(1).execute()
            logger.info("   ✅ Migration tables appear to be available")
        except Exception:
            logger.warning("   ⚠️  Migration may not have been run yet")
            logger.info("   💡 Run the migration: database/migrations/001_vector_memory.sql")
        
        return True
        
    except Exception as e:
        logger.warning(f"   ⚠️  Database test failed: {str(e)}")
        logger.info("   💡 Check SUPABASE_URL and SUPABASE_KEY environment variables")
        return False

def create_missing_directories():
    """Create any missing directory structure."""
    logger.info("📁 Creating missing directories...")
    
    required_dirs = [
        'orchestrator/langgraph',
        'memory',
        'tools/registry', 
        'runbooks/engine',
        'database/migrations',
        'scripts'
    ]
    
    created_dirs = []
    for dir_path in required_dirs:
        full_path = Path(project_root) / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py if it's a Python package directory
            if not dir_path.startswith('scripts') and not dir_path.startswith('database'):
                init_file = full_path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
                    logger.info(f"   ✅ Created {init_file}")
            
            created_dirs.append(dir_path)
    
    if created_dirs:
        logger.info(f"   📁 Created directories: {', '.join(created_dirs)}")
    else:
        logger.info("   ✅ All directories already exist")

def generate_summary_report(missing_deps, missing_files, failed_imports, 
                          langgraph_ok, orchestrator_ok, runbook_ok, db_ok):
    """Generate a summary report of the foundation status."""
    logger.info("\n" + "="*60)
    logger.info("🚀 LANGGRAPH FOUNDATION SETUP SUMMARY")
    logger.info("="*60)
    
    # Overall status
    issues = len(missing_deps) + len(missing_files) + len(failed_imports)
    issues += sum([not langgraph_ok, not orchestrator_ok, not runbook_ok, not db_ok])
    
    if issues == 0:
        logger.info("🎉 FOUNDATION SETUP COMPLETE! All systems ready.")
        status = "READY"
    elif issues <= 3:
        logger.info("⚠️  FOUNDATION MOSTLY READY with minor issues.")
        status = "MOSTLY_READY"
    else:
        logger.info("❌ FOUNDATION NEEDS WORK before proceeding.")
        status = "NEEDS_WORK"
    
    # Detailed breakdown
    logger.info(f"\n📊 Status Breakdown:")
    logger.info(f"   Dependencies: {'✅' if not missing_deps else '❌'} ({len(missing_deps)} missing)")
    logger.info(f"   Package Structure: {'✅' if not missing_files else '❌'} ({len(missing_files)} missing)")
    logger.info(f"   Imports: {'✅' if not failed_imports else '❌'} ({len(failed_imports)} failed)")
    logger.info(f"   LangGraph Integration: {'✅' if langgraph_ok else '❌'}")
    logger.info(f"   Orchestrator Integration: {'✅' if orchestrator_ok else '❌'}")
    logger.info(f"   Runbook Structure: {'✅' if runbook_ok else '❌'}")
    logger.info(f"   Database Connection: {'✅' if db_ok else '⚠️ '}")
    
    # Next steps
    logger.info(f"\n📋 Next Steps:")
    
    if missing_deps:
        logger.info(f"   1. Install missing dependencies:")
        logger.info(f"      pip install {' '.join(missing_deps)}")
    
    if missing_files:
        logger.info(f"   2. Create missing files (or rerun this script)")
    
    if failed_imports:
        logger.info(f"   3. Fix import issues in: {', '.join(failed_imports)}")
    
    if not db_ok:
        logger.info(f"   4. Run database migration: database/migrations/001_vector_memory.sql")
    
    if status == "READY":
        logger.info(f"   🚀 START BUILDING! Try testing with answer-question.yaml")
        logger.info(f"   💡 Use orchestrator.process_with_langgraph() for workflow execution")
    
    logger.info(f"\n📖 Documentation:")
    logger.info(f"   - LangGraph integration: orchestrator/langgraph/README.md")
    logger.info(f"   - Memory system: memory/README.md") 
    logger.info(f"   - Tool registry: tools/registry/README.md")
    
    return status

async def main():
    """Run the complete foundation setup and validation."""
    logger.info("🚀 Starting LangGraph Foundation Setup...")
    logger.info("="*60)
    
    # Create missing directories first
    create_missing_directories()
    
    # Run all validation checks
    missing_deps = check_dependencies()
    missing_files = verify_package_structure()
    failed_imports = test_imports()
    
    langgraph_ok = test_langgraph_integration()
    orchestrator_ok = test_orchestrator_integration() 
    runbook_ok = test_runbook_structure()
    db_ok = await test_database_connection()
    
    # Generate final report
    status = generate_summary_report(
        missing_deps, missing_files, failed_imports,
        langgraph_ok, orchestrator_ok, runbook_ok, db_ok
    )
    
    # Return appropriate exit code
    if status == "READY":
        return 0
    elif status == "MOSTLY_READY":
        return 1  
    else:
        return 2

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n💥 Setup failed with error: {str(e)}")
        sys.exit(1) 