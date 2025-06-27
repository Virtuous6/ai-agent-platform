#!/usr/bin/env python3
"""
Migrate Runbooks to Supabase

Script to migrate existing YAML runbooks from filesystem to Supabase database.
Preserves all runbook definitions, triggers, and metadata.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from runbooks.manager import initialize_runbook_manager, get_runbook_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RunbookMigrator:
    """Handles migration of YAML runbooks to Supabase."""
    
    def __init__(self):
        self.runbook_manager = None
        self.source_dir = project_root / "runbooks" / "active"
        self.migrated_count = 0
        self.failed_count = 0
        self.errors = []
    
    async def initialize(self):
        """Initialize the migration environment."""
        logger.info("🚀 Initializing Runbook Migration to Supabase...")
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Missing Supabase credentials!")
            logger.error("   Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
            return False
        
        # Initialize runbook manager
        success = await initialize_runbook_manager(supabase_url, supabase_key)
        if not success:
            logger.error("❌ Failed to initialize runbook manager")
            return False
        
        self.runbook_manager = await get_runbook_manager()
        logger.info("✅ Runbook manager initialized")
        return True
    
    async def discover_yaml_runbooks(self) -> List[Path]:
        """Discover all YAML runbooks in the active directory."""
        logger.info(f"🔍 Discovering YAML runbooks in {self.source_dir}...")
        
        if not self.source_dir.exists():
            logger.warning(f"⚠️  Source directory does not exist: {self.source_dir}")
            return []
        
        yaml_files = []
        for file_path in self.source_dir.glob("*.yaml"):
            if file_path.is_file():
                yaml_files.append(file_path)
                logger.info(f"   📄 Found: {file_path.name}")
        
        logger.info(f"📊 Discovered {len(yaml_files)} YAML runbooks")
        return yaml_files
    
    async def check_existing_runbooks(self) -> List[str]:
        """Check what runbooks already exist in Supabase."""
        logger.info("🔍 Checking existing runbooks in Supabase...")
        
        existing_runbooks = await self.runbook_manager.list_runbooks()
        existing_names = [rb.name for rb in existing_runbooks]
        
        if existing_names:
            logger.info(f"📊 Found {len(existing_names)} existing runbooks:")
            for name in existing_names:
                logger.info(f"   📋 {name}")
        else:
            logger.info("📊 No existing runbooks found")
        
        return existing_names
    
    async def migrate_runbook(self, yaml_path: Path, force: bool = False) -> bool:
        """Migrate a single YAML runbook to Supabase."""
        logger.info(f"📤 Migrating: {yaml_path.name}")
        
        try:
            # Check if runbook already exists
            runbook_name = yaml_path.stem  # filename without extension
            existing = await self.runbook_manager.get_runbook(runbook_name)
            
            if existing and not force:
                logger.warning(f"   ⚠️  Runbook '{runbook_name}' already exists - skipping")
                logger.info(f"      Use --force to overwrite existing runbooks")
                return True
            
            # Migrate the runbook
            success = await self.runbook_manager.migrate_yaml_runbook(str(yaml_path))
            
            if success:
                logger.info(f"   ✅ Successfully migrated: {runbook_name}")
                self.migrated_count += 1
                return True
            else:
                error_msg = f"Failed to migrate {runbook_name}"
                logger.error(f"   ❌ {error_msg}")
                self.errors.append(error_msg)
                self.failed_count += 1
                return False
        
        except Exception as e:
            error_msg = f"Error migrating {yaml_path.name}: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            self.errors.append(error_msg)
            self.failed_count += 1
            return False
    
    async def migrate_all_runbooks(self, force: bool = False) -> bool:
        """Migrate all discovered YAML runbooks."""
        logger.info("🚀 Starting migration of all runbooks...")
        
        # Discover runbooks
        yaml_files = await self.discover_yaml_runbooks()
        if not yaml_files:
            logger.warning("⚠️  No YAML runbooks found to migrate")
            return True
        
        # Check existing runbooks
        await self.check_existing_runbooks()
        
        # Migrate each runbook
        for yaml_path in yaml_files:
            await self.migrate_runbook(yaml_path, force)
        
        # Report results
        self.print_migration_summary()
        return self.failed_count == 0
    
    async def verify_migration(self) -> bool:
        """Verify that migrated runbooks are accessible in Supabase."""
        logger.info("🔍 Verifying migration results...")
        
        # Get all runbooks from Supabase
        runbooks = await self.runbook_manager.list_runbooks()
        
        if not runbooks:
            logger.error("❌ No runbooks found in Supabase after migration")
            return False
        
        logger.info(f"✅ Found {len(runbooks)} runbooks in Supabase:")
        for runbook in runbooks:
            logger.info(f"   📋 {runbook.name} (v{runbook.version}) - {runbook.category}")
            logger.info(f"      Triggers: {len(runbook.triggers or [])}")
            logger.info(f"      Usage: {runbook.usage_count}, Success: {runbook.success_rate:.1%}")
        
        return True
    
    async def get_analytics(self) -> None:
        """Display runbook analytics from Supabase."""
        logger.info("📊 Retrieving runbook analytics...")
        
        analytics = await self.runbook_manager.get_analytics()
        
        if analytics:
            logger.info(f"📈 Analytics Summary:")
            logger.info(f"   Total Runbooks: {analytics['total_runbooks']}")
            logger.info(f"   Average Success Rate: {analytics['performance_summary']['avg_success_rate']:.1%}")
            logger.info(f"   Total Usage: {analytics['performance_summary']['total_usage']}")
            
            if analytics['performance_summary']['most_used']:
                logger.info(f"   Most Used: {analytics['performance_summary']['most_used']}")
            
            logger.info(f"   Categories: {list(analytics['categories'].keys())}")
        else:
            logger.warning("⚠️  No analytics available")
    
    def print_migration_summary(self):
        """Print a summary of the migration results."""
        logger.info("=" * 60)
        logger.info("🎯 MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully migrated: {self.migrated_count} runbooks")
        logger.info(f"❌ Failed migrations: {self.failed_count} runbooks")
        
        if self.errors:
            logger.info("\n❌ Migration Errors:")
            for error in self.errors:
                logger.error(f"   • {error}")
        
        if self.migrated_count > 0:
            logger.info(f"\n🎉 Migration completed! {self.migrated_count} runbooks now in Supabase")
        
        logger.info("=" * 60)

async def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate YAML runbooks to Supabase")
    parser.add_argument("--force", action="store_true", 
                       help="Force overwrite existing runbooks")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing runbooks, don't migrate")
    parser.add_argument("--analytics", action="store_true",
                       help="Show runbook analytics")
    args = parser.parse_args()
    
    migrator = RunbookMigrator()
    
    # Initialize
    if not await migrator.initialize():
        sys.exit(1)
    
    try:
        if args.verify_only:
            # Only verify existing runbooks
            success = await migrator.verify_migration()
        elif args.analytics:
            # Show analytics
            await migrator.get_analytics()
            success = True
        else:
            # Full migration
            success = await migrator.migrate_all_runbooks(force=args.force)
            
            # Verify after migration
            if success:
                await migrator.verify_migration()
                await migrator.get_analytics()
        
        if success:
            logger.info("🎉 Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Migration completed with errors")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 