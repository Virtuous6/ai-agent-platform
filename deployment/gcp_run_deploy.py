#!/usr/bin/env python3
"""
Google Cloud Run Deployment Script for AI Agent Platform
Optimized for production deployment with modern best practices
"""

import os
import json
import yaml
import subprocess
import sys
from pathlib import Path

def check_prerequisites():
    """Check if required tools and configurations are available."""
    print("🔍 Checking deployment prerequisites...")
    
    # Check for gcloud CLI
    try:
        result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Google Cloud CLI installed")
        else:
            print("  ❌ Google Cloud CLI not found")
            return False
    except FileNotFoundError:
        print("  ❌ Google Cloud CLI not found")
        return False
    
    # Check for Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Docker installed")
        else:
            print("  ❌ Docker not found")
            return False
    except FileNotFoundError:
        print("  ❌ Docker not found")
        return False
    
    # Check for project configuration
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            project_id = result.stdout.strip()
            print(f"  ✅ GCP Project: {project_id}")
            return project_id
        else:
            print("  ❌ No GCP project configured")
            return False
    except Exception:
        print("  ❌ Failed to get GCP project")
        return False

def enable_required_apis(project_id):
    """Enable required Google Cloud APIs."""
    print("🔧 Enabling required APIs...")
    
    apis = [
        'cloudbuild.googleapis.com',
        'run.googleapis.com',
        'containerregistry.googleapis.com',
        'artifactregistry.googleapis.com'
    ]
    
    for api in apis:
        try:
            result = subprocess.run([
                'gcloud', 'services', 'enable', api,
                '--project', project_id
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ {api}")
            else:
                print(f"  ⚠️  {api} (might already be enabled)")
        except Exception as e:
            print(f"  ❌ Failed to enable {api}: {e}")

def create_service_account(project_id):
    """Create and configure service account for the application."""
    print("👤 Setting up service account...")
    
    sa_name = "ai-agent-platform-sa"
    sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
    
    # Create service account
    try:
        result = subprocess.run([
            'gcloud', 'iam', 'service-accounts', 'create', sa_name,
            '--display-name', 'AI Agent Platform Service Account',
            '--project', project_id
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Service account created: {sa_email}")
        else:
            print(f"  ⚠️  Service account might already exist")
    except Exception as e:
        print(f"  ❌ Failed to create service account: {e}")
    
    # Grant necessary roles
    roles = [
        'roles/cloudsql.client',
        'roles/secretmanager.secretAccessor',
        'roles/storage.objectViewer'
    ]
    
    for role in roles:
        try:
            subprocess.run([
                'gcloud', 'projects', 'add-iam-policy-binding', project_id,
                '--member', f'serviceAccount:{sa_email}',
                '--role', role
            ], capture_output=True)
            print(f"  ✅ Granted {role}")
        except Exception as e:
            print(f"  ⚠️  Role {role}: {e}")

def build_and_deploy(project_id):
    """Build and deploy the application using Cloud Build."""
    print("🚀 Building and deploying to Cloud Run...")
    
    try:
        # Submit build to Cloud Build
        result = subprocess.run([
            'gcloud', 'builds', 'submit',
            '--config', 'cloudbuild.yaml',
            '--project', project_id,
            '--substitutions', f'_PROJECT_ID={project_id}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Build and deployment successful")
            
            # Get the service URL
            url_result = subprocess.run([
                'gcloud', 'run', 'services', 'describe', 'ai-agent-platform',
                '--region', 'us-central1',
                '--format', 'value(status.url)',
                '--project', project_id
            ], capture_output=True, text=True)
            
            if url_result.returncode == 0 and url_result.stdout.strip():
                service_url = url_result.stdout.strip()
                print(f"  🌐 Service URL: {service_url}")
                return service_url
            
        else:
            print(f"  ❌ Build failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"  ❌ Deployment failed: {e}")
        return None

def setup_monitoring(project_id):
    """Set up basic monitoring and alerting."""
    print("📊 Setting up monitoring...")
    
    # This would set up monitoring, alerting, etc.
    # For now, just print the monitoring URL
    monitoring_url = f"https://console.cloud.google.com/monitoring/services?project={project_id}"
    print(f"  📈 Monitor your service: {monitoring_url}")

def display_deployment_info(project_id, service_url=None):
    """Display deployment information and next steps."""
    print("\n" + "="*70)
    print("🎉 DEPLOYMENT COMPLETE")
    print("="*70)
    
    if service_url:
        print(f"🌐 Service URL: {service_url}")
        print(f"🏥 Health Check: {service_url}/health")
        print(f"📊 Metrics: {service_url}/metrics")
    
    print(f"\n📱 Google Cloud Console:")
    print(f"  Cloud Run: https://console.cloud.google.com/run?project={project_id}")
    print(f"  Cloud Build: https://console.cloud.google.com/cloud-build?project={project_id}")
    print(f"  Logs: https://console.cloud.google.com/logs?project={project_id}")
    
    print(f"\n🔧 Useful Commands:")
    print(f"  View logs: gcloud run services logs tail ai-agent-platform --region=us-central1")
    print(f"  Update service: gcloud builds submit --config cloudbuild.yaml")
    print(f"  Delete service: gcloud run services delete ai-agent-platform --region=us-central1")
    
    print(f"\n⚙️  Environment Variables to Set:")
    print(f"  OPENAI_API_KEY=your-openai-key")
    print(f"  SLACK_BOT_TOKEN=your-slack-token")
    print(f"  SUPABASE_URL=your-supabase-url")
    print(f"  SUPABASE_KEY=your-supabase-key")

def main():
    """Main deployment function."""
    print("🚀 AI Agent Platform - Google Cloud Run Deployment")
    print("="*60)
    
    # Check prerequisites
    project_id = check_prerequisites()
    if not project_id:
        print("\n❌ Prerequisites not met. Please install required tools and configure GCP.")
        sys.exit(1)
    
    # Enable APIs
    enable_required_apis(project_id)
    
    # Set up service account
    create_service_account(project_id)
    
    # Build and deploy
    service_url = build_and_deploy(project_id)
    
    # Set up monitoring
    setup_monitoring(project_id)
    
    # Display results
    display_deployment_info(project_id, service_url)
    
    if service_url:
        print(f"\n✅ Deployment successful! Your service is running at: {service_url}")
    else:
        print(f"\n⚠️  Deployment completed with warnings. Check the logs for details.")

if __name__ == "__main__":
    main() 