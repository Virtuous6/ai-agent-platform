"""
MCP Credential Store

Handles secure storage and retrieval of credentials for MCP connections.
Supports multiple storage backends: local .env, GCP Secret Manager, 
AWS Secrets Manager, and Supabase vault.
"""

import os
import logging
import json
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CredentialReference:
    """Reference to stored credentials."""
    storage_type: str  # 'env', 'gcp', 'aws', 'supabase'
    reference_key: str
    created_at: datetime
    expires_at: Optional[datetime] = None

class CredentialManager:
    """
    Manages secure storage and retrieval of MCP connection credentials.
    Automatically detects environment and uses appropriate storage backend.
    """
    
    def __init__(self):
        """Initialize credential manager with appropriate storage backend."""
        self.environment = self._detect_environment()
        self.storage_backend = self._init_storage_backend()
        
        logger.info(f"🔐 Credential Manager initialized for {self.environment} environment")
    
    def _detect_environment(self) -> str:
        """Detect the current deployment environment."""
        if os.path.exists('.env'):
            return 'local'
        elif os.getenv('GOOGLE_CLOUD_PROJECT'):
            return 'gcp'
        elif os.getenv('AWS_LAMBDA_FUNCTION_NAME') or os.getenv('AWS_REGION'):
            return 'aws'
        elif os.getenv('VERCEL') or os.getenv('NETLIFY'):
            return 'serverless'
        else:
            return 'hosted'
    
    def _init_storage_backend(self):
        """Initialize the appropriate storage backend."""
        if self.environment == 'local':
            return LocalEnvStorage()
        elif self.environment == 'gcp':
            return GCPSecretStorage()
        elif self.environment == 'aws':
            return AWSSecretStorage()
        else:
            return SupabaseVaultStorage()
    
    async def store_credential(self, connection_name: str, 
                             credentials: Dict[str, str]) -> str:
        """
        Store credentials securely and return reference.
        
        Args:
            connection_name: Unique name for the connection
            credentials: Dictionary of credential key-value pairs
            
        Returns:
            Reference string for credential retrieval
        """
        try:
            # Validate credentials
            self._validate_credentials(credentials)
            
            # Generate unique reference key
            reference_key = f"mcp_{connection_name}_{int(datetime.utcnow().timestamp())}"
            
            # Store using appropriate backend
            await self.storage_backend.store(reference_key, credentials)
            
            logger.info(f"✅ Stored credentials for connection: {connection_name}")
            
            return reference_key
            
        except Exception as e:
            logger.error(f"❌ Failed to store credentials for {connection_name}: {str(e)}")
            raise
    
    async def retrieve_credential(self, reference_key: str) -> Optional[Dict[str, str]]:
        """
        Retrieve credentials using reference key.
        
        Args:
            reference_key: Reference to stored credentials
            
        Returns:
            Dictionary of credentials or None if not found
        """
        try:
            credentials = await self.storage_backend.retrieve(reference_key)
            
            if credentials:
                logger.info(f"🔑 Retrieved credentials for reference: {reference_key}")
            else:
                logger.warning(f"⚠️ No credentials found for reference: {reference_key}")
            
            return credentials
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve credentials for {reference_key}: {str(e)}")
            return None
    
    async def delete_credential(self, reference_key: str) -> bool:
        """
        Delete stored credentials.
        
        Args:
            reference_key: Reference to credentials to delete
            
        Returns:
            True if successfully deleted
        """
        try:
            success = await self.storage_backend.delete(reference_key)
            
            if success:
                logger.info(f"🗑️ Deleted credentials for reference: {reference_key}")
            else:
                logger.warning(f"⚠️ Failed to delete credentials for reference: {reference_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting credentials for {reference_key}: {str(e)}")
            return False
    
    async def list_credentials(self, prefix: str = "mcp_") -> List[str]:
        """
        List all credential references with given prefix.
        
        Args:
            prefix: Prefix to filter references
            
        Returns:
            List of credential reference keys
        """
        try:
            references = await self.storage_backend.list_keys(prefix)
            logger.info(f"📋 Found {len(references)} credential references")
            return references
            
        except Exception as e:
            logger.error(f"❌ Failed to list credentials: {str(e)}")
            return []
    
    def _validate_credentials(self, credentials: Dict[str, str]):
        """Validate credential dictionary."""
        if not isinstance(credentials, dict):
            raise ValueError("Credentials must be a dictionary")
        
        if not credentials:
            raise ValueError("Credentials cannot be empty")
        
        # Check for suspicious values that might indicate injection attempts
        for key, value in credentials.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("All credential keys and values must be strings")
            
            if len(value) > 10000:  # Reasonable limit for credential values
                raise ValueError(f"Credential value for '{key}' is too long")


class LocalEnvStorage:
    """Local .env file storage for development."""
    
    def __init__(self):
        self.env_file = '.env'
        self.prefix = 'MCP_CRED_'
    
    async def store(self, reference_key: str, credentials: Dict[str, str]):
        """Store credentials in .env file."""
        try:
            # Read existing .env content
            env_content = []
            if os.path.exists(self.env_file):
                with open(self.env_file, 'r') as f:
                    env_content = f.readlines()
            
            # Encode credentials as JSON and base64 for .env storage
            credentials_json = json.dumps(credentials)
            encoded_credentials = base64.b64encode(credentials_json.encode()).decode()
            
            # Add new credential entry
            env_var = f"{self.prefix}{reference_key.upper()}"
            new_line = f"{env_var}={encoded_credentials}\n"
            
            # Remove existing entry if it exists
            env_content = [line for line in env_content if not line.startswith(f"{env_var}=")]
            env_content.append(new_line)
            
            # Write back to .env file
            with open(self.env_file, 'w') as f:
                f.writelines(env_content)
            
            logger.info(f"💾 Stored credential in .env: {env_var}")
            
        except Exception as e:
            logger.error(f"Failed to store credential in .env: {str(e)}")
            raise
    
    async def retrieve(self, reference_key: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials from .env file."""
        try:
            env_var = f"{self.prefix}{reference_key.upper()}"
            encoded_value = os.getenv(env_var)
            
            if not encoded_value:
                return None
            
            # Decode base64 and parse JSON
            credentials_json = base64.b64decode(encoded_value.encode()).decode()
            credentials = json.loads(credentials_json)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from .env: {str(e)}")
            return None
    
    async def delete(self, reference_key: str) -> bool:
        """Delete credential from .env file."""
        try:
            if not os.path.exists(self.env_file):
                return False
            
            env_var = f"{self.prefix}{reference_key.upper()}"
            
            # Read and filter out the credential
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            
            original_count = len(lines)
            filtered_lines = [line for line in lines if not line.startswith(f"{env_var}=")]
            
            # Write back if something was removed
            if len(filtered_lines) < original_count:
                with open(self.env_file, 'w') as f:
                    f.writelines(filtered_lines)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete credential from .env: {str(e)}")
            return False
    
    async def list_keys(self, prefix: str) -> List[str]:
        """List credential keys from .env file."""
        try:
            if not os.path.exists(self.env_file):
                return []
            
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            
            keys = []
            env_prefix = f"{self.prefix}{prefix.upper()}"
            
            for line in lines:
                if line.startswith(env_prefix):
                    # Extract the reference key from the environment variable name
                    env_var = line.split('=')[0]
                    reference_key = env_var.replace(self.prefix, '').lower()
                    keys.append(reference_key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list credential keys: {str(e)}")
            return []


class GCPSecretStorage:
    """Google Cloud Secret Manager storage."""
    
    def __init__(self):
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        
        try:
            from google.cloud import secretmanager
            self.client = secretmanager.SecretManagerServiceClient()
            self.parent = f"projects/{self.project_id}"
        except ImportError:
            raise ImportError("google-cloud-secret-manager package required for GCP storage")
    
    async def store(self, reference_key: str, credentials: Dict[str, str]):
        """Store credentials in GCP Secret Manager."""
        try:
            secret_id = f"mcp-credential-{reference_key}"
            secret_data = json.dumps(credentials).encode()
            
            # Create secret
            secret = {
                'replication': {'automatic': {}}
            }
            
            response = self.client.create_secret(
                request={
                    'parent': self.parent,
                    'secret_id': secret_id,
                    'secret': secret
                }
            )
            
            # Add secret version
            self.client.add_secret_version(
                request={
                    'parent': response.name,
                    'payload': {'data': secret_data}
                }
            )
            
            logger.info(f"💾 Stored credential in GCP Secret Manager: {secret_id}")
            
        except Exception as e:
            logger.error(f"Failed to store credential in GCP: {str(e)}")
            raise
    
    async def retrieve(self, reference_key: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials from GCP Secret Manager."""
        try:
            secret_id = f"mcp-credential-{reference_key}"
            name = f"{self.parent}/secrets/{secret_id}/versions/latest"
            
            response = self.client.access_secret_version(request={'name': name})
            secret_data = response.payload.data.decode()
            
            return json.loads(secret_data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from GCP: {str(e)}")
            return None
    
    async def delete(self, reference_key: str) -> bool:
        """Delete credential from GCP Secret Manager."""
        try:
            secret_id = f"mcp-credential-{reference_key}"
            name = f"{self.parent}/secrets/{secret_id}"
            
            self.client.delete_secret(request={'name': name})
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete credential from GCP: {str(e)}")
            return False
    
    async def list_keys(self, prefix: str) -> List[str]:
        """List credential keys from GCP Secret Manager."""
        try:
            secrets = self.client.list_secrets(request={'parent': self.parent})
            
            keys = []
            for secret in secrets:
                secret_name = secret.name.split('/')[-1]
                if secret_name.startswith(f"mcp-credential-{prefix}"):
                    reference_key = secret_name.replace("mcp-credential-", "")
                    keys.append(reference_key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list credential keys from GCP: {str(e)}")
            return []


class AWSSecretStorage:
    """AWS Secrets Manager storage."""
    
    def __init__(self):
        try:
            import boto3
            self.client = boto3.client('secretsmanager')
        except ImportError:
            raise ImportError("boto3 package required for AWS storage")
    
    async def store(self, reference_key: str, credentials: Dict[str, str]):
        """Store credentials in AWS Secrets Manager."""
        try:
            secret_name = f"mcp-credential-{reference_key}"
            secret_value = json.dumps(credentials)
            
            self.client.create_secret(
                Name=secret_name,
                SecretString=secret_value,
                Description=f"MCP connection credentials for {reference_key}"
            )
            
            logger.info(f"💾 Stored credential in AWS Secrets Manager: {secret_name}")
            
        except Exception as e:
            logger.error(f"Failed to store credential in AWS: {str(e)}")
            raise
    
    async def retrieve(self, reference_key: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials from AWS Secrets Manager."""
        try:
            secret_name = f"mcp-credential-{reference_key}"
            
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_data = response['SecretString']
            
            return json.loads(secret_data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from AWS: {str(e)}")
            return None
    
    async def delete(self, reference_key: str) -> bool:
        """Delete credential from AWS Secrets Manager."""
        try:
            secret_name = f"mcp-credential-{reference_key}"
            
            self.client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete credential from AWS: {str(e)}")
            return False
    
    async def list_keys(self, prefix: str) -> List[str]:
        """List credential keys from AWS Secrets Manager."""
        try:
            paginator = self.client.get_paginator('list_secrets')
            
            keys = []
            for page in paginator.paginate():
                for secret in page['SecretList']:
                    secret_name = secret['Name']
                    if secret_name.startswith(f"mcp-credential-{prefix}"):
                        reference_key = secret_name.replace("mcp-credential-", "")
                        keys.append(reference_key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list credential keys from AWS: {str(e)}")
            return []


class SupabaseVaultStorage:
    """Supabase-based credential storage as fallback."""
    
    def __init__(self):
        # This would be implemented to store encrypted credentials in Supabase
        # For now, fall back to local storage
        logger.warning("SupabaseVaultStorage not fully implemented, using local storage")
        self.fallback = LocalEnvStorage()
    
    async def store(self, reference_key: str, credentials: Dict[str, str]):
        """Store credentials in Supabase vault."""
        return await self.fallback.store(reference_key, credentials)
    
    async def retrieve(self, reference_key: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials from Supabase vault."""
        return await self.fallback.retrieve(reference_key)
    
    async def delete(self, reference_key: str) -> bool:
        """Delete credential from Supabase vault."""
        return await self.fallback.delete(reference_key)
    
    async def list_keys(self, prefix: str) -> List[str]:
        """List credential keys from Supabase vault."""
        return await self.fallback.list_keys(prefix) 