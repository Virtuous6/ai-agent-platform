# 🚀 AI Agent Platform - Google Cloud Deployment Guide

## Quick Start

### Prerequisites
1. **Google Cloud SDK** installed and configured
2. **Docker** installed and running
3. **GCP Project** with billing enabled
4. **Required API keys** (OpenAI, Slack, Supabase)

### One-Click Deployment
```bash
# Run the automated deployment script
python deployment/gcp_run_deploy.py
```

This script will:
- ✅ Check prerequisites
- ✅ Enable required Google Cloud APIs
- ✅ Create service accounts with proper IAM roles
- ✅ Build and deploy using Cloud Build
- ✅ Configure Cloud Run service
- ✅ Set up monitoring and health checks

## Manual Deployment Steps

### 1. Setup Google Cloud
```bash
# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Deploy Application
```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Get service URL
gcloud run services describe ai-agent-platform \
  --region=us-central1 \
  --format="value(status.url)"
```

### 3. Configure Environment Variables
In Google Cloud Console → Cloud Run → ai-agent-platform → Edit:

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_TRACKING=true
OPENAI_API_KEY=your-openai-key
SLACK_BOT_TOKEN=your-slack-token
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

## Monitoring and Maintenance

### Health Checks
- **Health endpoint**: `https://your-service-url/health`
- **Metrics endpoint**: `https://your-service-url/metrics`
- **Status endpoint**: `https://your-service-url/status`

### Useful Commands
```bash
# View logs
gcloud run services logs tail ai-agent-platform --region=us-central1

# Update deployment
gcloud builds submit --config cloudbuild.yaml

# Scale service
gcloud run services update ai-agent-platform \
  --region=us-central1 \
  --max-instances=50

# Delete service
gcloud run services delete ai-agent-platform --region=us-central1
```

### Monitoring URLs
- **Cloud Run Console**: `https://console.cloud.google.com/run`
- **Cloud Build**: `https://console.cloud.google.com/cloud-build`
- **Logs Explorer**: `https://console.cloud.google.com/logs`
- **Monitoring**: `https://console.cloud.google.com/monitoring`

## Configuration Options

### Performance Tuning
```yaml
# In cloud-run-service.yaml
resources:
  limits:
    cpu: '2'        # Adjust based on load
    memory: 2Gi     # Adjust based on memory usage
  requests:
    cpu: '1'        # Minimum CPU allocation
    memory: 1Gi     # Minimum memory allocation
```

### Auto-scaling
```yaml
annotations:
  autoscaling.knative.dev/maxScale: '100'  # Maximum instances
  autoscaling.knative.dev/minScale: '0'    # Minimum instances (scale to zero)
  autoscaling.knative.dev/target: '80'     # Target concurrent requests per instance
```

## Cost Optimization

### Automatic Cost Controls
- **Scale to zero** when not in use
- **Request-based pricing** - only pay for actual usage
- **Optimized builds** with caching reduce build costs
- **Right-sized resources** prevent over-provisioning

### Cost Monitoring
```bash
# View current costs
gcloud billing budgets list

# Set up budget alerts
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="AI Agent Platform Budget" \
  --budget-amount=100USD
```

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check build logs
gcloud builds log BUILD_ID

# Debug locally
docker build -t test-image .
docker run --rm test-image
```

#### Service Not Starting
```bash
# Check service logs
gcloud run services logs tail ai-agent-platform --region=us-central1

# Check health endpoint
curl https://your-service-url/health
```

#### Permission Issues
```bash
# Check service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Add missing permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:ai-agent-platform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
```

## Security Best Practices

### Service Configuration
- ✅ Non-root container user
- ✅ Read-only root filesystem options
- ✅ Minimal attack surface
- ✅ Secure environment variable handling
- ✅ IAM least-privilege access

### Network Security
- ✅ VPC-native networking ready
- ✅ HTTPS-only endpoints
- ✅ Cloud Armor integration ready
- ✅ Private Google Access configured

## Next Steps

1. **Test your deployment** with the health endpoint
2. **Configure monitoring alerts** for production use
3. **Set up CI/CD** with Cloud Build triggers
4. **Enable Cloud Armor** for DDoS protection
5. **Configure custom domains** if needed

---

🎉 **Your AI Agent Platform is now running on Google Cloud Run!**

For support, check the logs and monitoring dashboards in the Google Cloud Console. 