/**
 * AI Agent Platform Dashboard JavaScript
 * Handles tab navigation, data fetching, real-time updates, and interactions
 */

class Dashboard {
    constructor() {
        this.currentTab = 'overview';
        this.updateInterval = null;
        this.connectionStatus = 'connecting';
        
        this.init();
    }
    
    init() {
        this.initEventListeners();
        this.initModal();
        this.startDataRefresh();
        this.loadInitialData();
    }
    
    initEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Add MCP button
        const addMCPBtn = document.getElementById('addMCPBtn');
        if (addMCPBtn) {
            addMCPBtn.addEventListener('click', () => {
                this.showMCPModal();
            });
        }
    }
    
    initModal() {
        const modal = document.getElementById('mcpModal');
        const closeBtn = document.getElementById('modalClose');
        const cancelBtn = document.getElementById('modalCancel');
        const saveBtn = document.getElementById('modalSave');
        
        // Close modal events
        [closeBtn, cancelBtn].forEach(btn => {
            if (btn) {
                btn.addEventListener('click', () => {
                    this.hideMCPModal();
                });
            }
        });
        
        // Save MCP connection
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                this.createMCPConnection();
            });
        }
        
        // Close modal on backdrop click
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideMCPModal();
                }
            });
        }
    }
    
    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load data for the new tab
        this.loadTabData(tabName);
    }
    
    async loadInitialData() {
        try {
            await this.loadTabData('overview');
            this.updateConnectionStatus('connected');
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateConnectionStatus('error');
        }
    }
    
    async loadTabData(tabName) {
        try {
            switch (tabName) {
                case 'overview':
                    await this.loadOverviewData();
                    break;
                case 'messages':
                    await this.loadMessagesData();
                    break;
                case 'workflows':
                    await this.loadWorkflowsData();
                    break;
                case 'mcp-connections':
                    await this.loadMCPConnectionsData();
                    break;
                case 'conversations':
                    await this.loadConversationsData();
                    break;
                case 'costs':
                    await this.loadCostsData();
                    break;
            }
        } catch (error) {
            console.error(`Failed to load ${tabName} data:`, error);
        }
    }
    
    async loadOverviewData() {
        const response = await fetch('/api/overview');
        const data = await response.json();
        
        if (data.error) {
            console.error('Overview data error:', data.error);
            return;
        }
        
        // Update system health
        const healthScore = data.system_health?.overall_score || 0;
        document.getElementById('healthScore').textContent = `${Math.round(healthScore * 100)}%`;
        
        const healthIndicator = document.getElementById('healthIndicator');
        if (healthScore >= 0.8) {
            healthIndicator.style.color = 'var(--success-color)';
        } else if (healthScore >= 0.6) {
            healthIndicator.style.color = 'var(--warning-color)';
        } else {
            healthIndicator.style.color = 'var(--error-color)';
        }
        
        // Update cost metrics
        document.getElementById('todayCost').textContent = `$${data.costs?.today_cost?.toFixed(4) || '0.0000'}`;
        document.getElementById('weeklyCost').textContent = `$${data.costs?.weekly_cost?.toFixed(4) || '0.0000'}`;
        
        // Update activity metrics
        document.getElementById('totalMessages').textContent = data.activity?.total_messages || 0;
        document.getElementById('activeMCPs').textContent = data.activity?.active_mcps || 0;
        document.getElementById('activeConversations').textContent = data.activity?.active_conversations || 0;
        
        // Update recent activity
        this.updateRecentMessages(data.recent_activity?.messages || []);
        this.updateRecentWorkflows(data.recent_activity?.workflows || []);
    }
    
    async loadMessagesData() {
        const response = await fetch('/api/messages');
        const data = await response.json();
        
        if (data.error) {
            console.error('Messages data error:', data.error);
            return;
        }
        
        // Update stats
        document.getElementById('messagesTotalCount').textContent = data.total_count || 0;
        document.getElementById('messagesSuccessRate').textContent = `${(data.success_rate || 0).toFixed(1)}%`;
        
        // Update table
        this.updateMessagesTable(data.messages || []);
    }
    
    async loadWorkflowsData() {
        const response = await fetch('/api/workflows');
        const data = await response.json();
        
        if (data.error) {
            console.error('Workflows data error:', data.error);
            return;
        }
        
        // Update stats
        document.getElementById('workflowsTotalCount').textContent = data.total_count || 0;
        document.getElementById('workflowsSuccessRate').textContent = `${(data.success_rate || 0).toFixed(1)}%`;
        
        // Update table
        this.updateWorkflowsTable(data.workflows || []);
    }
    
    async loadMCPConnectionsData() {
        const response = await fetch('/api/mcp-connections');
        const data = await response.json();
        
        if (data.error) {
            console.error('MCP connections data error:', data.error);
            return;
        }
        
        // Update table
        this.updateMCPConnectionsTable(data.connections || []);
    }
    
    async loadConversationsData() {
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        if (data.error) {
            console.error('Conversations data error:', data.error);
            return;
        }
        
        // Update stats
        document.getElementById('conversationsTotalCount').textContent = data.total_count || 0;
        document.getElementById('conversationsActiveCount').textContent = data.active_count || 0;
        
        // Update table
        this.updateConversationsTable(data.conversations || []);
    }
    
    async loadCostsData() {
        const response = await fetch('/api/costs');
        const data = await response.json();
        
        if (data.error) {
            console.error('Costs data error:', data.error);
            return;
        }
        
        // Update cost metrics
        document.getElementById('costTodayValue').textContent = `$${(data.today_cost || 0).toFixed(4)}`;
        document.getElementById('costWeeklyValue').textContent = `$${(data.weekly_cost || 0).toFixed(4)}`;
        document.getElementById('costMonthlyValue').textContent = `$${(data.monthly_projection || 0).toFixed(2)}`;
        document.getElementById('costEfficiencyValue').textContent = `${Math.round((data.efficiency_score || 0) * 100)}%`;
        
        // Update optimizations
        this.updateOptimizationsList(data.optimizations || []);
    }
    
    updateRecentMessages(messages) {
        const container = document.getElementById('recentMessages');
        if (!container) return;
        
        if (messages.length === 0) {
            container.innerHTML = '<div class="loading">No recent messages</div>';
            return;
        }
        
        container.innerHTML = messages.map(msg => `
            <div class="activity-item">
                <div>${this.truncateText(msg.content || '', 60)}</div>
                <div class="activity-meta">${msg.role} • ${msg.timestamp}</div>
            </div>
        `).join('');
    }
    
    updateRecentWorkflows(workflows) {
        const container = document.getElementById('recentWorkflows');
        if (!container) return;
        
        if (workflows.length === 0) {
            container.innerHTML = '<div class="loading">No recent workflows</div>';
            return;
        }
        
        container.innerHTML = workflows.map(wf => `
            <div class="activity-item">
                <div>${wf.runbook_name || 'Unknown'}</div>
                <div class="activity-meta">${this.getStatusBadge(wf.status)} • ${wf.started_at}</div>
            </div>
        `).join('');
    }
    
    updateMessagesTable(messages) {
        const tbody = document.querySelector('#messagesTable tbody');
        if (!tbody) return;
        
        if (messages.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No messages found</td></tr>';
            return;
        }
        
        tbody.innerHTML = messages.map(msg => `
            <tr>
                <td>${msg.id || 'N/A'}</td>
                <td>${this.getStatusBadge(msg.role, 'info')}</td>
                <td>${this.truncateText(msg.content || '', 50)}</td>
                <td>${msg.user_id || 'N/A'}</td>
                <td>${msg.timestamp || 'N/A'}</td>
                <td>${msg.tokens || 0}</td>
                <td>${msg.error ? this.getStatusBadge('Error', 'error') : this.getStatusBadge('OK', 'success')}</td>
            </tr>
        `).join('');
    }
    
    updateWorkflowsTable(workflows) {
        const tbody = document.querySelector('#workflowsTable tbody');
        if (!tbody) return;
        
        if (workflows.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No workflows found</td></tr>';
            return;
        }
        
        tbody.innerHTML = workflows.map(wf => `
            <tr>
                <td>${wf.id || 'N/A'}</td>
                <td>${wf.runbook_name || 'N/A'}</td>
                <td>${this.getStatusBadge(wf.status)}</td>
                <td>${wf.duration || 'N/A'}</td>
                <td>${wf.user_id || 'N/A'}</td>
                <td>${wf.cost || '$0.0000'}</td>
                <td>${wf.agents_used || 0}</td>
            </tr>
        `).join('');
    }
    
    updateMCPConnectionsTable(connections) {
        const tbody = document.querySelector('#mcpTable tbody');
        if (!tbody) return;
        
        if (connections.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No MCP connections found</td></tr>';
            return;
        }
        
        tbody.innerHTML = connections.map(conn => `
            <tr>
                <td>${conn.name || 'N/A'}</td>
                <td>${this.getStatusBadge(conn.type, 'info')}</td>
                <td>${this.getStatusBadge(conn.status)}</td>
                <td>${conn.success_rate || 'N/A'}</td>
                <td>${conn.tools_count || 0}</td>
                <td>${conn.last_used || 'Never'}</td>
                <td>${conn.monthly_cost || '$0.00'}</td>
            </tr>
        `).join('');
    }
    
    updateConversationsTable(conversations) {
        const tbody = document.querySelector('#conversationsTable tbody');
        if (!tbody) return;
        
        if (conversations.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="loading-row">No conversations found</td></tr>';
            return;
        }
        
        tbody.innerHTML = conversations.map(conv => `
            <tr>
                <td>${conv.id || 'N/A'}</td>
                <td>${conv.user_id || 'N/A'}</td>
                <td>${this.getStatusBadge(conv.status)}</td>
                <td>${conv.started_at || 'N/A'}</td>
                <td>${conv.total_messages || 0}</td>
                <td>${conv.total_cost || '$0.0000'}</td>
            </tr>
        `).join('');
    }
    
    updateOptimizationsList(optimizations) {
        const container = document.getElementById('optimizationsList');
        if (!container) return;
        
        if (optimizations.length === 0) {
            container.innerHTML = '<div class="loading">No optimizations available</div>';
            return;
        }
        
        container.innerHTML = optimizations.map(opt => `
            <div class="optimization-item">
                <div class="optimization-info">
                    <h4>${opt.title || 'Unknown'}</h4>
                    <p>Complexity: ${opt.complexity || 'Unknown'} • Priority: ${opt.priority || 'N/A'}</p>
                </div>
                <div class="optimization-savings">${opt.savings || '$0.00'}</div>
            </div>
        `).join('');
    }
    
    async showMCPModal() {
        // Load MCP types first
        try {
            const response = await fetch('/api/mcp-types');
            const data = await response.json();
            
            const typeSelect = document.getElementById('mcpType');
            if (typeSelect && data.mcp_types) {
                typeSelect.innerHTML = '<option value="">Select a type...</option>' +
                    data.mcp_types.map(type => `
                        <option value="${type.mcp_type}">${type.display_name}</option>
                    `).join('');
            }
        } catch (error) {
            console.error('Failed to load MCP types:', error);
        }
        
        // Show modal
        const modal = document.getElementById('mcpModal');
        if (modal) {
            modal.classList.add('show');
        }
    }
    
    hideMCPModal() {
        const modal = document.getElementById('mcpModal');
        if (modal) {
            modal.classList.remove('show');
        }
        
        // Reset form
        const form = document.getElementById('mcpForm');
        if (form) {
            form.reset();
        }
    }
    
    async createMCPConnection() {
        const form = document.getElementById('mcpForm');
        if (!form) return;
        
        const formData = new FormData(form);
        const data = {
            mcp_type: formData.get('mcp_type'),
            connection_name: formData.get('connection_name'),
            user_id: formData.get('user_id'),
            description: formData.get('description'),
            status: 'active'
        };
        
        // Validate required fields
        if (!data.mcp_type || !data.connection_name || !data.user_id) {
            alert('Please fill in all required fields');
            return;
        }
        
        try {
            const response = await fetch('/api/mcp-connections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                alert(`MCP connection created successfully: ${result.connection_id}`);
                this.hideMCPModal();
                
                // Refresh MCP connections if on that tab
                if (this.currentTab === 'mcp-connections') {
                    await this.loadMCPConnectionsData();
                }
            } else {
                alert(`Failed to create connection: ${result.error}`);
            }
        } catch (error) {
            console.error('Failed to create MCP connection:', error);
            alert('Failed to create connection. Please try again.');
        }
    }
    
    getStatusBadge(status, type = null) {
        if (!status) return '<span class="status-badge info">Unknown</span>';
        
        let badgeType = type;
        if (!badgeType) {
            switch (status.toLowerCase()) {
                case 'completed':
                case 'active':
                case 'success':
                case 'ok':
                    badgeType = 'success';
                    break;
                case 'failed':
                case 'error':
                case 'inactive':
                    badgeType = 'error';
                    break;
                case 'running':
                case 'pending':
                case 'warning':
                    badgeType = 'warning';
                    break;
                default:
                    badgeType = 'info';
            }
        }
        
        return `<span class="status-badge ${badgeType}">${status}</span>`;
    }
    
    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
    
    updateConnectionStatus(status) {
        this.connectionStatus = status;
        
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }
        
        if (statusText) {
            switch (status) {
                case 'connected':
                    statusText.textContent = 'Connected';
                    break;
                case 'error':
                    statusText.textContent = 'Connection Error';
                    break;
                default:
                    statusText.textContent = 'Connecting...';
            }
        }
    }
    
    updateLastUpdated() {
        const lastUpdated = document.getElementById('lastUpdated');
        if (lastUpdated) {
            lastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        }
    }
    
    startDataRefresh() {
        // Update data every 30 seconds
        this.updateInterval = setInterval(() => {
            if (this.connectionStatus === 'connected') {
                this.loadTabData(this.currentTab);
                this.updateLastUpdated();
            }
        }, 30000);
        
        // Update timestamp every second
        setInterval(() => {
            this.updateLastUpdated();
        }, 1000);
    }
    
    stopDataRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.stopDataRefresh();
    }
}); 