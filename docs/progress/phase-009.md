# Phase 9: System Integration Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Review the complete system by examining all handoff files:
```bash
# Generate system overview
for file in docs/handoff/phase_*_complete.json; do
  phase=$(basename "$file" | grep -o '[0-9]')
  echo "Phase $phase: $(cat "$file" | jq -r '.created_files | length') files created"
done
```

## Your Mission
Integrate all components into a unified, production-ready system with proper initialization, health checks, error handling, and graceful lifecycle management.

## Prerequisites Check
1. Verify all components: `find src/ -name "*.py" | wc -l` (should be 15+ files)
2. Check database exists: `ls -la data/rag_vectors.db`
3. Verify main CLI works: `python main.py --help`

## Implementation Tasks

### Task 9.1: Unified Application Entry Point
Update `main.py` with complete integration:

```python
#!/usr/bin/env python
"""
Local RAG System - Main Application
Unified entry point for all functionality
"""

# Required structure:
# 1. Environment validation
# 2. Component initialization
# 3. Health checks
# 4. Graceful shutdown
# 5. Error recovery
```

Complete implementation:
```python
import click
import sys
import signal
from pathlib import Path
from src.system_manager import SystemManager

# Global system manager
system = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    if system:
        system.shutdown()
    sys.exit(0)

@click.group()
@click.pass_context
def cli(ctx):
    """Local RAG System - Offline AI Assistant"""
    global system
    
    # Initialize system manager
    system = SystemManager()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run health checks
    if not system.health_check():
        click.echo("System health check failed. Run 'python main.py doctor' for details.")
        sys.exit(1)
    
    ctx.obj = system

# Add all commands
@cli.command()
@click.pass_obj
def chat(system):
    """Interactive chat interface"""
    from src.cli_chat import run_chat
    run_chat(system)

# ... other commands
```

### Task 9.2: System Manager
Create `src/system_manager.py`:

```python
# Central system coordination:
# 1. Component lifecycle management
# 2. Dependency injection
# 3. Resource management
# 4. State persistence
# 5. Error recovery
```

Implementation:
```python
import json
import logging
from pathlib import Path
from datetime import datetime

class SystemManager:
    def __init__(self, config_path="config/system.json"):
        self.config = self.load_config(config_path)
        self.components = {}
        self.state = {}
        self.logger = self.setup_logging()
        
    def initialize_components(self):
        """Lazy initialization of components"""
        # Only load what's needed
        pass
        
    def health_check(self):
        """Verify system readiness"""
        checks = {
            "model_exists": self.check_model(),
            "database_ready": self.check_database(),
            "embeddings_available": self.check_embeddings(),
            "memory_sufficient": self.check_memory()
        }
        return all(checks.values())
        
    def get_component(self, name):
        """Lazy-load components on demand"""
        if name not in self.components:
            self.components[name] = self._create_component(name)
        return self.components[name]
        
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down system")
        self.save_state()
        for component in self.components.values():
            if hasattr(component, 'close'):
                component.close()
```

### Task 9.3: Health Check System
Create `src/health_checks.py`:

```python
# System health monitoring:
# 1. Model availability
# 2. Database connectivity
# 3. Memory availability
# 4. Disk space
# 5. Component status
```

Health check implementation:
```python
class HealthChecker:
    def __init__(self, system_manager):
        self.system = system_manager
        
    def run_all_checks(self):
        """Run comprehensive health checks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Model check
        results["checks"]["model"] = {
            "status": self.check_model(),
            "details": self.get_model_info()
        }
        
        # Database check
        results["checks"]["database"] = {
            "status": self.check_database(),
            "details": self.get_database_stats()
        }
        
        # Memory check
        results["checks"]["memory"] = {
            "status": self.check_memory(),
            "available_gb": self.get_available_memory()
        }
        
        # Disk check
        results["checks"]["disk"] = {
            "status": self.check_disk_space(),
            "available_gb": self.get_available_disk()
        }
        
        return results
    
    def generate_report(self, results):
        """Generate health report"""
        # Format as markdown or JSON
        pass
```

### Task 9.4: Error Handling and Recovery
Create `src/error_handler.py`:

```python
# Centralized error handling:
# 1. Error categorization
# 2. Recovery strategies
# 3. User-friendly messages
# 4. Logging and alerting
```

Error handling patterns:
```python
class ErrorHandler:
    def __init__(self):
        self.recovery_strategies = {
            "OutOfMemoryError": self.handle_oom,
            "ModelLoadError": self.handle_model_error,
            "DatabaseError": self.handle_db_error,
            "NetworkError": self.handle_network_error
        }
        
    def handle_error(self, error, context=None):
        """Route errors to appropriate handlers"""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, context)
        else:
            return self.handle_unknown_error(error, context)
            
    def handle_oom(self, error, context):
        """Handle out of memory errors"""
        # 1. Try to free memory
        # 2. Reduce batch sizes
        # 3. Swap to smaller model
        # 4. Graceful degradation
        
    def handle_model_error(self, error, context):
        """Handle model loading errors"""
        # 1. Verify model file
        # 2. Check compatibility
        # 3. Try alternative model
        # 4. Download if missing
```

### Task 9.5: Utility Scripts
Create utility scripts in `scripts/` directory:

1. **scripts/doctor.py** - System diagnostics:
```python
#!/usr/bin/env python
"""System diagnostics and troubleshooting"""

def diagnose_system():
    # Check all components
    # Identify issues
    # Suggest fixes
    pass
```

2. **scripts/setup_check.py** - Verify installation:
```python
#!/usr/bin/env python
"""Verify system setup"""

def check_installation():
    # Check Python version
    # Verify packages
    # Test imports
    # Check Metal support
    pass
```

3. **scripts/reset_system.py** - Factory reset:
```python
#!/usr/bin/env python
"""Reset system to clean state"""

def reset_system(keep_models=True):
    # Clear database
    # Reset config
    # Clean logs
    # Preserve models if requested
    pass
```

### Task 9.6: Docker Alternative (Local Container)
Create `deployment/local_deploy.sh`:

```bash
#!/bin/bash
# Local deployment script (no Docker needed)

# Create isolated environment
python -m venv rag_env_prod
source rag_env_prod/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup systemd service (optional)
# Create launch agent for macOS
```

Create `deployment/launch_agent.plist` for macOS:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.rag</string>
    <key>Program</key>
    <string>/path/to/main.py</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

## Testing Requirements
Create `test_phase_9.py`:
1. Test system initialization
2. Test health checks
3. Test error recovery
4. Test graceful shutdown
5. Test component lazy loading
6. Test resource cleanup

## Output Requirements
Create `handoff/phase_9_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 9,
  "created_files": [
    "src/system_manager.py",
    "src/health_checks.py",
    "src/error_handler.py",
    "scripts/doctor.py",
    "scripts/setup_check.py",
    "scripts/reset_system.py",
    "deployment/local_deploy.sh",
    "deployment/launch_agent.plist",
    "test_phase_9.py"
  ],
  "integration_features": {
    "unified_entry_point": true,
    "health_monitoring": true,
    "error_recovery": true,
    "graceful_shutdown": true,
    "resource_management": true,
    "lazy_loading": true
  },
  "system_commands": [
    "python main.py chat",
    "python main.py doctor",
    "python main.py status",
    "python scripts/doctor.py",
    "python scripts/reset_system.py"
  ],
  "health_check_results": {
    "model": "OK",
    "database": "OK",
    "memory": "OK",
    "disk": "OK"
  },
  "deployment_ready": true
}
```

## System Requirements Validation
Ensure the system meets these requirements:
1. **Startup time**: <5 seconds to interactive
2. **Memory baseline**: <500MB before model load
3. **Graceful shutdown**: <2 seconds
4. **Error recovery**: Handles OOM without crash
5. **Component isolation**: Failed component doesn't crash system

## Final Integration Checklist
- [ ] Main entry point handles all commands
- [ ] System manager coordinates components
- [ ] Health checks identify issues
- [ ] Errors are handled gracefully
- [ ] Resources are managed efficiently
- [ ] Shutdown is clean
- [ ] Diagnostic tools work
- [ ] System can be reset
- [ ] Deployment scripts ready
- [ ] Handoff file created

Remember: This integration phase ensures the system is production-ready. The next phase will optimize performance.