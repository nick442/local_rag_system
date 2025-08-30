#!/bin/bash
# Local Deployment Script for RAG System
# Creates isolated environment and sets up the system for production use

set -e  # Exit on error

# Configuration
PROJECT_NAME="rag_system"
PYTHON_VERSION="3.11"
VENV_NAME="rag_env_prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        log_info "Found Python $python_version"
        
        # Check if version is sufficient (3.8+)
        if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_error "Python 3.8+ required, found $python_version"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check available tools
    if command -v conda &> /dev/null; then
        log_info "Conda found - will use conda for environment"
        USE_CONDA=true
    elif command -v python3 -m venv &> /dev/null; then
        log_info "Python venv available - will use venv for environment"
        USE_CONDA=false
    else
        log_error "Neither conda nor venv available. Cannot create virtual environment."
        exit 1
    fi
}

create_environment() {
    log_info "Creating production environment..."
    
    if [ "$USE_CONDA" = true ]; then
        # Use conda
        if conda env list | grep -q "^${VENV_NAME}"; then
            log_warning "Conda environment '${VENV_NAME}' already exists"
            read -p "Remove and recreate? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda env remove -n "${VENV_NAME}" -y
            else
                log_info "Using existing environment"
                return 0
            fi
        fi
        
        log_info "Creating conda environment with Python ${PYTHON_VERSION}..."
        conda create -n "${VENV_NAME}" python="${PYTHON_VERSION}" -y
        
        log_info "Activating conda environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${VENV_NAME}"
        
    else
        # Use venv
        if [ -d "${VENV_NAME}" ]; then
            log_warning "Virtual environment '${VENV_NAME}' already exists"
            read -p "Remove and recreate? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "${VENV_NAME}"
            else
                log_info "Using existing environment"
                source "${VENV_NAME}/bin/activate"
                return 0
            fi
        fi
        
        log_info "Creating virtual environment..."
        python3 -m venv "${VENV_NAME}"
        
        log_info "Activating virtual environment..."
        source "${VENV_NAME}/bin/activate"
    fi
    
    log_success "Environment created and activated"
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Check for requirements file
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found, installing core dependencies..."
        pip install torch sentence-transformers llama-cpp-python click rich psutil
    fi
    
    log_success "Dependencies installed"
}

setup_directories() {
    log_info "Setting up directory structure..."
    
    # Create necessary directories
    mkdir -p data logs models config
    mkdir -p models/embeddings
    
    # Set permissions
    chmod 755 data logs models config
    
    log_success "Directory structure created"
}

download_models() {
    log_info "Checking model availability..."
    
    # Check if LLM model exists
    if [ ! -f "models/gemma-3-4b-it-q4_0.gguf" ] && [ ! -f "models/llama-3.2-3b-instruct-q4_0.gguf" ]; then
        log_warning "No LLM model found in models/ directory"
        log_info "You'll need to download a GGUF model file manually"
        log_info "Recommended models:"
        log_info "  - Gemma-3-4b-it-q4_0.gguf"
        log_info "  - Llama-3.2-3b-instruct-q4_0.gguf"
    else
        log_success "LLM model found"
    fi
    
    # Embedding model will be downloaded automatically on first use
    log_info "Embedding model will be downloaded automatically on first use"
}

create_systemd_service() {
    if command -v systemctl &> /dev/null && [ -d "/etc/systemd/system" ]; then
        log_info "Creating systemd service file..."
        
        # Get current directory
        CURRENT_DIR=$(pwd)
        PYTHON_PATH=$(which python)
        
        cat > /tmp/rag-system.service << EOF
[Unit]
Description=RAG System - Local AI Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PYTHON_PATH main.py status
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        log_info "Systemd service file created at /tmp/rag-system.service"
        log_info "To install: sudo mv /tmp/rag-system.service /etc/systemd/system/"
        log_info "To enable: sudo systemctl enable rag-system.service"
        log_info "To start: sudo systemctl start rag-system.service"
    else
        log_info "Systemd not available, skipping service creation"
    fi
}

create_macos_launch_agent() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Creating macOS launch agent..."
        
        # Get current directory and python path
        CURRENT_DIR=$(pwd)
        PYTHON_PATH=$(which python)
        
        # Create launch agent directory if it doesn't exist
        mkdir -p ~/Library/LaunchAgents
        
        cat > ~/Library/LaunchAgents/com.local.rag.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.rag</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>main.py</string>
        <string>status</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$CURRENT_DIR</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardErrorPath</key>
    <string>$CURRENT_DIR/logs/launch_agent.log</string>
    <key>StandardOutPath</key>
    <string>$CURRENT_DIR/logs/launch_agent.log</string>
</dict>
</plist>
EOF
        
        log_success "macOS launch agent created at ~/Library/LaunchAgents/com.local.rag.plist"
        log_info "To load: launchctl load ~/Library/LaunchAgents/com.local.rag.plist"
        log_info "To start: launchctl start com.local.rag"
    fi
}

run_system_check() {
    log_info "Running system verification..."
    
    if [ -f "scripts/setup_check.py" ]; then
        python scripts/setup_check.py --verbose
        if [ $? -eq 0 ]; then
            log_success "System check passed"
        else
            log_warning "System check found issues - check output above"
        fi
    else
        log_warning "setup_check.py not found, skipping system verification"
    fi
}

create_start_script() {
    log_info "Creating start script..."
    
    cat > start_rag_system.sh << EOF
#!/bin/bash
# Start script for RAG System

# Activate environment
if [ "$USE_CONDA" = true ]; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${VENV_NAME}"
else
    source "${VENV_NAME}/bin/activate"
fi

# Start the system
echo "Starting RAG System..."
python main.py "\$@"
EOF
    
    chmod +x start_rag_system.sh
    log_success "Start script created: ./start_rag_system.sh"
}

main() {
    echo "========================================="
    echo "RAG System Local Deployment"
    echo "========================================="
    echo
    
    # Parse command line arguments
    SKIP_CHECKS=false
    SKIP_MODELS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--skip-checks] [--skip-models]"
                echo
                echo "Options:"
                echo "  --skip-checks    Skip system requirements checking"
                echo "  --skip-models    Skip model download prompts"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    if [ "$SKIP_CHECKS" = false ]; then
        check_requirements
    fi
    
    create_environment
    install_dependencies
    setup_directories
    
    if [ "$SKIP_MODELS" = false ]; then
        download_models
    fi
    
    create_systemd_service
    create_macos_launch_agent
    create_start_script
    
    if [ "$SKIP_CHECKS" = false ]; then
        run_system_check
    fi
    
    echo
    echo "========================================="
    log_success "Deployment completed successfully!"
    echo "========================================="
    echo
    log_info "Quick start:"
    log_info "  1. Use the start script: ./start_rag_system.sh status"
    log_info "  2. Or activate environment and run directly:"
    
    if [ "$USE_CONDA" = true ]; then
        log_info "     conda activate ${VENV_NAME}"
    else
        log_info "     source ${VENV_NAME}/bin/activate"
    fi
    
    log_info "     python main.py --help"
    echo
    log_info "For system diagnostics: python main.py doctor"
    log_info "For setup verification: python scripts/setup_check.py"
    echo
}

# Run main function
main "$@"