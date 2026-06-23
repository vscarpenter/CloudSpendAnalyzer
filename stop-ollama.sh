#!/bin/bash

# Ollama Stop Script
# Stops the Ollama server started by start-ollama.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "${BLUE}🛑 Stopping Ollama Backend${NC}"

# Check for PID file
if [ -f .ollama.pid ]; then
    PID=$(cat .ollama.pid)
    
    if kill -0 "$PID" 2>/dev/null; then
        log "${BLUE}⏹ Stopping Ollama server (PID: $PID)...${NC}"
        kill "$PID"
        
        # Wait for process to stop
        sleep 2
        
        if kill -0 "$PID" 2>/dev/null; then
            log "${YELLOW}⚠ Process still running, forcing stop...${NC}"
            kill -9 "$PID" 2>/dev/null
        fi
        
        log "${GREEN}✓ Ollama server stopped${NC}"
    else
        log "${YELLOW}⚠ Process with PID $PID is not running${NC}"
    fi
    
    # Clean up PID file
    rm -f .ollama.pid
    log "${GREEN}✓ Cleanup complete${NC}"
else
    log "${YELLOW}⚠ No PID file found (.ollama.pid)${NC}"
    
    # Try to find and stop ollama processes
    OLLAMA_PIDS=$(pgrep -f "ollama serve" || true)
    
    if [ -n "$OLLAMA_PIDS" ]; then
        log "${BLUE}🔍 Found running Ollama processes: $OLLAMA_PIDS${NC}"
        
        # Graceful shutdown first
        echo "$OLLAMA_PIDS" | xargs kill -TERM 2>/dev/null || true
        
        # Wait for graceful shutdown (important for large models)
        local wait_count=0
        while [ $wait_count -lt 10 ]; do
            REMAINING_PIDS=$(pgrep -f "ollama serve" || true)
            if [ -z "$REMAINING_PIDS" ]; then
                break
            fi
            sleep 1
            wait_count=$((wait_count + 1))
            log "${BLUE}⏳ Waiting for graceful shutdown... (${wait_count}/10)${NC}"
        done
        
        # Force kill if still running
        REMAINING_PIDS=$(pgrep -f "ollama serve" || true)
        if [ -n "$REMAINING_PIDS" ]; then
            log "${YELLOW}⚠ Force stopping remaining processes...${NC}"
            echo "$REMAINING_PIDS" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
        
        log "${GREEN}✓ Ollama processes stopped${NC}"
    else
        log "${GREEN}ℹ No running Ollama processes found${NC}"
    fi
fi

log "${GREEN}🏁 Ollama backend stopped${NC}"
