#!/bin/bash

# Queue SciFact Experiment - Waits for FIQA to complete, then starts SciFact
# This ensures sequential execution to avoid memory issues

echo "🔬 Queuing SciFact experiment to run after FIQA completes..."
echo "Monitoring FIQA experiment results file..."

FIQA_RESULTS="/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments/experiments/chunking/results/fiqa_sequential_optimization.json"
SCIFACT_OUTPUT="/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments/experiments/chunking/results/scifact_sequential_optimization.json"

# Function to check if FIQA experiment is complete
is_fiqa_complete() {
    if [ -f "$FIQA_RESULTS" ]; then
        # Check if file contains completion marker or has all 510 results
        result_count=$(jq -r '.results | length' "$FIQA_RESULTS" 2>/dev/null || echo "0")
        if [ "$result_count" -eq 510 ] 2>/dev/null; then
            return 0  # Complete
        fi
        
        # Also check for completion metadata
        status=$(jq -r '.metadata.status' "$FIQA_RESULTS" 2>/dev/null || echo "running")
        if [ "$status" = "completed" ]; then
            return 0  # Complete
        fi
    fi
    return 1  # Not complete
}

# Monitor FIQA completion
check_count=0
while true; do
    if is_fiqa_complete; then
        echo "✅ FIQA experiment completed! Starting SciFact experiment..."
        break
    fi
    
    check_count=$((check_count + 1))
    
    # Show progress every 10 checks (5 minutes)
    if [ $((check_count % 10)) -eq 0 ]; then
        if [ -f "$FIQA_RESULTS" ]; then
            result_count=$(jq -r '.results | length' "$FIQA_RESULTS" 2>/dev/null || echo "0")
            echo "⏳ FIQA progress: $result_count/510 results ($(date '+%H:%M:%S'))"
        else
            echo "⏳ Waiting for FIQA experiment to generate results file... ($(date '+%H:%M:%S'))"
        fi
    fi
    
    sleep 30  # Check every 30 seconds
done

# Start SciFact experiment
echo "🚀 Starting SciFact chunk optimization experiment..."
cd /Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments

# Use conda environment and run SciFact experiment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && echo "y" | python main.py experiment template chunk_optimization \
    --corpus scifact_scientific \
    --queries experiments/chunking/queries/chunking_queries.json \
    --output "$SCIFACT_OUTPUT"

echo "🎉 SciFact experiment queued and started successfully!"