#!/usr/bin/env python3
"""
Main orchestrator script for chunking optimization experiments.

This script coordinates the entire experimental pipeline from system preparation
through analysis and reporting, following the approved experimental plan.
"""

import sys
import time
import json
from pathlib import Path
from typing import List

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from experiment_logger import ExperimentLogger
from experiment_runner import ChunkingExperimentRunner
from analysis_tools import ChunkingAnalyzer

class ExperimentOrchestrator:
    """Main orchestrator for the complete experimental pipeline."""
    
    def __init__(self):
        self.base_dir = Path("/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments")
        self.runner = ChunkingExperimentRunner(str(self.base_dir))
        self.logger = self.runner.logger
        
        # Load test queries
        self.queries = self.load_evaluation_queries()
        
    def load_evaluation_queries(self) -> List[str]:
        """Load evaluation queries from JSON file."""
        queries_file = self.base_dir / "test_data" / "chunking_queries.json"
        
        try:
            with open(queries_file, 'r') as f:
                data = json.load(f)
            
            # Extract all queries from categories
            all_queries = []
            for category, queries in data['categories'].items():
                for query_item in queries:
                    if isinstance(query_item, dict):
                        all_queries.append(query_item['query'])
                    else:
                        all_queries.append(str(query_item))
            
            self.logger.log_info(f"Loaded {len(all_queries)} evaluation queries", "query_loading")
            return all_queries
            
        except Exception as e:
            self.logger.log_error(e, "query_loading")
            # Fallback to default queries
            return [
                "What are FSA health insurance premiums?",
                "What is the apparent diffusion coefficient?",
                "What prevents false credit ratings in financial markets?"
            ]
    
    def run_phase_1_preparation(self) -> bool:
        """Phase 1: System Preparation & Documentation Setup (2 hours)."""
        phase_start = self.logger.start_phase(
            "Phase 1: System Preparation",
            "System validation, test runs, and infrastructure setup"
        )
        
        success = True
        
        try:
            # Step 1.1: System health check
            self.logger.log_info("Running system health check", "phase1")
            if not self.runner.test_system_health():
                self.logger.log_warning("System health check failed, continuing anyway", "phase1")
            
            # Step 1.2: Prepare test corpora
            self.logger.log_info("Preparing test corpora (100 docs each)", "phase1")
            
            # Create test subsets
            success &= self.runner.prepare_test_corpus(
                "fiqa_test", 
                "technical/fiqa/corpus.jsonl", 
                max_docs=100
            )
            
            success &= self.runner.prepare_test_corpus(
                "scifact_test", 
                "narrative/scifact/corpus.jsonl", 
                max_docs=100
            )
            
            # Step 1.3: Test ingestion
            if success:
                self.logger.log_info("Running test ingestion", "phase1")
                test_source = self.base_dir / "corpus" / "test" / "fiqa_test"
                metrics = self.runner.ingest_collection(
                    "fiqa_test", 
                    str(test_source), 
                    chunk_size=512, 
                    chunk_overlap=128, 
                    test_mode=True
                )
                success &= metrics.success
            
            # Step 1.4: Test experiments
            if success:
                self.logger.log_info("Running pipeline validation tests", "phase1")
                success &= self.runner.run_test_experiments()
            
            # Cleanup test data
            self.runner.cleanup_test_data()
            
        except Exception as e:
            self.logger.log_error(e, "phase1_execution")
            success = False
        
        self.logger.end_phase("Phase 1: System Preparation", phase_start)
        return success
    
    def run_phase_2_ingestion(self) -> bool:
        """Phase 2: Full Corpus Ingestion (3-4 hours)."""
        phase_start = self.logger.start_phase(
            "Phase 2: Full Corpus Ingestion",
            "Ingest complete FIQA and SciFact collections with optimized chunking"
        )
        
        success = True
        
        try:
            # Step 2.1: FIQA Technical Collection
            self.logger.log_info("Starting FIQA collection ingestion (Est: 90-120 min)", "phase2")
            fiqa_source = self.base_dir / "corpus" / "technical" / "fiqa"
            
            fiqa_metrics = self.runner.ingest_collection(
                "fiqa_technical",
                str(fiqa_source),
                chunk_size=512,  # Default for comparison
                chunk_overlap=128
            )
            success &= fiqa_metrics.success
            
            if fiqa_metrics.success:
                self.logger.log_info(
                    f"FIQA ingestion complete: {fiqa_metrics.documents_processed} docs, "
                    f"{fiqa_metrics.chunks_created} chunks in {fiqa_metrics.processing_time_seconds:.1f}s",
                    "phase2"
                )
            
            # Brief pause between ingestions
            time.sleep(30)
            
            # Step 2.2: SciFact Scientific Collection
            self.logger.log_info("Starting SciFact collection ingestion (Est: 90-120 min)", "phase2")
            scifact_source = self.base_dir / "corpus" / "narrative" / "scifact"
            
            scifact_metrics = self.runner.ingest_collection(
                "scifact_scientific",
                str(scifact_source),
                chunk_size=512,
                chunk_overlap=128
            )
            success &= scifact_metrics.success
            
            if scifact_metrics.success:
                self.logger.log_info(
                    f"SciFact ingestion complete: {scifact_metrics.documents_processed} docs, "
                    f"{scifact_metrics.chunks_created} chunks in {scifact_metrics.processing_time_seconds:.1f}s",
                    "phase2"
                )
            
        except Exception as e:
            self.logger.log_error(e, "phase2_execution")
            success = False
        
        self.logger.end_phase("Phase 2: Full Corpus Ingestion", phase_start)
        return success
    
    def run_phase_3_baseline(self) -> bool:
        """Phase 3: Baseline Establishment (1 hour)."""
        phase_start = self.logger.start_phase(
            "Phase 3: Baseline Establishment",
            "Establish baseline measurements with current configuration"
        )
        
        success = True
        
        try:
            # Run baseline with default configuration (512 tokens, 128 overlap)
            baseline_queries = self.queries[:10]  # Use first 10 queries for baseline
            collections = ["fiqa_technical", "scifact_scientific"]
            
            total_baseline_experiments = len(baseline_queries) * len(collections)
            self.logger.log_info(f"Running {total_baseline_experiments} baseline experiments", "phase3")
            
            all_results = []
            for collection in collections:
                self.logger.log_info(f"Running baseline on {collection}", "phase3")
                results = self.runner.run_parameter_sweep(collection, baseline_queries)
                all_results.extend(results)
            
            successful_experiments = sum(1 for r in all_results if r.success)
            success_rate = successful_experiments / len(all_results) * 100
            
            self.logger.log_info(
                f"Baseline complete: {successful_experiments}/{len(all_results)} successful ({success_rate:.1f}%)",
                "phase3"
            )
            
            success = success_rate >= 80.0  # Require 80% success rate
            
        except Exception as e:
            self.logger.log_error(e, "phase3_execution")
            success = False
        
        self.logger.end_phase("Phase 3: Baseline Establishment", phase_start)
        return success
    
    def run_phase_4_parameter_sweep(self) -> bool:
        """Phase 4: Parameter Sweep Experiments (6-8 hours)."""
        phase_start = self.logger.start_phase(
            "Phase 4: Parameter Sweep",
            "Complete parameter sweep across chunk sizes and overlap ratios"
        )
        
        success = True
        
        try:
            collections = ["fiqa_technical", "scifact_scientific"]
            
            # Calculate total experiments
            total_configs = 0
            for chunk_size in self.runner.chunk_sizes:
                for overlap in self.runner.overlap_ratios:
                    if overlap < chunk_size:  # Valid combinations only
                        total_configs += 1
            
            total_experiments = total_configs * len(self.queries) * len(collections)
            
            self.logger.log_info(
                f"Starting parameter sweep: {total_experiments} total experiments "
                f"({total_configs} configs × {len(self.queries)} queries × {len(collections)} collections)",
                "phase4"
            )
            
            # Estimate time
            estimated_time_hours = total_experiments * 45 / 3600  # 45 seconds per experiment
            self.logger.log_info(f"Estimated completion time: {estimated_time_hours:.1f} hours", "phase4")
            
            all_results = []
            
            # Run experiments for each collection
            for collection in collections:
                self.logger.log_info(f"Starting parameter sweep for {collection}", "phase4")
                collection_start = time.time()
                
                results = self.runner.run_parameter_sweep(collection, self.queries)
                all_results.extend(results)
                
                collection_time = time.time() - collection_start
                successful_in_collection = sum(1 for r in results if r.success)
                
                self.logger.log_info(
                    f"Collection {collection} complete: {successful_in_collection}/{len(results)} successful "
                    f"in {collection_time/3600:.2f} hours",
                    "phase4"
                )
                
                # Brief pause between collections
                time.sleep(60)
            
            # Overall success metrics
            successful_experiments = sum(1 for r in all_results if r.success)
            success_rate = successful_experiments / len(all_results) * 100
            
            self.logger.log_info(
                f"Parameter sweep complete: {successful_experiments}/{len(all_results)} successful ({success_rate:.1f}%)",
                "phase4"
            )
            
            success = success_rate >= 70.0  # Allow lower threshold for large experiment
            
        except Exception as e:
            self.logger.log_error(e, "phase4_execution")
            success = False
        
        self.logger.end_phase("Phase 4: Parameter Sweep", phase_start)
        return success
    
    def run_phase_5_analysis(self) -> bool:
        """Phase 5: Analysis & Documentation (2 hours)."""
        phase_start = self.logger.start_phase(
            "Phase 5: Analysis & Documentation",
            "Statistical analysis, visualization, and final reporting"
        )
        
        success = True
        
        try:
            # Initialize analyzer
            analyzer = ChunkingAnalyzer(str(self.runner.logger.log_dir))
            
            # Load experiment data
            self.logger.log_info("Loading experiment data for analysis", "phase5")
            df = analyzer.load_experiment_data(self.runner.logger.session_id)
            
            # Generate comprehensive analysis
            self.logger.log_info("Generating comprehensive analysis report", "phase5")
            report_file = analyzer.generate_comprehensive_report(df)
            
            self.logger.log_info(f"Analysis complete! Report saved to: {report_file}", "phase5")
            
            # Log key findings
            summary = analyzer.generate_summary_statistics(df)
            optimal_configs = analyzer.find_optimal_configurations(df)
            
            if optimal_configs:
                best = optimal_configs[0]
                self.logger.log_info(
                    f"Optimal configuration found: {best.chunk_size} tokens, {best.chunk_overlap} overlap "
                    f"({best.metric_value:.3f}s avg response time)",
                    "phase5"
                )
            
            self.logger.log_info(
                f"Experiment completed successfully: {summary['total_experiments']} experiments analyzed",
                "phase5"
            )
            
        except Exception as e:
            self.logger.log_error(e, "phase5_execution")
            success = False
        
        self.logger.end_phase("Phase 5: Analysis & Documentation", phase_start)
        return success
    
    def run_complete_experiment(self) -> bool:
        """Run the complete experimental pipeline."""
        total_start = time.time()
        
        self.logger.start_experiment(
            total_experiments=1000,  # Estimated total
            description="Complete chunking strategy optimization experiment pipeline"
        )
        
        phases = [
            ("Phase 1: System Preparation", self.run_phase_1_preparation),
            ("Phase 2: Full Corpus Ingestion", self.run_phase_2_ingestion),
            ("Phase 3: Baseline Establishment", self.run_phase_3_baseline),
            ("Phase 4: Parameter Sweep", self.run_phase_4_parameter_sweep),
            ("Phase 5: Analysis & Documentation", self.run_phase_5_analysis)
        ]
        
        overall_success = True
        completed_phases = 0
        
        for phase_name, phase_function in phases:
            self.logger.log_info(f"Starting {phase_name}", "orchestrator")
            
            try:
                phase_success = phase_function()
                
                if phase_success:
                    completed_phases += 1
                    self.logger.log_info(f"✓ {phase_name} completed successfully", "orchestrator")
                else:
                    overall_success = False
                    self.logger.log_warning(f"✗ {phase_name} failed or completed with issues", "orchestrator")
                    
                    # Ask user if they want to continue
                    user_input = input(f"\n{phase_name} had issues. Continue to next phase? (y/n): ")
                    if user_input.lower() != 'y':
                        self.logger.log_info("Experiment stopped by user request", "orchestrator")
                        break
                        
            except KeyboardInterrupt:
                self.logger.log_info("Experiment interrupted by user", "orchestrator")
                break
            except Exception as e:
                self.logger.log_error(e, f"orchestrator_{phase_name}")
                overall_success = False
                break
        
        total_time = time.time() - total_start
        
        self.logger.log_info(
            f"Experiment pipeline complete: {completed_phases}/{len(phases)} phases successful "
            f"in {total_time/3600:.2f} hours",
            "orchestrator"
        )
        
        self.logger.end_experiment()
        return overall_success

def main():
    """Main entry point for the experiment orchestrator."""
    print("=" * 80)
    print("Document Chunking Strategy Optimization Experiment")
    print("=" * 80)
    print()
    
    orchestrator = ExperimentOrchestrator()
    
    print("Starting complete experimental pipeline...")
    print("This will take approximately 14-18 hours to complete.")
    print()
    
    user_input = input("Proceed with full experiment? (y/n): ")
    if user_input.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    try:
        success = orchestrator.run_complete_experiment()
        
        print("\n" + "=" * 80)
        if success:
            print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
        else:
            print("✗ EXPERIMENT COMPLETED WITH ISSUES")
        print("=" * 80)
        
        print(f"\nLogs and results available in:")
        print(f"- Main log: {orchestrator.logger.log_file}")
        print(f"- Metrics: {orchestrator.logger.metrics_file}")
        print(f"- System metrics: {orchestrator.logger.system_metrics_file}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        orchestrator.logger.log_error(e, "main_execution")

if __name__ == "__main__":
    main()