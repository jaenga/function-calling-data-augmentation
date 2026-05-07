"""
Main orchestration script for mission data augmentation & validation pipeline
Modes:
  - single: Augment and validate single-turn data only
  - multi: Augment and validate multi-turn data only
  - all: Full pipeline (single + multi)
  - export: Export validated data to JSONL format only
"""

import sys
import argparse
import logging
from pathlib import Path

# Import pipeline modules
from augment import GapAnalyzer, AugmentationEngine
from validate import ValidationPipeline
from export import ExportPipeline
from analyze import AnalysisEngine

from config import DATA_DIR, OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate the entire augmentation and validation pipeline"""
    
    def __init__(self):
        self.step_count = 0
    
    def _log_step(self, step_name: str, description: str) -> None:
        """Log pipeline step"""
        self.step_count += 1
        logger.info("\n" + "="*80)
        logger.info(f"STEP {self.step_count}: {step_name}")
        logger.info(f"  {description}")
        logger.info("="*80)
    
    def augment_pipeline(self, data_type: str) -> None:
        """
        Run augmentation pipeline
        data_type: "single", "multi", or "all"
        """
        self._log_step(
            "AUGMENTATION",
            f"Generate new utterances using OpenAI (data_type: {data_type})"
        )
        
        try:
            from config import SEED_SINGLE_PATH, SEED_MULTI_PATH
            
            # Gap Analysis
            logger.info("\n🔍 Gap Analysis...")
            analyzer = GapAnalyzer(SEED_SINGLE_PATH, SEED_MULTI_PATH)
            logger.info("\n🤖 Generating data with OpenAI...")
            
            if data_type in ["single", "all"]:
                logger.info("\n" + "-"*80)
                logger.info("AUGMENTING SINGLE-TURN DATA")
                logger.info("-"*80)
                gap_analysis = analyzer.analyze()
                engine = AugmentationEngine(gap_analysis, analyzer.seed_single_df, analyzer.seed_multi_df)
                engine.augment(data_type="single")
            
            if data_type in ["multi", "all"]:
                logger.info("\n" + "-"*80)
                logger.info("AUGMENTING MULTI-TURN DATA")
                logger.info("-"*80)
                gap_analysis = analyzer.analyze_multi()
                engine = AugmentationEngine(gap_analysis, analyzer.seed_single_df, analyzer.seed_multi_df)
                engine.augment(data_type="multi")
            
            logger.info("\n✅ Augmentation pipeline completed!")
            
        except Exception as e:
            logger.error(f"❌ Augmentation pipeline failed: {str(e)}")
            raise
    
    def validate_pipeline(self, data_type: str) -> None:
        """
        Run validation pipeline
        data_type: "single", "multi", or "all"
        """
        self._log_step(
            "VALIDATION",
            f"2-stage validation using OpenAI (data_type: {data_type})"
        )
        
        try:
            logger.info("\n🔎 Validating generated data...")
            pipeline = ValidationPipeline()
            
            if data_type in ["single", "all"]:
                logger.info("\n" + "-"*80)
                logger.info("VALIDATING SINGLE-TURN DATA")
                logger.info("-"*80)
                pipeline.validate(data_type="single")
            
            if data_type in ["multi", "all"]:
                logger.info("\n" + "-"*80)
                logger.info("VALIDATING MULTI-TURN DATA")
                logger.info("-"*80)
                pipeline.validate(data_type="multi")
            
            logger.info("\n✅ Validation pipeline completed!")
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline failed: {str(e)}")
            raise
    
    def export_pipeline(self, format: str = "qwen") -> None:
        """
        Run export pipeline
        format: "qwen", "functiongemma", or "all"
        """
        self._log_step(
            "EXPORT",
            f"Convert validated data to JSONL format (format: {format})"
        )
        
        try:
            logger.info("\n💾 Exporting validated data...")
            pipeline = ExportPipeline()
            pipeline.export(format=format)
            
            logger.info("\n✅ Export pipeline completed!")
            
        except Exception as e:
            logger.error(f"❌ Export pipeline failed: {str(e)}")
            raise
    
    def analyze_pipeline(self, save_report: bool = True) -> None:
        """
        Run analysis pipeline
        save_report: whether to save analysis report to file
        """
        self._log_step(
            "ANALYSIS",
            "Generate analysis report on augmentation and validation results"
        )
        
        try:
            logger.info("\n📊 Analyzing results...")
            engine = AnalysisEngine()
            engine.generate_report()
            engine.print_report()
            
            if save_report:
                engine.save_report()
            
            logger.info("\n✅ Analysis completed!")
            
        except Exception as e:
            logger.error(f"❌ Analysis pipeline failed: {str(e)}")
            raise
    
    def run_full_pipeline(self, data_type: str = "all") -> None:
        """Run complete pipeline: augment → validate → export → analyze"""
        logger.info("\n" + "🚀 "*40)
        logger.info("STARTING FULL PIPELINE")
        logger.info("🚀 "*40)
        
        try:
            # Step 1: Augmentation
            self.augment_pipeline(data_type=data_type)
            
            # Step 2: Validation
            self.validate_pipeline(data_type=data_type)
            
            # Step 3: Export
            self.export_pipeline(format="qwen")
            
            # Step 4: Analysis
            self.analyze_pipeline(save_report=True)
            
            logger.info("\n" + "✨ "*40)
            logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("✨ "*40)
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {str(e)}")
            sys.exit(1)
    
    def run_export_only(self, format: str = "qwen") -> None:
        """Run export pipeline only (for quick JSONL generation)"""
        logger.info("\n" + "🚀 "*40)
        logger.info("STARTING EXPORT ONLY")
        logger.info("🚀 "*40)
        
        try:
            self.export_pipeline(format=format)
            self.analyze_pipeline(save_report=True)
            
            logger.info("\n" + "✨ "*40)
            logger.info("EXPORT COMPLETED SUCCESSFULLY!")
            logger.info("✨ "*40)
            
        except Exception as e:
            logger.error(f"\n❌ Export failed: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mission data augmentation & validation pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode single        # Augment & validate single-turn data
  python run.py --mode multi         # Augment & validate multi-turn data
  python run.py --mode all           # Full pipeline (single + multi)
  python run.py --mode export        # Export validated data to JSONL only
  python run.py --mode export --format qwen  # Export in Qwen format
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "all", "export"],
        default="all",
        help="Pipeline mode (default: all)"
    )
    
    parser.add_argument(
        "--format",
        choices=["qwen", "functiongemma", "all"],
        default="qwen",
        help="Export format for --mode export (default: qwen)"
    )
    
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis report generation"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Run pipeline based on mode
    if args.mode == "export":
        orchestrator.run_export_only(format=args.format)
    else:
        orchestrator.run_full_pipeline(data_type=args.mode)


if __name__ == "__main__":
    main()
