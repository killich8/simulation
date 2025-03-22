#!/usr/bin/env python

"""
Road Object Detection - Data Processing Workflow
This script defines the overall data processing workflow for the project
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DataProcessing")

class DataProcessingPipeline:
    """
    Main class for the data processing pipeline
    """
    def __init__(self, config_path):
        """Initialize the data processing pipeline"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up paths
        self.input_dir = self.config.get('input_dir', './input')
        self.output_dir = self.config.get('output_dir', './output')
        self.temp_dir = self.config.get('temp_dir', './temp')
        
        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize processing modules
        self._init_modules()
        
        logger.info(f"Data processing pipeline initialized with config: {config_path}")
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _init_modules(self):
        """Initialize processing modules"""
        # Import modules here to avoid circular imports
        from data_processing.validation.data_validator import DataValidator
        from data_processing.transformation.data_transformer import DataTransformer
        from data_processing.augmentation.data_augmenter import DataAugmenter
        
        # Initialize modules with configuration
        self.validator = DataValidator(self.config.get('validation', {}))
        self.transformer = DataTransformer(self.config.get('transformation', {}))
        self.augmenter = DataAugmenter(self.config.get('augmentation', {}))
        
        logger.info("Processing modules initialized")
    
    def run(self):
        """Run the complete data processing pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting data processing pipeline at {start_time}")
        
        try:
            # Step 1: Validate input data
            logger.info("Step 1: Validating input data")
            validation_results = self.validator.validate(self.input_dir)
            if not validation_results['success']:
                logger.error(f"Data validation failed: {validation_results['message']}")
                return False
            
            # Step 2: Transform data
            logger.info("Step 2: Transforming data")
            transformation_results = self.transformer.transform(
                self.input_dir, 
                os.path.join(self.temp_dir, 'transformed')
            )
            if not transformation_results['success']:
                logger.error(f"Data transformation failed: {transformation_results['message']}")
                return False
            
            # Step 3: Augment data
            logger.info("Step 3: Augmenting data")
            augmentation_results = self.augmenter.augment(
                os.path.join(self.temp_dir, 'transformed'),
                self.output_dir
            )
            if not augmentation_results['success']:
                logger.error(f"Data augmentation failed: {augmentation_results['message']}")
                return False
            
            # Step 4: Generate statistics and reports
            logger.info("Step 4: Generating statistics and reports")
            self._generate_reports()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Data processing completed successfully in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            return False
    
    def _generate_reports(self):
        """Generate statistics and reports about the processed data"""
        # Create reports directory
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate dataset statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'input_data': self._count_files(self.input_dir),
            'output_data': self._count_files(self.output_dir),
            'validation_results': self.validator.get_statistics(),
            'transformation_results': self.transformer.get_statistics(),
            'augmentation_results': self.augmenter.get_statistics()
        }
        
        # Save statistics to file
        with open(os.path.join(reports_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Generated statistics report: {os.path.join(reports_dir, 'statistics.json')}")
    
    def _count_files(self, directory):
        """Count files in directory by type"""
        file_counts = {
            'images': 0,
            'annotations': 0,
            'other': 0
        }
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_counts['images'] += 1
                elif file.lower().endswith(('.json', '.xml', '.txt')):
                    file_counts['annotations'] += 1
                else:
                    file_counts['other'] += 1
        
        file_counts['total'] = sum(file_counts.values())
        return file_counts


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = DataProcessingPipeline(args.config)
    success = pipeline.run()
    
    if success:
        logger.info("Pipeline completed successfully")
        return 0
    else:
        logger.error("Pipeline failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
