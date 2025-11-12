#!/usr/bin/env python3
"""
Test script to verify configuration loading works correctly.
"""

from pathlib import Path
import json
from rich.console import Console

console = Console()

def test_config_loading():
    """Test that saved configurations can be loaded correctly."""
    
    # Look for database files in __data__ directory
    data_dir = Path("__data__")
    if not data_dir.exists():
        console.log("No __data__ directory found. Run evolve_headless.py first.")
        return
    
    db_files = list(data_dir.glob("*.db"))
    if not db_files:
        console.log("No database files found in __data__. Run evolve_headless.py first.")
        return
    
    console.log(f"Found {len(db_files)} database files:")
    
    for db_file in db_files:
        console.log(f"\n--- Testing {db_file.name} ---")
        
        # Check for corresponding config file
        config_file = db_file.parent / (db_file.stem + "_config.json")
        
        if config_file.exists():
            console.log(f"✅ Config file found: {config_file.name}")
            
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Display key information
                metadata = config_data.get("metadata", {})
                resolved = config_data.get("resolved_settings", {})
                
                console.log(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
                console.log(f"  Genotype: {resolved.get('genotype_name', 'unknown')}")
                console.log(f"  Task: {resolved.get('task', 'unknown')}")
                console.log(f"  Generations: {resolved.get('num_of_generations', 'unknown')}")
                console.log(f"  Population size: {resolved.get('target_population_size', 'unknown')}")
                console.log(f"  Mutation: {resolved.get('mutation_name', 'unknown')}")
                console.log(f"  Crossover: {resolved.get('crossover_name', 'unknown')}")
                
                # Test that we can load the genotype
                from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
                genotype_name = resolved.get('genotype_name')
                if genotype_name and genotype_name in GENOTYPES_MAPPING:
                    genotype_class = GENOTYPES_MAPPING[genotype_name]
                    console.log(f"  ✅ Genotype class loaded: {genotype_class.__name__}")
                else:
                    console.log(f"  ❌ Could not load genotype: {genotype_name}")
                
            except Exception as e:
                console.log(f"  ❌ Error loading config: {e}")
        else:
            console.log(f"❌ No config file found: {config_file.name}")
            console.log("   This database was created before the config saving feature.")

if __name__ == "__main__":
    test_config_loading()
