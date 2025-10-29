#!/usr/bin/env python3
"""
Simple test script for morphology analysis functionality.
"""

import numpy as np
from src.ariel.utils.graph_ops import load_robot_json_file
from src.ariel.utils.morphological_descriptor import MorphologicalMeasures

def test_6d_descriptor():
    """Test computing 6D morphological descriptor."""
    print("Testing 6D descriptor computation...")
    
    # Test with one of our target robots
    json_path = "examples/target_robots/small_robot_8.json"
    
    try:
        # Load robot
        robot_graph = load_robot_json_file(json_path)
        print(f"Loaded robot with {robot_graph.number_of_nodes()} nodes")
        
        # Compute measures
        measures = MorphologicalMeasures(robot_graph)
        
        # Test individual properties
        print(f"B (Branching): {measures.B}")
        print(f"L (Limbs): {measures.L}")
        print(f"E (Extensiveness): {measures.E}")
        print(f"S (Symmetry): {measures.S}")
        print(f"P (Proportion): {measures.P}")
        print(f"J (Joints): {measures.J}")
        print(f"Is 2D: {measures.is_2d}")
        
        # Compute 6D descriptor
        descriptor = np.array([measures.B, measures.L, measures.E, measures.S, measures.P, measures.J])
        print(f"6D descriptor: {descriptor}")
        
        print("✓ 6D descriptor computation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_all_targets():
    """Test all target robots."""
    print("\nTesting all target robots...")
    
    target_paths = [
        "examples/target_robots/small_robot_8.json",
        "examples/target_robots/medium_robot_15.json", 
        "examples/target_robots/large_robot_25.json"
    ]
    
    for json_path in target_paths:
        print(f"\n--- Testing {json_path} ---")
        try:
            robot_graph = load_robot_json_file(json_path)
            measures = MorphologicalMeasures(robot_graph)
            
            descriptor = np.array([measures.B, measures.L, measures.E, measures.S, measures.P, measures.J])
            print(f"Modules: {measures.num_modules}")
            print(f"Descriptor: {descriptor}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    success = test_6d_descriptor()
    if success:
        test_all_targets()
    else:
        print("Basic test failed, skipping comprehensive test.")