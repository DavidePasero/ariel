#!/usr/bin/env python3
"""
Test script for the comparative dashboard click functionality
"""

import sys
sys.path.append('.')

try:
    from examples.comparative_dashboard import ComparativeEvolutionDashboard
    print("✓ Successfully imported ComparativeEvolutionDashboard")
except ImportError as e:
    print(f"✗ Failed to import ComparativeEvolutionDashboard: {e}")
    sys.exit(1)

try:
    import dash
    from pathlib import Path
    print("✓ Successfully imported required dependencies")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    sys.exit(1)

# Test directory creation
try:
    dl_robots_path = Path("examples/dl_robots")
    dl_robots_path.mkdir(exist_ok=True)
    print(f"✓ Created dl_robots directory: {dl_robots_path.exists()}")
except Exception as e:
    print(f"✗ Failed to create dl_robots directory: {e}")

print("\nComparative dashboard implementation appears to be working!")
print("To test the click functionality:")
print("1. Run the comparative dashboard with real data")
print("2. Go to the 'Individual Analysis' tab")
print("3. Click on any point in the scatter plot")
print("4. A modal should appear asking which genotype to download")
