"""
Open All Visualizations

This script opens all visualization images so you can view them.
"""

import os
import subprocess
import platform

print("=" * 70)
print("Opening All Visualizations")
print("=" * 70)

viz_dir = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/visualizations"

# List all PNG files
png_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
png_files.sort()

print(f"\nFound {len(png_files)} visualization files:")
for i, file in enumerate(png_files, 1):
    print(f"   {i}. {file}")

print("\nOpening files...")

# Open files based on OS
system = platform.system()
for file in png_files:
    file_path = os.path.join(viz_dir, file)
    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['open', file_path])
        elif system == 'Windows':
            subprocess.run(['start', file_path], shell=True)
        else:  # Linux
            subprocess.run(['xdg-open', file_path])
        print(f"   Opened: {file}")
    except Exception as e:
        print(f"   Could not open {file}: {e}")

print("\n" + "=" * 70)
print("All visualizations should now be open!")
print("=" * 70)
print("\nTo view them manually, go to:")
print(f"   {viz_dir}")

