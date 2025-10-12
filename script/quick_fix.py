#!/usr/bin/env python
"""
quick_fix.py

Quick fix for variable name errors in music_analysis_parallel.py
================================================================

Run this in the script/ directory to fix the OUTPUT_FILE error.
"""

from pathlib import Path
import shutil

def fix_music_analysis():
    """Fix OUTPUT_FILE variable name error."""
    script_file = Path('music_analysis_parallel.py')
    
    if not script_file.exists():
        print("✗ Error: music_analysis_parallel.py not found")
        print("Make sure you're in the script/ directory")
        return False
    
    print("Fixing music_analysis_parallel.py...")
    
    # Backup
    backup_file = script_file.with_suffix('.py.backup')
    shutil.copy2(script_file, backup_file)
    print(f"  ✓ Backup created: {backup_file.name}")
    
    # Read file
    content = script_file.read_text()
    
    # Fix variable name
    content = content.replace('OUTPUT_FILE', 'OUTPUT_PKL')
    
    # Also ensure OUTPUT_PKL is defined at the top
    if 'OUTPUT_PKL' not in content.split('OUTPUT_TXT')[0]:
        # Need to add the variable definition
        old_line = "OUTPUT_FILE = STATS_DIR / 'music_analysis_results.pkl'"
        new_lines = """OUTPUT_PKL = STATS_DIR / 'music_analysis_results.pkl'
OUTPUT_TXT = STATS_DIR / 'music_analysis_results.txt'
OUTPUT_CSV = STATS_DIR / 'music_analysis_results.csv'"""
        
        if old_line in content:
            content = content.replace(old_line, new_lines)
            print("  ✓ Added output file variables")
    
    # Write back
    script_file.write_text(content)
    print("  ✓ Fixed variable names")
    
    print("\n✓ All fixes applied!")
    print("\nNow run:")
    print("  python music_analysis_parallel.py")
    
    return True


if __name__ == "__main__":
    import sys
    try:
        success = fix_music_analysis()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
