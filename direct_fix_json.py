#!/usr/bin/env python3
"""
Direct fix for the JSON serialization error
Manually patches the get_status function in local_backend.py
"""

import os
import shutil
import re

def show_problem_area():
    """Show the problematic code"""
    filepath = 'local_backend.py'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the get_status function
    for i, line in enumerate(lines):
        if 'async def get_status' in line:
            print(f"\nFound get_status function at line {i+1}")
            print("\nCurrent code (lines around the error):")
            print("="*60)
            for j in range(max(0, i-2), min(len(lines), i+25)):
                print(f"{j+1:4d}: {lines[j]}", end='')
            print("="*60)
            return i
    
    return None

def apply_direct_fix():
    """Directly modify the get_status function"""
    filepath = 'local_backend.py'
    
    if not os.path.exists(filepath):
        print("❌ Error: local_backend.py not found!")
        return False
    
    # Backup
    backup = filepath + '.before_direct_fix'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"📦 Backup created: {backup}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First, ensure numpy is imported
    if 'import numpy as np' not in content:
        # Add numpy import at the top
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import') or line.strip().startswith('from'):
                continue
            else:
                lines.insert(i, 'import numpy as np')
                break
        content = '\n'.join(lines)
        print("✓ Added: import numpy as np")
    
    # Add the conversion function if not present
    if 'def convert_to_serializable' not in content:
        conversion_func = '''
def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types"""
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

'''
        # Insert before the first route decorator
        insert_pos = content.find('@app.')
        if insert_pos != -1:
            content = content[:insert_pos] + conversion_func + content[insert_pos:]
            print("✓ Added: convert_to_serializable function")
    
    # Now fix the get_status function specifically
    # Find and replace the problematic return statement
    
    # Pattern 1: Multi-line JSONResponse
    old_pattern1 = r'(@app\.get\("/api/status/\{video_id\}"\)[^\n]*\nasync def get_status[^:]*:.*?)(return JSONResponse\(\{)'
    
    # Look for the get_status function and its return
    get_status_match = re.search(
        r'async def get_status\(video_id: str\):.*?return JSONResponse\(\{.*?\}\)',
        content,
        re.DOTALL
    )
    
    if get_status_match:
        old_func = get_status_match.group(0)
        
        # Replace just the return statement
        new_func = old_func.replace(
            'return JSONResponse({',
            'return JSONResponse(content=convert_to_serializable({'
        )
        
        # Make sure the closing is correct
        # Count braces to find where to close the convert_to_serializable
        if new_func != old_func:
            content = content.replace(old_func, new_func)
            print("✓ Fixed: get_status function return statement")
    else:
        # Fallback: replace all JSONResponse calls
        print("⚠ Could not find exact pattern, applying general fix...")
        content = re.sub(
            r'return JSONResponse\(\{',
            r'return JSONResponse(content=convert_to_serializable({',
            content
        )
        print("✓ Fixed: All JSONResponse calls")
    
    # Write the modified content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def verify_fix():
    """Verify the fix was applied"""
    filepath = 'local_backend.py'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'Numpy import': 'import numpy as np' in content,
        'Conversion function': 'def convert_to_serializable' in content,
        'Fixed get_status': 'convert_to_serializable' in content and 'get_status' in content,
    }
    
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    
    all_good = True
    for check, passed in checks.items():
        if passed:
            print(f"✓ {check}")
        else:
            print(f"✗ {check}")
            all_good = False
    
    return all_good

def main():
    print("="*60)
    print("Direct Fix for JSON Serialization Error")
    print("="*60)
    
    if not os.path.exists('local_backend.py'):
        print("\n❌ Error: Not in CourtCheck directory!")
        return 1
    
    # Show the problem
    print("\nAnalyzing the problem...")
    show_problem_area()
    
    # Apply fix
    print("\n" + "="*60)
    print("Applying Fix...")
    print("="*60)
    
    if apply_direct_fix():
        print("\n" + "="*60)
        print("Fix Applied!")
        print("="*60)
        
        if verify_fix():
            print("\n✅ All checks passed!")
            print("\nNext steps:")
            print("  1. Restart backend: python local_backend.py")
            print("  2. Upload a video")
            print("  3. The error should be gone!")
            
            # Show what was changed
            print("\n" + "="*60)
            print("What changed:")
            print("="*60)
            print("BEFORE:")
            print("  return JSONResponse({")
            print("      'status': status,")
            print("      'progress': np.float32(0.5)  # ❌ Not serializable")
            print("  })")
            print("\nAFTER:")
            print("  return JSONResponse(content=convert_to_serializable({")
            print("      'status': status,")
            print("      'progress': np.float32(0.5)  # ✅ Converted to float")
            print("  }))")
            
            return 0
        else:
            print("\n⚠ Fix applied but verification failed")
            print("Check the code manually")
            return 1
    else:
        print("\n❌ Fix failed")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
