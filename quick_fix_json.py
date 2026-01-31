#!/usr/bin/env python3
"""
Quick fix for the float32 JSON serialization error
This modifies local_backend.py to handle numpy types properly
"""

import os
import shutil

def apply_fix():
    filepath = 'local_backend.py'
    
    if not os.path.exists(filepath):
        print("❌ Error: local_backend.py not found!")
        print("Make sure you're in the CourtCheck directory")
        return False
    
    # Backup
    backup = filepath + '.before_json_fix'
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"📦 Backup created: {backup}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'convert_to_serializable' in content:
        print("✓ Already fixed!")
        return True
    
    # Add the conversion function right after imports
    conversion_function = '''
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types"""
    if isinstance(obj, dict):
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
    return obj

'''
    
    # Find where to insert (after imports, before app = FastAPI())
    insert_pos = content.find('app = FastAPI()')
    if insert_pos == -1:
        print("❌ Could not find 'app = FastAPI()' in file")
        return False
    
    # Insert the function
    content = content[:insert_pos] + conversion_function + content[insert_pos:]
    
    # Fix the specific error location (line 276: get_status function)
    # Replace JSONResponse({ with JSONResponse(convert_to_serializable({
    content = content.replace(
        'return JSONResponse({',
        'return JSONResponse(convert_to_serializable({'
    )
    
    # Also fix any other JSONResponse calls
    content = content.replace(
        'JSONResponse({',
        'JSONResponse(convert_to_serializable({'
    )
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed JSON serialization!")
    print("\nWhat was changed:")
    print("  • Added convert_to_serializable() function")
    print("  • Wrapped all JSONResponse calls with type conversion")
    print("  • This converts numpy float32/int32 to Python float/int")
    
    return True

def main():
    print("="*60)
    print("Fixing: TypeError: Object of type float32 is not JSON serializable")
    print("="*60)
    print()
    
    if apply_fix():
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Stop your backend (CTRL+C if running)")
        print("  2. Restart: python local_backend.py")
        print("  3. Try uploading a video again")
        print("\nThe error should be fixed!")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
