#!/usr/bin/env python3
"""
CourtCheck Automated Fix Script
Automatically fixes common issues preventing the model from running.
"""

import os
import sys
import re
import shutil
from pathlib import Path

class Fixer:
    def __init__(self):
        self.fixes_applied = []
        self.errors = []
    
    def backup_file(self, filepath):
        """Create backup of file before modifying"""
        backup_path = f"{filepath}.backup"
        if os.path.exists(filepath) and not os.path.exists(backup_path):
            shutil.copy2(filepath, backup_path)
            print(f"  📦 Backed up: {filepath} -> {backup_path}")
    
    def fix_device_configuration(self, filepath):
        """Fix CUDA/CPU device configuration issues"""
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes = []
        
        # Fix 1: Force CPU mode for compatibility
        pattern1 = r'device\s*=\s*["\']cuda["\']'
        if re.search(pattern1, content):
            content = re.sub(pattern1, 'device = "cpu"', content)
            fixes.append("Forced CPU mode")
        
        # Fix 2: Add map_location to torch.load
        pattern2 = r'torch\.load\s*\(\s*([^,\)]+)\s*\)'
        matches = re.finditer(pattern2, content)
        for match in matches:
            if 'map_location' not in match.group(0):
                old_call = match.group(0)
                new_call = f'torch.load({match.group(1)}, map_location="cpu")'
                content = content.replace(old_call, new_call)
                fixes.append("Added map_location to torch.load()")
        
        # Fix 3: Add device specification to model.to()
        pattern3 = r'model\.to\s*\(\s*\)'
        if re.search(pattern3, content) and 'device' in content:
            content = re.sub(pattern3, 'model.to(device)', content)
            fixes.append("Fixed model.to() device specification")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.fixes_applied.append(f"{filepath}: {', '.join(fixes)}")
            return True
        
        return False
    
    def fix_import_errors(self, filepath):
        """Add missing imports and fix import issues"""
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = lines.copy()
        
        # Check for common missing imports
        has_os = any('import os' in line for line in lines)
        has_sys = any('import sys' in line for line in lines)
        has_pathlib = any('from pathlib import Path' in line or 'import pathlib' in line for line in lines)
        
        uses_os = any('os.' in line for line in lines)
        uses_sys = any('sys.' in line for line in lines)
        uses_path = any('Path(' in line for line in lines)
        
        # Add missing imports at the beginning (after docstring)
        import_index = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                import_index = i
                break
        
        imports_to_add = []
        if uses_os and not has_os:
            imports_to_add.append('import os\n')
        if uses_sys and not has_sys:
            imports_to_add.append('import sys\n')
        if uses_path and not has_pathlib:
            imports_to_add.append('from pathlib import Path\n')
        
        if imports_to_add:
            lines[import_index:import_index] = imports_to_add
            self.fixes_applied.append(f"{filepath}: Added missing imports")
        
        if lines != original_lines:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
        return False
    
    def fix_path_issues(self, filepath):
        """Fix hardcoded paths and path construction"""
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes = []
        
        # Fix Windows-style paths to be cross-platform
        pattern = r'["\']([A-Za-z]:\\[^"\']+)["\']'
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                lambda m: f'os.path.join(os.path.dirname(__file__), "{os.path.basename(m.group(1))}")',
                content
            )
            fixes.append("Fixed hardcoded Windows paths")
        
        # Ensure model weight paths use os.path.join
        weight_files = ['tracknet_weights.pt', 'model_tennis_court_det.pt', 
                       'stroke_classifier_weights.pth', 'bounce_detection_weights.cbm']
        
        for weight_file in weight_files:
            pattern = f'["\'].*{weight_file}["\']'
            if re.search(pattern, content):
                # Check if it's not already using os.path
                if 'os.path' not in re.search(pattern, content).group(0):
                    content = re.sub(
                        pattern,
                        f'os.path.join(os.path.dirname(__file__), "{weight_file}")',
                        content
                    )
                    fixes.append(f"Fixed path for {weight_file}")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.fixes_applied.append(f"{filepath}: {', '.join(fixes)}")
            return True
        
        return False
    
    def create_required_directories(self):
        """Create required directories if they don't exist"""
        directories = ['uploads', 'outputs', 'temp', 'logs']
        created = []
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                created.append(directory)
        
        if created:
            self.fixes_applied.append(f"Created directories: {', '.join(created)}")
            return True
        
        return False
    
    def fix_cors_issues(self):
        """Add CORS middleware if missing"""
        filepath = 'local_backend.py'
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'CORSMiddleware' not in content:
            # Add CORS import
            if 'from fastapi import FastAPI' in content:
                content = content.replace(
                    'from fastapi import FastAPI',
                    'from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware'
                )
            
            # Add CORS middleware after app creation
            if 'app = FastAPI()' in content:
                cors_middleware = '''

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
                content = content.replace('app = FastAPI()', f'app = FastAPI(){cors_middleware}')
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append("Added CORS middleware to backend")
                return True
        
        return False
    
    def fix_opencv_codec(self):
        """Fix OpenCV video codec issues"""
        filepath = 'video_processor.py'
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for VideoWriter fourcc
        if "cv2.VideoWriter_fourcc" in content:
            # Try to use a more compatible codec
            content = re.sub(
                r"cv2\.VideoWriter_fourcc\([^)]+\)",
                "cv2.VideoWriter_fourcc(*'mp4v')",  # mp4v is widely supported
                content
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_applied.append("Updated video codec to mp4v for compatibility")
            return True
        
        return False
    
    def add_error_handling(self, filepath):
        """Add try-except blocks for model loading"""
        if not os.path.exists(filepath):
            return False
        
        self.backup_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Wrap torch.load calls in try-except
        pattern = r'(\s*)([\w.]+\.load_state_dict\(torch\.load\([^)]+\)\))'
        
        def add_try_except(match):
            indent = match.group(1)
            statement = match.group(2)
            return f'''{indent}try:
{indent}    {statement}
{indent}except RuntimeError as e:
{indent}    print(f"Warning: Loading model with strict=False: {{e}}")
{indent}    {statement.replace(')', ', strict=False)')}'''
        
        content = re.sub(pattern, add_try_except, content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.fixes_applied.append(f"{filepath}: Added error handling for model loading")
            return True
        
        return False
    
    def verify_model_weights(self):
        """Check if model weight files exist and have reasonable sizes"""
        weight_files = {
            'tracknet_weights.pt': (10, 100),  # Min 10MB, Max 100MB
            'model_tennis_court_det.pt': (50, 200),
            'stroke_classifier_weights.pth': (5, 50),
            'bounce_detection_weights.cbm': (0.1, 10),
        }
        
        missing = []
        invalid_size = []
        
        for weight_file, (min_mb, max_mb) in weight_files.items():
            if not os.path.exists(weight_file):
                missing.append(weight_file)
            else:
                size_mb = os.path.getsize(weight_file) / (1024 * 1024)
                if size_mb < min_mb or size_mb > max_mb:
                    invalid_size.append(f"{weight_file} ({size_mb:.1f}MB, expected {min_mb}-{max_mb}MB)")
        
        if missing:
            self.errors.append(f"Missing weight files: {', '.join(missing)}")
        if invalid_size:
            self.errors.append(f"Suspicious file sizes: {', '.join(invalid_size)}")
        
        return len(missing) == 0 and len(invalid_size) == 0
    
    def run_all_fixes(self):
        """Run all automated fixes"""
        print("\n" + "="*60)
        print("CourtCheck Automated Fix Script")
        print("="*60 + "\n")
        
        python_files = [
            'local_backend.py',
            'video_processor.py',
            'ball_detection.py',
            'court_detection_module.py',
            'stroke_classifier.py',
            'bounce_detection.py',
        ]
        
        print("🔧 Applying fixes...\n")
        
        # Fix 1: Device configuration
        print("1. Fixing device configuration (CUDA/CPU)...")
        for file in python_files:
            if self.fix_device_configuration(file):
                print(f"  ✓ Fixed {file}")
        
        # Fix 2: Import errors
        print("\n2. Fixing import statements...")
        for file in python_files:
            if self.fix_import_errors(file):
                print(f"  ✓ Fixed {file}")
        
        # Fix 3: Path issues
        print("\n3. Fixing file paths...")
        for file in python_files:
            if self.fix_path_issues(file):
                print(f"  ✓ Fixed {file}")
        
        # Fix 4: Create directories
        print("\n4. Creating required directories...")
        if self.create_required_directories():
            print("  ✓ Directories created")
        
        # Fix 5: CORS
        print("\n5. Fixing CORS issues...")
        if self.fix_cors_issues():
            print("  ✓ CORS configured")
        
        # Fix 6: Video codec
        print("\n6. Fixing video codec...")
        if self.fix_opencv_codec():
            print("  ✓ Codec updated")
        
        # Fix 7: Error handling
        print("\n7. Adding error handling...")
        for file in python_files:
            if self.add_error_handling(file):
                print(f"  ✓ Fixed {file}")
        
        # Verification
        print("\n8. Verifying model weights...")
        if self.verify_model_weights():
            print("  ✓ All weights present")
        
        # Report
        print("\n" + "="*60)
        print("Fix Report")
        print("="*60 + "\n")
        
        if self.fixes_applied:
            print(f"✅ Applied {len(self.fixes_applied)} fixes:\n")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        else:
            print("ℹ️  No fixes needed or no files found to fix")
        
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} warnings:\n")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n" + "="*60)
        print("\n✨ Done! You can now try running:")
        print("   python local_backend.py")
        print("\n💡 Backup files created with .backup extension")
        print("   To restore: mv file.py.backup file.py")
        print("\n")

def main():
    # Check if we're in the right directory
    if not os.path.exists('local_backend.py'):
        print("❌ Error: Not in CourtCheck project directory!")
        print("Please navigate to your CourtCheck directory and run this script again.")
        sys.exit(1)
    
    fixer = Fixer()
    fixer.run_all_fixes()

if __name__ == '__main__':
    main()
