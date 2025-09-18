#!/bin/bash

# Quick AdamW Fix for ColBERT
# This script fixes the "cannot import name 'AdamW' from 'transformers'" error

set -e

echo "🔧 Quick AdamW Fix for ColBERT"
echo "================================"

# Detect OS for sed compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SED_CMD="sed -i ''"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    SED_CMD="sed -i"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash/WSL)
    SED_CMD="sed -i"
else
    # Default to Linux style
    SED_CMD="sed -i"
fi

# Find ColBERT installation (don't try to import it first)
COLBERT_DIR=$(python3 -c "
import sys
import os
import site

# Just search for colbert directory without importing
for path in site.getsitepackages() + [os.path.join(sys.prefix, 'lib', 'python' + '.'.join(map(str, sys.version_info[:2])), 'site-packages')]:
    colbert_path = os.path.join(path, 'colbert')
    if os.path.exists(colbert_path):
        print(colbert_path)
        break
" 2>/dev/null)

if [[ -z "${COLBERT_DIR}" ]]; then
    echo "Error: ColBERT package not found. Please install it first:"
    echo "   pip install ragatouille"
    exit 1
fi

echo "Found ColBERT at: ${COLBERT_DIR}"

# Backup original file
BACKUP_FILE="${COLBERT_DIR}/training/training.py.backup.$(date +%Y%m%d_%H%M%S)"
cp "${COLBERT_DIR}/training/training.py" "${BACKUP_FILE}"
echo "Backed up original to: ${BACKUP_FILE}"

# Apply the fix: replace transformers.AdamW with torch.optim.AdamW
$SED_CMD 's/from transformers import AdamW, get_linear_schedule_with_warmup/from transformers import get_linear_schedule_with_warmup\nfrom torch.optim import AdamW/' "${COLBERT_DIR}/training/training.py"

echo "Applied AdamW fix"

# Test the fix
echo "Testing the fix..."
export KMP_DUPLICATE_LIB_OK=TRUE
if python3 -c "import colbert; print('SUCCESS: ColBERT imports correctly!')" 2>/dev/null; then
    echo "Fix applied successfully!"
else
    echo "Fix failed. Restoring backup..."
    cp "${BACKUP_FILE}" "${COLBERT_DIR}/training/training.py"
    exit 1
fi 