#!/bin/bash

# Script to find all console.log statements in the codebase
# Usage: ./scripts/find-console-logs.sh

echo "🔍 Finding all console statements in the codebase..."
echo ""

# Count total console statements
TOTAL=$(grep -r "console\." --include="*.ts" --include="*.tsx" --exclude-dir=node_modules --exclude-dir=.next . | wc -l | tr -d ' ')
echo "📊 Total console statements found: $TOTAL"
echo ""

# Group by file
echo "📁 Files with console statements:"
echo "=================================="
grep -r "console\." --include="*.ts" --include="*.tsx" --exclude-dir=node_modules --exclude-dir=.next . | cut -d':' -f1 | sort | uniq -c | sort -rn

echo ""
echo "🔍 Top 10 files with most console statements:"
echo "=============================================="
grep -r "console\." --include="*.ts" --include="*.tsx" --exclude-dir=node_modules --exclude-dir=.next . | cut -d':' -f1 | sort | uniq -c | sort -rn | head -10

echo ""
echo "⚠️  Critical files (API & hooks) with console statements:"
echo "=========================================================="
grep -rn "console\." --include="*.ts" --include="*.tsx" --exclude-dir=node_modules --exclude-dir=.next lib/api lib/hooks | head -20

echo ""
echo "💡 To replace console.log with logger, run:"
echo "   1. Import: import { logger } from '@/lib/logger'"
echo "   2. Replace:"
echo "      - console.log()    → logger.info()"
echo "      - console.debug()  → logger.debug()"
echo "      - console.warn()   → logger.warn()"
echo "      - console.error()  → logger.error()"
echo ""
echo "🎯 Priority order:"
echo "   1. lib/api/* (API clients)"
echo "   2. lib/hooks/* (React hooks)"
echo "   3. components/* (UI components)"
echo "   4. app/* (Pages and routes)"
