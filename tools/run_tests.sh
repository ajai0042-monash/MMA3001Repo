#!/bin/bash
set -e
pytest tests/ -v --tb=short > tests/pytest_report.txt 2>&1 || true
cat tests/pytest_report.txt
