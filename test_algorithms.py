"""
Quick test runner for all algorithms on n-to-n dataset.
"""
import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "experiments"))

ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "utils"))

# Test SVM
print("\n" + "="*70)
print("Testing SVM on n-to-n dataset...")
print("="*70)
try:
    from experiments_02_svm_import import train_svm, evaluate_svm
    from logging_utils import ExperimentLogger

    logger = ExperimentLogger("test_svm_nton")
    result = train_svm(ROOT, "n-to-n", logger=logger)
    if result:
        clf, scaler, le, logger = result
        metrics = evaluate_svm(clf, scaler, le, ROOT, "n-to-n", logger)
        print("[OK] SVM test completed")
        if metrics:
            print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
except Exception as e:
    print(f"[ERROR] SVM test failed: {e}")
    import traceback
    traceback.print_exc()
