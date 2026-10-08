import sys
import os
import traceback

print(f"Python version: {sys.version}")

try:
    import tomllib

    print("tomllib is available (standard library)")
except ImportError:
    try:
        import tomli as tomllib

        print("tomli is available (package)")
    except ImportError:
        print("Neither tomllib nor tomli is available! Migration will fail.")


try:
    print("-" * 50)
    print("Checking MetaModelConfig...")
    from smartstablemodel.config.metamodel_config import MetaModelConfig

    # Try loading from default (should be .toml now)
    try:
        cfg = MetaModelConfig.from_toml()
        print(
            f"SUCCESS: MetaModelConfig loaded via from_toml. Found {len(cfg.cluster_seconds_per_severity_level)} cluster labels."
        )
        # print first label to verify content
        if cfg.cluster_seconds_per_severity_level:
            print(
                f"Sample cluster label: {list(cfg.cluster_seconds_per_severity_level.keys())[0]} = {list(cfg.cluster_seconds_per_severity_level.values())[0]}"
            )
    except Exception as e:
        print(f"FAILURE: MetaModelConfig.from_toml() failed: {e}")
        traceback.print_exc()

    # Try backward compatibility
    try:
        cfg2 = MetaModelConfig.from_yaml()
        print(
            f"SUCCESS: MetaModelConfig loaded via from_yaml (compat). Found {len(cfg2.cluster_seconds_per_severity_level)} cluster labels."
        )
    except Exception as e:
        print(f"FAILURE: MetaModelConfig.from_yaml() failed: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"Critical error importing MetaModelConfig: {e}")
    traceback.print_exc()


try:
    print("-" * 50)
    print("Checking AnomalyConfig...")
    from smartstablemodel.config.anomaly_config import (
        load_anomaly_config,
        AnomalyConfig,
    )

    ac = load_anomaly_config()
    print(
        f"SUCCESS: AnomalyConfig loaded. Enabled={ac.enabled}, Threshold={ac.threshold}"
    )

except Exception as e:
    print(f"Critical error importing AnomalyConfig: {e}")
    traceback.print_exc()

# Check manager loader if possible
try:
    print("-" * 50)
    print("Checking Manager Loader...")
    try:
        # We need to make sure we can import this.
        # ml_backend should be in sys.path if running from /app/src
        from ml_backend.smartstable_core.manager_loader import get_multistall_manager

        print("SUCCESS: Imported get_multistall_manager")
    except ImportError as e:
        print(f"Skipping Manager Loader check: {e}")
except Exception as e:
    print(f"Error checking Manager Loader: {e}")
    traceback.print_exc()
