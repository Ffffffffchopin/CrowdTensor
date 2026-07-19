from scripts import lightning_api_gpu_provider_check as check


def valid_report() -> dict:
    return {
        "schema": "lightning_api_gpu_provider_probe_v1",
        "api_auth_verified": True,
        "public_artifact_safe": True,
        "credentials_public": False,
        "create_or_start_attempted": False,
        "token_file": {
            "api_key_present": True,
            "user_id_present": True,
            "secret_values_public": False,
        },
        "default_accelerators": {
            "gpu_accelerator_count": 1,
        },
        "blockers": [
            "lightning_free_cloud_space_start_not_allowed",
        ],
    }


def test_accepts_public_safe_readonly_report() -> None:
    assert check.check_report(valid_report()) == []


def test_rejects_start_attempt() -> None:
    report = valid_report()
    report["create_or_start_attempted"] = True
    assert "create_or_start_was_attempted" in check.check_report(report)


def test_rejects_secret_public_flag() -> None:
    report = valid_report()
    report["token_file"]["secret_values_public"] = True
    assert "token_secret_values_public_not_false" in check.check_report(report)


def test_rejects_missing_gpu_evidence() -> None:
    report = valid_report()
    report["default_accelerators"]["gpu_accelerator_count"] = 0
    assert "no_gpu_accelerators_reported" in check.check_report(report)
