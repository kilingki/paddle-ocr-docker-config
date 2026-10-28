from app.prepare import status_valid


def test_prepare_accepts_existing_lifecycle_but_checks_invariants():
    for state in ("ready", "loading", "unloading", "failed"):
        assert status_valid({"state":state, "residency":"resident", "active_requests":1, "last_error":None})
    assert not status_valid({"state":"unloaded", "residency":"resident", "active_requests":0,"last_error":None})
    assert not status_valid({"state":"ready", "residency":"resident", "active_requests":False,"last_error":None})
    assert not status_valid({"state":"ready", "residency":"unknown", "active_requests":0,"last_error":None})
    assert not status_valid({"state":"ready", "residency":"resident", "active_requests":0,"last_error":None,"pid":123})
