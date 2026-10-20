"""Idle-session expiry (abandoned sessions after a segd crash/restart must not pin
capacity or memory forever), in-flight protection, and /healthz session accounting."""
from __future__ import annotations

import threading

import httpx
import numpy as np
import pytest

from sammed3d import client, service
from conftest import _FakeService, lesion_centre, make_phantom


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _open(url, img=None):
    sess = client.SamMed3DSession(url)
    if img is not None:
        sess.set_target_buffer(np.zeros(img.shape, dtype=np.uint8))
        sess.set_image(img.astype(np.float16)[None], image_properties={"spacing": [1.0, 1.0, 1.0]})
    return sess


def _health(url) -> dict:
    return httpx.get(url + "/healthz").json()


def test_abandoned_sessions_expire_and_capacity_recovers():
    clock = _Clock()
    with _FakeService(max_sessions=2, session_ttl_s=100.0, clock=clock) as svc:
        a, b = _open(svc.url), _open(svc.url)             # "abandoned": never closed
        with pytest.raises(client.ServerAtCapacityError):  # full: the old behaviour, forever
            _open(svc.url)
        h = _health(svc.url)["sessions"]
        assert h["count"] == 2 and h["max"] == 2 and h["ttl_s"] == 100.0
        clock.t += 60.0
        assert _health(svc.url)["sessions"]["count"] == 2   # not yet idle long enough
        assert _health(svc.url)["sessions"]["oldest_age_s"] == pytest.approx(60.0)
        with pytest.raises(client.ServerAtCapacityError):
            _open(svc.url)
        clock.t += 41.0                                    # idle 101 s > ttl
        c = _open(svc.url)                                 # capacity recovered without restart
        assert c.ping()
        h = _health(svc.url)["sessions"]
        assert h["count"] == 1 and h["reaped_total"] == 2
        with pytest.raises(client.SessionExpiredError):    # segd would reconnect on this
            a.add_point_interaction((1, 1, 1), True)
        c.close()


def test_activity_refreshes_the_idle_clock():
    clock = _Clock()
    img, lesion, _b = make_phantom()
    with _FakeService(max_sessions=2, session_ttl_s=100.0, clock=clock) as svc:
        a = _open(svc.url, img)
        clock.t += 80.0
        a.add_point_interaction(lesion_centre(lesion), True)   # activity at t+80
        clock.t += 80.0                                        # idle 80 s < ttl
        assert _health(svc.url)["sessions"]["count"] == 1
        assert a.ping()
        a.add_point_interaction(lesion_centre(lesion), True)   # still alive
        clock.t += 101.0
        assert _health(svc.url)["sessions"]["count"] == 0
        a.close()


class _BlockingPredictor(service.FakePatchPredictor):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_patch(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(30), "test never released the in-flight prediction"
        return super().predict_patch(*args, **kwargs)


def test_in_flight_session_is_never_reaped():
    clock = _Clock()
    img, lesion, _b = make_phantom()
    pred = _BlockingPredictor()
    with _FakeService(predictor=pred, max_sessions=2, session_ttl_s=100.0, clock=clock) as svc:
        busy = _open(svc.url, img)
        idle = _open(svc.url)
        result = {}

        def run():
            busy.add_point_interaction(lesion_centre(lesion), True)
            result["mask"] = int((busy.target_buffer > 0).sum())

        t = threading.Thread(target=run)
        t.start()
        assert pred.entered.wait(30)                       # the prediction is in flight
        clock.t += 1000.0                                  # far beyond the ttl
        h = _health(svc.url)["sessions"]
        assert h["busy"] == 1 and h["count"] == 1 and h["reaped_total"] == 1   # idle one reaped
        third = _open(svc.url)                             # capacity used by the reaper
        pred.release.set()
        t.join(30)
        assert result["mask"] > 0                          # the in-flight request completed
        assert _health(svc.url)["sessions"]["count"] == 2
        with pytest.raises(client.SessionExpiredError):
            idle.add_point_interaction((1, 1, 1), True)
        # finishing refreshed the idle clock: the formerly busy session survives ttl-1
        clock.t += 99.0
        assert busy.ping() and _health(svc.url)["sessions"]["count"] == 2
        third.close()
        busy.close()


def test_healthz_reports_session_count_and_age():
    clock = _Clock()
    with _FakeService(max_sessions=4, session_ttl_s=500.0, clock=clock) as svc:
        s1 = _open(svc.url)
        clock.t += 30.0
        s2 = _open(svc.url)
        clock.t += 10.0
        h = _health(svc.url)["sessions"]
        assert h == {"count": 2, "max": 4, "busy": 0, "oldest_age_s": 40.0, "max_idle_s": 40.0,
                     "ttl_s": 500.0, "reaped_total": 0}
        s1.close(); s2.close()
        assert _health(svc.url)["sessions"]["count"] == 0
