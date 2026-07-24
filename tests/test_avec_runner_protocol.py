from __future__ import annotations

import contextlib
import io
import json
import unittest

from team1_asr import avec_manifest_runner


class AvecRunnerProtocolTest(unittest.TestCase):
    def test_protocol_output_survives_model_stdout_redirect(self) -> None:
        protocol = io.StringIO()
        redirected = io.StringIO()
        previous = avec_manifest_runner._PROTOCOL_OUTPUT
        avec_manifest_runner._PROTOCOL_OUTPUT = protocol
        try:
            with contextlib.redirect_stdout(redirected):
                avec_manifest_runner._write(
                    {"type": "progress", "event": {"stage": "beam"}}
                )
        finally:
            avec_manifest_runner._PROTOCOL_OUTPUT = previous

        self.assertEqual(redirected.getvalue(), "")
        message = json.loads(protocol.getvalue())
        self.assertEqual(message["type"], "progress")
        self.assertEqual(message["event"]["stage"], "beam")


if __name__ == "__main__":
    unittest.main()
