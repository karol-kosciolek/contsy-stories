# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Restart from Settings: only the typed word RESTART restarts, never while a job generates;
the detached helper waits for the old process and the port, then runs start.bat."""
import socket
import sys
import unittest
from unittest.mock import patch

from app import llm, queue_manager, restart


class RestartTests(unittest.TestCase):
    def test_wrong_or_cancelled_confirmation_restarts_nothing(self):
        for typed in ("", "restar", "yes", None):
            with self.subTest(typed=typed), patch.object(restart.threading, "Thread") as thread:
                ok, msg = restart.request(typed)
            self.assertFalse(ok)
            self.assertTrue(msg.startswith("Nothing restarted"), msg)
            thread.assert_not_called()

    def test_generating_job_blocks_the_restart(self):
        job = queue_manager.Job(7, "wideo", "Video 5 s: test", {}, status=queue_manager.RUNNING)
        with patch.object(queue_manager, "snapshot", return_value=[job]), \
                patch.object(restart.threading, "Thread") as thread:
            ok, msg = restart.request("RESTART")
        self.assertFalse(ok)
        self.assertIn("#7", msg)
        thread.assert_not_called()

    def test_a_working_agent_blocks_the_restart_and_is_paused_if_it_began_meanwhile(self):
        with patch.object(queue_manager, "snapshot", return_value=[]),                 patch.object(restart.threading, "Thread") as thread,                 patch("app.agent_runner.busy", return_value=True):
            ok, msg = restart.request("RESTART")
        self.assertFalse(ok)
        self.assertIn("Agent", msg)
        thread.assert_not_called()
        with patch.object(restart.subprocess, "Popen"), patch.object(restart.os, "_exit"),                 patch.object(restart, "port", return_value=7860), patch.object(llm, "shutdown"),                 patch("app.agent_runner.shutdown") as shutdown:
            restart._restart(delay=0)
        shutdown.assert_called_once_with()

    def test_confirmed_restart_starts_the_helper_thread(self):
        with patch.object(queue_manager, "snapshot", return_value=[]), \
                patch.object(restart.threading, "Thread") as thread:
            ok, msg = restart.request(" restart ")
        self.assertTrue(ok, msg)
        self.assertIn("20 seconds", msg)
        self.assertIs(thread.call_args.kwargs["target"], restart._restart)
        thread.return_value.start.assert_called_once()

    def test_restart_thread_spawns_the_helper_then_exits(self):
        with patch.object(restart.subprocess, "Popen") as popen, \
                patch.object(restart.os, "_exit") as exit_, \
                patch.object(restart, "port", return_value=7860), \
                patch.object(llm, "shutdown") as shutdown:
            restart._restart(delay=0)
        args = popen.call_args.args[0]
        self.assertEqual(args[0], sys.executable)
        self.assertTrue(args[1].endswith("restart.py"))
        self.assertEqual(args[2:], [str(restart.os.getpid()), "7860"])
        shutdown.assert_called_once_with()
        exit_.assert_called_once_with(0)

    def test_helper_waits_for_process_and_port_then_runs_start_bat(self):
        with patch.object(restart, "wait_for_exit", return_value=True) as waited, \
                patch.object(restart, "wait_for_port", return_value=True) as freed, \
                patch.object(restart.subprocess, "Popen") as popen:
            self.assertEqual(restart.launch_after(1234, 7860), 0)
        waited.assert_called_once_with(1234)
        freed.assert_called_once_with(7860)
        command = popen.call_args.args[0]
        self.assertEqual(command, restart.start_command())
        self.assertTrue(command.startswith('cmd.exe /c start "'))
        self.assertTrue(command.endswith('start.bat"'))
        self.assertEqual(popen.call_args.kwargs["cwd"], str(restart.ROOT))

    def test_the_window_title_is_quoted_so_start_does_not_run_it(self):
        """`start Contsy x.bat` looks for a program named Contsy (the restart then never came back): the
        title must be the first quoted argument, whatever the name."""
        command = restart.start_command()
        self.assertIn(f'start "{restart.TITLE}" "{restart.START}"', command)
        if restart.os.name == "nt":  # cmd receives the line as written (a string, not a list subprocess re-quotes)
            probe = restart.subprocess.run("cmd.exe /d /c echo " + command.split(" /c ", 1)[1],
                                           capture_output=True, text=True, timeout=20)
            self.assertEqual(probe.stdout.strip(), f'start "{restart.TITLE}" "{restart.START}"')

    def test_gone_process_and_free_port_do_not_wait(self):
        self.assertTrue(restart.wait_for_exit(2 ** 30 + 1, timeout=1.0))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        self.assertTrue(restart.wait_for_port(free_port, timeout=1.0))


if __name__ == "__main__":
    unittest.main()
