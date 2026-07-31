"""Tests for the logging module."""

import logging
import unittest

from tm_ecg.log import setup_logging


class LoggingTests(unittest.TestCase):
    def test_setup_logging_info(self) -> None:
        setup_logging(1)
        logger = logging.getLogger("tm_ecg")
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logging_debug(self) -> None:
        setup_logging(2)
        logger = logging.getLogger("tm_ecg")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logging_quiet(self) -> None:
        setup_logging(0)
        logger = logging.getLogger("tm_ecg")
        self.assertEqual(logger.level, logging.WARNING)

    def test_setup_logging_is_idempotent(self) -> None:
        setup_logging(1)
        setup_logging(1)
        logger = logging.getLogger("tm_ecg")
        # Should not add duplicate handlers
        self.assertLessEqual(len(logger.handlers), 1)


if __name__ == "__main__":
    unittest.main()
