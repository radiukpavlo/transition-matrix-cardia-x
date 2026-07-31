"""Tests for the consolidated io.readers module."""

import unittest
from pathlib import Path
import tempfile
import shutil

from tm_ecg.io.readers import find_table, read_table_rows


class ReadTableRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_read_csv_returns_string_values(self) -> None:
        csv_path = self.tmpdir / "data.csv"
        csv_path.write_text("record_id,value\nr1,42.5\nr2,10.0\n", encoding="utf-8")
        rows = read_table_rows(csv_path)
        self.assertEqual(len(rows), 2)
        # CSV DictReader always returns strings
        self.assertEqual(rows[0]["value"], "42.5")
        self.assertEqual(rows[1]["record_id"], "r2")

    def test_read_csv_empty_file_returns_empty(self) -> None:
        csv_path = self.tmpdir / "empty.csv"
        csv_path.write_text("col_a,col_b\n", encoding="utf-8")
        rows = read_table_rows(csv_path)
        self.assertEqual(rows, [])

    def test_unsupported_extension_raises(self) -> None:
        bad_path = self.tmpdir / "data.xyz"
        bad_path.write_text("hello", encoding="utf-8")
        with self.assertRaises((ValueError, Exception)):
            read_table_rows(bad_path)


class FindTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_find_table_prefers_parquet(self) -> None:
        (self.tmpdir / "data.csv").write_text("a\n1\n", encoding="utf-8")
        (self.tmpdir / "data.parquet").write_text("fake", encoding="utf-8")
        result = find_table(self.tmpdir, "data")
        self.assertIsNotNone(result)
        self.assertEqual(result.suffix, ".parquet")

    def test_find_table_falls_back_to_csv(self) -> None:
        (self.tmpdir / "data.csv").write_text("a\n1\n", encoding="utf-8")
        result = find_table(self.tmpdir, "data")
        self.assertIsNotNone(result)
        self.assertEqual(result.suffix, ".csv")

    def test_find_table_missing_returns_none(self) -> None:
        result = find_table(self.tmpdir, "nonexistent")
        self.assertIsNone(result)

    def test_find_table_required_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            find_table(self.tmpdir, "nonexistent", required=True)


if __name__ == "__main__":
    unittest.main()
