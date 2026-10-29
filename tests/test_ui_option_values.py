from html.parser import HTMLParser
from pathlib import Path
import unittest


class SelectOptions(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.options = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.current = attrs.get("id")
            self.options[self.current] = []
        elif tag == "option" and self.current:
            self.options[self.current].append(attrs.get("value"))

    def handle_endtag(self, tag):
        if tag == "select":
            self.current = None


class CoinOptionValueTests(unittest.TestCase):
    def test_existing_coin_values_are_independent_of_display_labels(self):
        # 표시명을 한글로 바꿔도 내부 영문 심벌이 유지돼야 합니다.
        root = Path(__file__).resolve().parents[1]
        parser = SelectOptions()
        parser.feed((root / "web" / "index.html").read_text(encoding="utf-8"))
        for select_id in ("data-filter", "trade-filter"):
            with self.subTest(select_id=select_id):
                values = parser.options[select_id]
                self.assertNotIn(None, values)
                self.assertTrue({"all", "BTC", "SOL", "ETH", "XRP"}.issubset(values))


if __name__ == "__main__":
    unittest.main()
