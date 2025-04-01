import re


class MarkdownCleaner:
    def __init__(self):
        pass

    def _remove_links_from_markdown(markdown: str) -> str:
        """Removes all markdown links from the text."""
        markdown = re.sub(r'(\[[^\]]+\])\((https?:\/\/[^\)]+)\)', ' ', markdown)  # Remove markdown-style links
        markdown = re.sub(r'\((https?:\/\/[^\)]+)\)', '', markdown)  # Remove raw links in parentheses
        return markdown

    def clean(self, markdown: str) -> str:
        markdown = self._remove_links_from_markdown(markdown)
