import re
from ftfy import fix_text


class MarkdownCleaner:
    """A utility class for cleaning and transforming markdown content.

    This class provides functionality to process markdown text by handling
    special elements like links and images, replacing them with standardized
    tokens for easier parsing or display.
    """

    _IMG_PATTERN = re.compile(r"!\[([^\]]*)\]\([^\)]*\)")
    _INTERNAL_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(<INTERNAL_PAGE>\)')
    _EXTERNAL_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(<EXTERNAL_SITE>\)')
    _EMPTY_LINK_PATTERN = re.compile(r'\[\]\((<INTERNAL_PAGE>|<EXTERNAL_SITE>)\)')
    _EMAIL_PATTERN = re.compile(r' <EXTERNAL_SITE>([^<]*@[^<]*)<EXTERNAL_SITE> ')
    _WHITESPACE_PATTERN = re.compile(r' +')

    def _remove_links_from_markdown(
        self, markdown: str,
        internal_links: list[str],
        external_links: list[str]
    ) -> str:
        """Transforms markdown links into webpage-specific tokens for LLM processing.

        This method processes markdown text by:
        1. Converting image elements to image caption tokens
        2. Replacing internal and external webpage links with appropriate tokens
        3. Identifying and marking email addresses
        4. Normalizing whitespace for consistent processing

        Args:
            markdown: The markdown text to process.
            internal_links: A list of internal webpage URLs to replace.
            external_links: A list of external website URLs to replace.

        Returns:
            The processed markdown with webpage-specific tokens for LLM comprehension.

        Raises:
            TypeError: If markdown is not a string.
        """
        if not isinstance(markdown, str):
            raise TypeError("Markdown input must be a string")

        if markdown.strip() == "":
            return ""

        # Transform image markdown to image caption tokens
        # ![Product Photo](photo.jpg) -> [IMAGE:Product Photo]
        markdown = MarkdownCleaner._IMG_PATTERN.sub(r'[IMAGE:\1]', markdown)

        # Replace links with webpage-specific placeholders
        # Process longer links first to avoid partial replacements
        for link in sorted(internal_links, key=len, reverse=True):
            markdown = markdown.replace(f'({link})', '(<INTERNAL_PAGE>)')

        for link in sorted(external_links, key=len, reverse=True):
            markdown = markdown.replace(f'({link})', '(<EXTERNAL_SITE>)')

        # Transform link syntax to webpage-specific tokens
        # [About Us](<INTERNAL_PAGE>) -> [SITE_PAGE:About Us]
        markdown = MarkdownCleaner._INTERNAL_LINK_PATTERN.sub(r' [SITE_PAGE:\1] ', markdown)

        # [GitHub](<EXTERNAL_SITE>) -> [EXTERNAL_SITE:GitHub]
        markdown = MarkdownCleaner._EXTERNAL_LINK_PATTERN.sub(r' [EXTERNAL_SITE:\1] ', markdown)

        # Remove empty links
        markdown = MarkdownCleaner._EMPTY_LINK_PATTERN.sub(r' ', markdown)

        # Transform email addresses to contact tokens
        # [EXTERNAL_SITE:contact@example.com] -> [EMAIL:contact@example.com]
        markdown = MarkdownCleaner._EMAIL_PATTERN.sub(r' [EMAIL:\1] ', markdown)

        # Normalize white spaces
        return MarkdownCleaner._WHITESPACE_PATTERN.sub(' ', markdown.strip())

    @staticmethod
    def normalize_whitespace(markdown: str) -> str:
        """Normalizes the whitespaces in a document by applying the following steps:

        1. Replace multiple \n or mixed whitespace+newlines with a single \n
        2. Replace multiple spaces/tabs with a single space (within lines)
        3. Strip leading/trailing whitespace on each line

        Args:
            markdown: The markdown text to clean.

        Returns:
            The normalized markdown text.
        """
        markdown = re.sub(r'[\s]*\n[\s\n]*', '\n', markdown)
        markdown = re.sub(r'[ \t]+', ' ', markdown)
        lines = [line.strip() for line in markdown.split('\n')]
        return '\n'.join(line for line in lines if line)

    @staticmethod
    def _remove_bracket_type(s: str, open_bracket: str, close_bracket: str):
        """Removes content from a given type of bracket including the brackets"""
        stack = []
        result = []
        for char in s:
            if char == open_bracket:
                stack.append(char)
            elif char == close_bracket and stack:
                stack.pop()
            elif not stack:
                result.append(char)
        return ''.join(result)

    def remove_nested_brackets(self, markdown: str):
        """Removes the content in brackets from
        the document, which is most often links, images
        or other potentially missleading text.

        Args:
            markdown: The markdown text to clean.

        Returns:
            The markdown text without brackets.
        """
        markdown = self._remove_bracket_type(markdown, '[', ']')
        markdown = self._remove_bracket_type(markdown, '(', ')')
        return self.normalize_whitespace(markdown)

    def clean(
        self, markdown: str,
        internal_links: list[str],
        external_links: list[str]
    ) -> str:
        """Clean markdown by processing links and normalizing content.

        This is the main entry point for markdown cleaning operations.

        Args:
            markdown: The markdown text to clean.
            internal_links: A list of internal links to process.
            external_links: A list of external links to process.

        Returns:
            The cleaned markdown text.
        """
        # Remove all links from the markdown text to remove noise
        no_link_md = self._remove_links_from_markdown(
            markdown,
            internal_links=internal_links,
            external_links=external_links
        )

        # Fix encoding problems in markdown
        fixed_text = fix_text(no_link_md)

        return fixed_text
