import pandoc
from pathlib import Path
import tempfile


class TestEpubProcessing:
    def test_process_sample_epub(self):
        """Test processing the sample minimal.epub file"""
        test_dir = Path(__file__).parent
        epub_path = test_dir / "test_files" / "minimal.epub"

        # Verify file exists
        assert Path(epub_path).exists(), f"Sample EPUB file not found: {epub_path}"

        # Read EPUB
        doc = pandoc.read(file=str(epub_path), format="epub")
        assert doc is not None, "Failed to read EPUB file"

        # Convert to markdown
        markdown_text = pandoc.write(doc, format="markdown")
        assert markdown_text, "Markdown conversion produced empty result"
        assert len(markdown_text) > 100, "Markdown output seems too short"

        # Check for expected content
        assert "# Chapter 1" in markdown_text, "Chapter 1 header not found"
        assert "# Chapter 2" in markdown_text, "Chapter 2 header not found"
        assert "Lorem ipsum" in markdown_text, "Expected Lorem ipsum content not found"

    def test_roundtrip_conversion(self):
        """Test markdown -> epub -> markdown round-trip conversion"""
        # Create sample markdown content
        original_markdown = """# Test Document

## Introduction

This is a test document with some **bold** and *italic* text.

## Chapter 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

- Item 1
- Item 2
- Item 3

## Chapter 2

More content here with a [link](https://example.com).

> This is a blockquote.

```python
def hello():
    print("Hello, world!")
```
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write original markdown
            original_file = tmp_path / "original.md"
            original_file.write_text(original_markdown)

            # Convert markdown to epub
            epub_file = tmp_path / "test.epub"
            doc = pandoc.read(file=str(original_file), format="markdown")
            pandoc.write(doc, file=str(epub_file), format="epub")

            # Verify epub was created
            assert epub_file.exists(), "EPUB file was not created"
            assert epub_file.stat().st_size > 0, "EPUB file is empty"

            # Convert epub back to markdown
            doc_from_epub = pandoc.read(file=str(epub_file), format="epub")
            recovered_markdown = pandoc.write(doc_from_epub, format="markdown")

            # Basic checks on recovered content
            assert recovered_markdown, "Recovery produced empty markdown"
            assert "# Test Document" in recovered_markdown, "Main title not recovered"
            assert "## Introduction" in recovered_markdown, (
                "Section header not recovered"
            )
            assert "Lorem ipsum" in recovered_markdown, "Body text not recovered"
            assert "**bold**" in recovered_markdown or "bold" in recovered_markdown, (
                "Bold formatting lost"
            )
