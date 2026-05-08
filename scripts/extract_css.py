"""Extract inline CSS from nbconvert HTML into a separate .css file."""
import re
import os

def extract_css(input_file, output_html, output_css):
    with open(input_file, 'r') as f:
        content = f.read()

    # Extract all <style>...</style> blocks
    style_pattern = re.compile(r'<style[^>]*>(.*?)</style>', re.DOTALL)
    styles = style_pattern.findall(content)

    # Combine all CSS
    css_content = '\n\n'.join(styles)

    # Write CSS file
    with open(output_css, 'w') as f:
        f.write(css_content)

    # Remove inline styles and replace with link to external CSS
    # Keep the first link to reveal.css if it's a slides file
    content_clean = style_pattern.sub('', content)

    # Insert CSS link after <head> or after existing links
    head_close = content_clean.find('</head>')
    if head_close != -1:
        # Use just the filename since CSS lives alongside HTML
        css_filename = os.path.basename(output_css)
        css_link = f'<link rel="stylesheet" href="{css_filename}">\n'
        content_clean = content_clean[:head_close] + css_link + content_clean[head_close:]

    # Clean up empty lines left behind
    content_clean = re.sub(r'\n{4,}', '\n\n\n', content_clean)

    with open(output_html, 'w') as f:
        f.write(content_clean)

    print(f"Extracted {len(styles)} style blocks ({len(css_content)} chars)")
    print(f"CSS:  {output_css}")
    print(f"HTML: {output_html}")

# Process slideshow
extract_css(
    'presentation/presentation_slides.html',
    'presentation/presentation_slides.html',
    'presentation/slides.css'
)

# Process scrollable
extract_css(
    'presentation/presentation_scroll.html',
    'presentation/presentation_scroll.html',
    'presentation/scroll.css'
)
