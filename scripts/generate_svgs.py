#!/usr/bin/env python3
"""
Generate SVG versions of Mermaid diagrams for documentation
"""

import os
import re
import requests
import json
from pathlib import Path
from typing import Optional

class MermaidToSVGConverter:
    """Convert Mermaid diagrams to SVG format"""

    def __init__(self):
        self.mermaid_api_url = "https://mermaid.ink/img/"

    def extract_mermaid_code(self, markdown_file: Path) -> Optional[str]:
        """Extract Mermaid code from a Markdown file"""
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find Mermaid code blocks
            mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
            match = re.search(mermaid_pattern, content, re.DOTALL)

            if match:
                return match.group(1).strip()
            else:
                print(f"No Mermaid diagram found in {markdown_file}")
                return None

        except Exception as e:
            print(f"Error reading {markdown_file}: {e}")
            return None

    def convert_to_svg(self, mermaid_code: str, output_path: Path) -> bool:
        """Convert Mermaid code to SVG using mermaid.ink service"""
        try:
            # Encode the Mermaid code for the URL
            import base64
            import urllib.parse

            # Create the data URL format expected by mermaid.ink
            mermaid_data = f"%%{{init: {{'theme': 'base', 'themeVariables': {{'primaryColor': '#e3f2fd', 'primaryTextColor': '#1976d2', 'primaryBorderColor': '#01579b', 'lineColor': '#424242', 'secondaryColor': '#fff3e0', 'tertiaryColor': '#fce4ec'}}}}}}%%\n{mermaid_code}"

            # Base64 encode the mermaid code
            encoded = base64.b64encode(mermaid_data.encode('utf-8')).decode('utf-8')
            url_encoded = urllib.parse.quote(encoded)

            # Create the full URL
            full_url = f"{self.mermaid_api_url}{url_encoded}"

            # Make the request
            response = requests.get(full_url, timeout=30)

            if response.status_code == 200:
                # Save the SVG
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Generated SVG: {output_path}")
                return True
            else:
                print(f"❌ Failed to generate SVG for {output_path}: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error generating SVG for {output_path}: {e}")
            return False

    def process_markdown_file(self, markdown_file: Path, output_dir: Path) -> bool:
        """Process a single Markdown file and generate its SVG"""
        # Extract Mermaid code
        mermaid_code = self.extract_mermaid_code(markdown_file)
        if not mermaid_code:
            return False

        # Create output filename
        svg_filename = markdown_file.stem + ".svg"
        output_path = output_dir / svg_filename

        # Convert to SVG
        return self.convert_to_svg(mermaid_code, output_path)

def main():
    """Main function to generate SVGs for all documentation files"""
    print("🔄 Generating SVG versions of documentation diagrams...")

    # Setup paths
    docs_dir = Path(__file__).parent.parent / "docs"
    svgs_dir = docs_dir / "svgs"

    # Create SVGs directory
    svgs_dir.mkdir(exist_ok=True)

    # Initialize converter
    converter = MermaidToSVGConverter()

    # Process all Markdown files in docs directory
    markdown_files = [
        "README.md",
        "architecture.md",
        "interface-workflow.md",
        "training-pipeline.md",
        "deployment-pipeline.md",
        "data-flow.md"
    ]

    success_count = 0
    total_count = len(markdown_files)

    for filename in markdown_files:
        markdown_path = docs_dir / filename
        if markdown_path.exists():
            print(f"\n📄 Processing {filename}...")
            if converter.process_markdown_file(markdown_path, svgs_dir):
                success_count += 1
        else:
            print(f"⚠️  File not found: {markdown_path}")

    print(f"\n🎉 SVG generation complete!")
    print(f"✅ Successfully generated: {success_count}/{total_count} SVGs")
    print(f"📁 SVGs saved to: {svgs_dir}")

    if success_count < total_count:
        print(f"❌ Failed to generate: {total_count - success_count} SVGs")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())

