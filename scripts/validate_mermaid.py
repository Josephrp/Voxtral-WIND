#!/usr/bin/env python3
"""
Validate Mermaid syntax in HTML documentation
"""

import re

def validate_mermaid_html(html_file):
    """Validate Mermaid diagrams in HTML file"""
    print(f"🔍 Validating Mermaid syntax in {html_file}")

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all Mermaid blocks
    mermaid_pattern = r'<div class="mermaid">(.*?)</div>'
    mermaid_blocks = re.findall(mermaid_pattern, content, re.DOTALL)

    print(f"📊 Found {len(mermaid_blocks)} Mermaid blocks")

    issues = []

    # Check each Mermaid block
    for i, block in enumerate(mermaid_blocks):
        lines = block.strip().split('\n')
        if not lines or not lines[0].strip():
            issues.append(f"Block {i+1}: Empty Mermaid block")
            continue

        first_line = lines[0].strip()

        # Check if it starts with a valid diagram type
        valid_starts = [
            'graph', 'flowchart', 'stateDiagram', 'sequenceDiagram',
            'classDiagram', 'erDiagram', 'journey', 'gantt', 'pie',
            'gitgraph', 'mindmap', 'timeline', 'sankey'
        ]

        if not any(first_line.startswith(start) for start in valid_starts):
            issues.append(f"Block {i+1}: Invalid diagram type start - '{first_line}'")

        # Check for classDef/class consistency
        if 'classDef' in block:
            class_statements = len(re.findall(r'^\s*class\s+', block, re.MULTILINE))
            if class_statements == 0:
                issues.append(f"Block {i+1}: classDef defined but no class statements found")

        # Check for basic syntax issues
        if block.count('[') != block.count(']'):
            issues.append(f"Block {i+1}: Unmatched square brackets")

        if block.count('(') != block.count(')'):
            issues.append(f"Block {i+1}: Unmatched parentheses")

        if 'subgraph' in block:
            subgraph_count = block.count('subgraph')
            end_count = block.count('end')
            if subgraph_count != end_count:
                issues.append(f"Block {i+1}: Unmatched subgraph/end blocks ({subgraph_count} vs {end_count})")

    # Report results
    print("\n🔍 Validation Results:")
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No syntax issues found!")
        return True

if __name__ == "__main__":
    validate_mermaid_html("docs/diagrams.html")
