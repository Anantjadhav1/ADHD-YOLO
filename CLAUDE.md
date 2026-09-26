# How to work with me on this project

- Do NOT edit files yourself unless I say "apply it". For each change give me:
  1. Problem (short)
  2. Solution (short)
  3. Exact file + line number, the exact text to find, and the exact replacement
- One step at a time. Wait for my output before giving the next step.
- I'm on Windows PowerShell. Use `py -m ...`, never `python` or `python3`.
- After every Python edit, tell me to run `py -m py_compile <file>`.
- Never change validation to raise accuracy: subject-wise splits and the
  §6R three-way split stay exactly as they are.
- Read PROGRESS.md and README.md before suggesting anything.
- You can reply in Hinglish.

## Format for every change
- File: <full path>
- Find: <exact existing text, in a code block>
- Replace with: <exact new text, in a code block>
- For a NEW file: say "New file" and give the full content.
- After each step works, give me the exact git commands:
  git add <exact file names, correct case>
  git commit -m "<message>"
  git push