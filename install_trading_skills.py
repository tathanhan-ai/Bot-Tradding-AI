#!/usr/bin/env python3
"""
1-Click Automated Trading Skills Installer & Portable Setup
Cai dat tu dong 68 Ky nang Giao dich Dinh luong (Trading Skills) cho moi moi truong:
- Google Antigravity / Gemini CLI (~/.gemini/skills & ~/.gemini/config/skills)
- Claude Code (~/.claude/skills)
- Cursor IDE (~/.cursor/skills)
- OpenAI Codex (~/.codex/skills)
- Local Project (.gemini/skills)

Hoat dong doc lap 100% tren Windows, Linux (Ubuntu/Debian) va macOS.
"""
import os
import sys
import shutil
from pathlib import Path

def main():
    print("==================================================================")
    print(" [PORTABLE QUANT SKILLS INSTALLER] 1-Click Setup cho Agent")
    print("==================================================================")
    
    script_dir = Path(__file__).resolve().parent
    local_skills_dir = script_dir / "skills"
    
    if not local_skills_dir.exists() or not local_skills_dir.is_dir():
        print(f" Loi: Khong tim thay thu muc ky nang tai: {local_skills_dir}")
        sys.exit(1)
        
    skills = [d for d in local_skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
    print(f" Tim thay {len(skills)} ky nang dinh luong chuan hoa trong du an.")

    home = Path.home()
    target_dirs = [
        home / ".gemini" / "config" / "skills",
        home / ".gemini" / "skills",
        home / ".claude" / "skills",
        home / ".cursor" / "skills",
        home / ".codex" / "skills",
        script_dir / ".gemini" / "skills"
    ]

    installed_count = 0
    target_locations = []

    for target in target_dirs:
        try:
            target.mkdir(parents=True, exist_ok=True)
            for skill_path in skills:
                dest = target / skill_path.name
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(skill_path, dest)
            installed_count = len(skills)
            target_locations.append(str(target))
            print(f" Da cai dat thanh cong {len(skills)} skills vao: {target}")
        except Exception as e:
            pass

    print("------------------------------------------------------------------")
    if target_locations:
        print(f" HOAN TAT CAI DAT! {installed_count} Ky Nang Dinh Luong da san sang hoat dong.")
        print("Cac duong dan da kich hoat:")
        for loc in target_locations:
            print(f"   {loc}")
    else:
        print(" Khong the ghi vao thu muc he thong, cac ky nang van kha dung cuc bo trong du an.")
    print("==================================================================")

if __name__ == "__main__":
    main()
