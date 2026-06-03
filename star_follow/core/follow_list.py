from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from star_follow.paths import follow_list_path as _follow_list_path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LIST_PATH = _follow_list_path()


@dataclass
class FollowEntry:
    name: str
    enabled: bool = True
    column_index: int | None = None


@dataclass
class FollowList:
    entries: list[FollowEntry] = field(default_factory=list)
    path: Path = field(default_factory=lambda: DEFAULT_LIST_PATH)

    def active(self) -> list[str]:
        return [e.name for e in self.entries if e.enabled]

    def active_entries(self) -> list[FollowEntry]:
        return [e for e in self.entries if e.enabled]

    def add(self, name: str) -> None:
        name = name.strip()
        if not name:
            return
        for e in self.entries:
            if e.name == name:
                return
        self.entries.append(FollowEntry(name=name))

    def remove(self, name: str) -> None:
        self.entries = [e for e in self.entries if e.name != name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        for e in self.entries:
            if e.name == name:
                e.enabled = enabled

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self.entries]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_names(cls, names: list[str]) -> FollowList:
        """用指定名單直接建立（掛房模式以啟動設定覆寫 follow_list 用）。"""
        fl = cls()
        seen: set[str] = set()
        for n in names:
            n = n.strip()
            if n and n not in seen:
                seen.add(n)
                fl.entries.append(FollowEntry(name=n))
        return fl

    @classmethod
    def load(cls, path: Path | None = None) -> FollowList:
        path = path or DEFAULT_LIST_PATH
        fl = cls(path=path)
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            fl.entries = [
                FollowEntry(
                    name=item["name"],
                    enabled=item.get("enabled", True),
                    column_index=item.get("column_index"),
                )
                for item in raw
            ]
        return fl
