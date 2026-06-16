from app.database import db_session, utc_now


DEFAULT_PROFILE = {
    "id": 1,
    "username": "cognix-user",
    "display_name": "Cognix User",
    "theme": "light",
    "default_answer_style": "memo",
    "raw_data_note": "",
}


def get_profile() -> dict:
    with db_session() as conn:
        ensure_profile_columns(conn)
        row = conn.execute("SELECT * FROM user_preferences WHERE id=1").fetchone()
        if row:
            return row
        now = utc_now()
        conn.execute(
            """
            INSERT INTO user_preferences
            (id, username, display_name, theme, default_answer_style, raw_data_note, created_at, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                DEFAULT_PROFILE["username"],
                DEFAULT_PROFILE["display_name"],
                DEFAULT_PROFILE["theme"],
                DEFAULT_PROFILE["default_answer_style"],
                DEFAULT_PROFILE["raw_data_note"],
                now,
                now,
            ),
        )
        return conn.execute("SELECT * FROM user_preferences WHERE id=1").fetchone()


def update_profile(payload: dict) -> dict:
    current = get_profile()
    display_name = str(payload.get("display_name", current["display_name"])).strip() or "Cognix User"
    username_value = payload.get("username") or display_name
    username = str(username_value).strip()
    username = slug_username(username or display_name)
    theme = payload.get("theme", current["theme"])
    if theme not in {"light", "dark"}:
        theme = current["theme"]
    default_answer_style = payload.get("default_answer_style", current["default_answer_style"])
    if default_answer_style not in {"brief", "memo", "deep"}:
        default_answer_style = current["default_answer_style"]
    raw_data_note = str(payload.get("raw_data_note", current["raw_data_note"]))

    with db_session() as conn:
        ensure_profile_columns(conn)
        conn.execute(
            """
            UPDATE user_preferences
            SET username=?, display_name=?, theme=?, default_answer_style=?, raw_data_note=?, updated_at=?
            WHERE id=1
            """,
            (username, display_name, theme, default_answer_style, raw_data_note, utc_now()),
        )
        return conn.execute("SELECT * FROM user_preferences WHERE id=1").fetchone()


def ensure_profile_columns(conn) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(user_preferences)").fetchall()}
    if "username" not in columns:
        conn.execute("ALTER TABLE user_preferences ADD COLUMN username TEXT NOT NULL DEFAULT 'cognix-user'")


def slug_username(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "cognix-user"
